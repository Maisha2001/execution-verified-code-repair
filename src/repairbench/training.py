import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .checkpoints import restore_checkpoint, save_checkpoint
from .data import digest
from .models import offline_environment, verify_model
from .training_data import load_examples, write_json


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    epochs: int = 3
    accumulation: int = 4
    learning_rate: float = 0.0002
    rank: int = 8
    max_tokens: int = 1024
    checkpoint_steps: int = 20

    def __post_init__(self):
        if not (1 <= self.epochs <= 20 and 1 <= self.accumulation <= 32 and 1 <= self.rank <= 64
                and 128 <= self.max_tokens <= 4096 and 1 <= self.checkpoint_steps <= 1000
                and 0 < self.learning_rate <= 0.01):
            raise ValueError('Invalid training budget')


def encode_example(tokenizer, row, max_tokens):
    prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
    prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
    full = tokenizer.apply_chat_template(row['messages'] + [{'role': 'assistant', 'content': row['target']}], tokenize=False)
    tokens = tokenizer(full, add_special_tokens=False)['input_ids']
    if tokens[:len(prefix)] != prefix:
        raise ValueError('Chat template prefix mismatch')
    if len(tokens) > max_tokens:
        return None
    if len(tokens) <= len(prefix):
        raise ValueError('Empty training target')
    return {'input_ids': tokens, 'labels': [-100] * len(prefix) + tokens[len(prefix):], 'id': row['id'], 'task_id': row['task_id']}


def training_order(count, epochs, seed):
    order = []
    for epoch in range(epochs):
        indices = list(range(count))
        random.Random(seed + epoch).shuffle(indices)
        order.extend(indices)
    return order


def tensor_batch(row, device):
    import torch
    return {k: torch.tensor([row[k]], dtype=torch.long, device=device) for k in ('input_ids', 'labels')}


def validation_loss(model, rows, device):
    import torch
    model.eval()
    total, tokens = 0., 0
    with torch.inference_mode():
        for row in rows:
            batch = tensor_batch(row, device)
            count = int((batch['labels'][:, 1:] != -100).sum())
            loss = float(model(**batch, use_cache=False).loss)
            if not math.isfinite(loss):
                raise ValueError('Nonfinite validation loss')
            total += count * loss
            tokens += count
    model.train()
    return {'loss': total / tokens, 'target_tokens': tokens, 'examples': len(rows)}


def train(models, data, output, settings=TrainingConfig(), *, resume=None, stop_after_steps=None, max_seconds=7200):
    offline_environment()
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('This measured training recipe requires CUDA; no automatic device fallback')
    if max_seconds <= 0 or (stop_after_steps is not None and stop_after_steps < 1):
        raise ValueError('Invalid interruption budget')
    invocation_started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    path, model_record = verify_model(models, 'code')
    rows, manifest = load_examples(data)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    encoded, rejected = {}, {}
    for split, values in rows.items():
        pairs = [(r, encode_example(tokenizer, r, settings.max_tokens)) for r in values]
        encoded[split] = [e for _, e in pairs if e is not None]
        rejected[split] = [r['id'] for r, e in pairs if e is None]
        if not encoded[split]:
            raise ValueError('No examples within the token budget')
    configuration = {'settings': asdict(settings), 'data_manifest_sha256': digest((data / 'manifest.json').read_bytes()),
        'base_model': model_record, 'torch': torch.__version__, 'peft': peft.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(), 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'cuda': torch.version.cuda, 'precision': 'bfloat16 base, float32 adapters; no quantization',
        'targets': ['q_proj', 'v_proj'], 'rejected_overlength': rejected,
        'examples': {s: len(v) for s, v in encoded.items()}, 'task_ids': {s: sorted({v['task_id'] for v in values}) for s, values in encoded.items()},
        'source_hashes': {n: digest(Path(__file__).with_name(n).read_text(encoding='utf-8').encode()) for n in ('training.py', 'training_data.py', 'checkpoints.py', 'models.py')}}
    if output.exists():
        if resume is None or json.loads((output / 'configuration.json').read_text()) != configuration:
            raise ValueError('Existing run requires a matching resume checkpoint')
        complete_checkpoints = sorted(p for p in output.glob('checkpoint-*') if p.is_dir() and (p / 'checksums.json').is_file() and not p.name.endswith('.partial'))
        if not complete_checkpoints or resume.resolve() != complete_checkpoints[-1].resolve():
            raise ValueError('Resume from the latest complete checkpoint; do not rewind an existing run')
    else:
        if resume is not None:
            raise ValueError('Resume in the original run directory')
        output.mkdir(parents=True)
        write_json(output / 'configuration.json', configuration)
    base = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                                               dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=settings.rank, lora_alpha=2*settings.rank,
                                          target_modules=configuration['targets'], lora_dropout=0., bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    if any('lora_' not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise ValueError('Unexpected trainable base parameter')
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate, weight_decay=0.01)
    state = {'step': 0, 'consumed_examples': 0, 'train_history': [], 'validation_history': [], 'seconds': 0.,
             'trainable_parameters': sum(p.numel() for p in trainable), 'total_parameters': sum(p.numel() for p in model.parameters()),
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'peak_gpu_allocated_bytes': 0}
    if resume is not None:
        if resume.parent.resolve() != output.resolve():
            raise ValueError('Checkpoint must belong to this run')
        state = restore_checkpoint(resume, model, optimizer, configuration)
        for key in ('configuration', 'optimizer_groups', 'trainable_names'):
            state.pop(key, None)
    torch.cuda.reset_peak_memory_stats()
    if not state['validation_history']:
        initial = validation_loss(model, encoded['validation'], 'cuda')
        state['validation_history'].append({'step': 0, **initial})
        print(json.dumps({'initial_validation': initial}), flush=True)
    order = training_order(len(encoded['train']), settings.epochs, settings.seed)
    total_steps = math.ceil(len(order) / settings.accumulation)
    resumed_seconds = state['seconds']
    model.train()
    while state['consumed_examples'] < len(order):
        begin = state['consumed_examples']
        indices = order[begin:begin + settings.accumulation]
        next_step = state['step'] + 1
        warmup = min(5, total_steps)
        multiplier = min(next_step / warmup, max(0., (total_steps-next_step+1) / max(1, total_steps-warmup)))
        for group in optimizer.param_groups:
            group['lr'] = settings.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in indices:
            batch = tensor_batch(encoded['train'][index], 'cuda')
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; last complete checkpoint retained')
            losses.append(float(loss.detach()))
            (loss / len(indices)).backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        state['step'] = next_step
        state['consumed_examples'] += len(indices)
        state['train_history'].append({'step': next_step, 'loss': sum(losses)/len(losses), 'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm': float(norm)})
        stopped = (stop_after_steps is not None and next_step >= stop_after_steps) or time.perf_counter()-invocation_started >= max_seconds
        complete = state['consumed_examples'] == len(order)
        if next_step % settings.checkpoint_steps == 0 or stopped or complete:
            measured = validation_loss(model, encoded['validation'], 'cuda')
            state['validation_history'].append({'step': next_step, **measured})
            state['seconds'] = resumed_seconds + time.perf_counter()-invocation_started
            state['peak_gpu_allocated_bytes'] = max(state['peak_gpu_allocated_bytes'], torch.cuda.max_memory_allocated())
            state['completed'] = complete
            state['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
            checkpoint = output / f'checkpoint-{next_step:06d}'
            save_checkpoint(checkpoint, model, optimizer, state, configuration)
            best = min(state['validation_history'][1:], key=lambda v: (v['loss'], v['step']))
            write_json(output / 'summary.json', {**state, 'best_checkpoint': f"checkpoint-{best['step']:06d}", 'last_checkpoint': checkpoint.name,
                                                'held_out_test_used': False, 'total_steps': total_steps})
            print(json.dumps({'step': next_step, 'total_steps': total_steps, 'validation': measured, 'peak_gpu_allocated_bytes': state['peak_gpu_allocated_bytes'], 'completed': complete}), flush=True)
        else:
            print(json.dumps({'step': next_step, 'total_steps': total_steps, 'loss': state['train_history'][-1]['loss']}), flush=True)
        if stopped or complete:
            break
    return json.loads((output / 'summary.json').read_text())
