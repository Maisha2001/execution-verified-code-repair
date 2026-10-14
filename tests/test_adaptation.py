import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repairbench.comparison import ADAPTED_ARMS, compare, compare_task, summary
from repairbench.data import Task, digest
from repairbench.inference import CodeModel, adapter_identity


def test_adapter_rejects_base_mismatch_and_changed_weights(tmp_path):
    files = {'adapter_config.json': {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'bias': 'none'},
             'state.json': {'step': 3, 'configuration': {'base_model': {'id': 'base', 'revision': 'pinned'},
                                                        'data_manifest_sha256': 'data'}}}
    for name, value in files.items():
        (tmp_path/name).write_text(json.dumps(value))
    for name in ['adapter_model.safetensors', 'optimizer.safetensors', 'rng.safetensors']:
        (tmp_path/name).write_bytes(b'fixture')
    (tmp_path/'checksums.json').write_text(json.dumps({p.name: digest(p.read_bytes()) for p in tmp_path.iterdir()}))
    record = adapter_identity(tmp_path, {'id': 'base', 'revision': 'pinned'})
    assert record['step'] == 3 and record['adapter_sha256'] == digest(b'fixture')
    with pytest.raises(ValueError, match='base model'):
        adapter_identity(tmp_path, {'id': 'base', 'revision': 'other'})
    (tmp_path/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        adapter_identity(tmp_path, {'id': 'base', 'revision': 'pinned'})


def test_all_arms_share_seeds_budgets_and_visible_context(monkeypatch):
    task = Task('sample', 'def f(x): return x\n', {'python_programs/sample.py': b'def f(x): return x\n'})
    monkeypatch.setattr('repairbench.comparison.load_task', lambda *a, **k: task)
    monkeypatch.setattr('repairbench.comparison.propose', lambda *a: [])
    calls = []
    class Generator:
        def generate(self, passed, context, seed, sampled, *, adapter=False):
            calls.append((context, seed, sampled, adapter))
            return 'invalid', {'seconds': 0.1}
    class Runner:
        def run(self, files, test, reference=False):
            return {'status': 'passed' if reference else 'failed', 'evidence': {'passed': 1, 'skipped': 0}}
    result = compare_task(Path('.'), 'sample', Runner(), Generator(),
                          SimpleNamespace(retrieve=lambda t: ['visible']), 4, ADAPTED_ARMS)
    assert len(calls) == 16
    assert [v[1:3] for v in calls[:4]] == [v[1:3] for v in calls[8:12]]
    assert [v[1:3] for v in calls[4:8]] == [v[1:3] for v in calls[12:]]
    assert [v[3] for v in calls] == [False]*8 + [True]*8
    assert [bool(v[0]) for v in calls] == [False]*4 + [True]*4 + [False]*4 + [True]*4
    assert all(len(result['arms'][a]['attempts']) == 4 for a in ADAPTED_ARMS[1:])


def test_paired_report_retains_wins_losses_and_common_exclusions():
    rows = []
    for name, wins in [('a', ['adapted']), ('b', ['unadapted']), ('c', ['adapted', 'unadapted']), ('d', [])]:
        arms = {arm: {'attempts': [], 'accepted_candidate': {'number': 1, 'patch': '-old\n+new'} if arm in wins else None}
                for arm in ADAPTED_ARMS}
        rows.append({'task': name, 'eligibility': 'eligible', 'arms': arms})
    rows.append({'task': 'excluded', 'eligibility': 'reference_not_passing', 'arms': {}})
    report = summary(rows, 4, 10, ADAPTED_ARMS)
    pair = report['paired']['adapted_versus_unadapted']
    assert pair == {'wins': ['a'], 'losses': ['b'], 'both': ['c'], 'neither': ['d'],
                    'net_resolved': 0, 'resolution_rate_difference': 0}
    assert report['eligible_tasks'] == 4 and report['arms']['adapted']['resolved'] == 2


def test_resume_skips_verified_tasks_and_rejects_configuration_or_tampering(tmp_path, monkeypatch):
    from repairbench import comparison
    import itertools
    clock = itertools.count()
    monkeypatch.setattr(comparison.time, 'perf_counter', lambda: next(clock))
    task = Task('a', 'def f(): return 1\n', {})
    monkeypatch.setattr(comparison, 'verify', lambda p: {'tasks': ['a', 'b'], 'revision': 'r', 'archive_sha256': 'd'})
    monkeypatch.setattr(comparison, 'load_task', lambda *a: task)
    monkeypatch.setattr(comparison.importlib.metadata, 'version', lambda n: '1')
    calls = []
    def execute(root, name, *args):
        calls.append(name)
        return {'task': name, 'source_sha256': digest(task.source.encode()), 'eligibility': 'reference_not_passing', 'arms': {}}
    monkeypatch.setattr(comparison, 'compare_task', execute)
    generator = SimpleNamespace(record={}, adapter_record=None, device='cpu', max_new_tokens=768)
    retriever = SimpleNamespace(record={})
    runner = SimpleNamespace(preflight=lambda: {'image_id': 'pinned'})
    out = tmp_path/'run'
    first = compare(tmp_path, out, runner, generator, retriever, max_seconds=3)
    assert not first['completed'] and first['tasks'] == 1 and calls == ['a']
    second = compare(tmp_path, out, runner, generator, retriever, resume=True)
    third = compare(tmp_path, out, runner, generator, retriever, resume=True)
    assert second['completed'] and calls == ['a', 'b']
    assert second['seconds'] == third['seconds'] == 2 * first['seconds']
    with pytest.raises(ValueError, match='configuration'):
        compare(tmp_path, out, runner, generator, retriever, budget=3, resume=True)
    path = out/'tasks/a.json'
    path.write_text(path.read_text().replace('reference_not_passing', 'tampered'))
    with pytest.raises(ValueError, match='task changed'):
        compare(tmp_path, out, runner, generator, retriever, resume=True)


def test_actual_lora_generation_preserves_base_outputs_and_restores_adapter():
    torch = pytest.importorskip('torch')
    peft = pytest.importorskip('peft')
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(17)
    torch.set_num_threads(1)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2)).eval()
    class Tokens(dict):
        @property
        def input_ids(self): return self['input_ids']
        def to(self, device): return self
    class Tokenizer:
        eos_token_id = 2
        def apply_chat_template(self, *a, **k): return 'same prompt'
        def __call__(self, *a, **k): return Tokens(input_ids=torch.tensor([[1, 3, 4]]), attention_mask=torch.ones((1, 3), dtype=torch.long))
        def decode(self, values, **k): return ','.join(str(v) for v in values.tolist())
    engine = CodeModel.__new__(CodeModel)
    engine.torch, engine.device, engine.max_new_tokens = torch, 'cpu', 4
    engine.model, engine.tokenizer, engine.adapter_record = base, Tokenizer(), None
    task = Task('x', 'def f(): return 1', {})
    original, _ = engine.generate(task, [], 9, False)
    engine.model = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name, parameter in engine.model.named_parameters():
            if 'lora_' in name: parameter.normal_(0, 0.7)
    engine.adapter_record = {'verified': True}
    before, _ = engine.generate(task, [], 9, False)
    adapted, evidence = engine.generate(task, [], 9, False, adapter=True)
    after, _ = engine.generate(task, [], 9, False)
    assert original == before == after
    assert evidence['adapter_enabled'] and adapted
    assert engine.model.get_model_status().enabled is True
    engine.adapter_record = None
    with pytest.raises(ValueError, match='requires'):
        engine.generate(task, [], 9, False, adapter=True)
