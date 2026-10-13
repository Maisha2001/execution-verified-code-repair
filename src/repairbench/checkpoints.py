import json
import re

from .data import digest
from .training_data import write_json


def save_checkpoint(directory, model, optimizer, state, configuration):
    import torch
    from safetensors.torch import save_file

    if directory.exists():
        raise ValueError("Checkpoint already exists")
    temporary = directory.with_name(directory.name + '.partial')
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temporary, safe_serialization=True)
    values = optimizer.state_dict()
    tensors = {f'{identity}.{name}': value.detach().cpu().contiguous() for identity, fields in values['state'].items()
               for name, value in fields.items()}
    save_file(tensors, str(temporary / 'optimizer.safetensors'))
    rng = {'cpu': torch.get_rng_state()}
    if torch.cuda.is_available():
        rng.update({f'cuda_{i}': value for i, value in enumerate(torch.cuda.get_rng_state_all())})
    save_file(rng, str(temporary / 'rng.safetensors'))
    write_json(temporary / 'state.json', {**state, 'configuration': configuration,
        'optimizer_groups': values['param_groups'], 'trainable_names': [n for n, p in model.named_parameters() if p.requires_grad]})
    inventory = {p.name: digest(p.read_bytes()) for p in temporary.iterdir() if p.is_file()}
    write_json(temporary / 'checksums.json', inventory)
    temporary.rename(directory)


def verify_checkpoint(directory):
    inventory = json.loads((directory / 'checksums.json').read_text())
    required = {'adapter_config.json', 'adapter_model.safetensors', 'optimizer.safetensors', 'rng.safetensors', 'state.json'}
    if not required <= inventory.keys() or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', n) for n in inventory):
        raise ValueError('Invalid checkpoint inventory')
    for name, expected in inventory.items():
        path = directory / name
        if path.is_symlink() or digest(path.read_bytes()) != expected:
            raise ValueError('Checkpoint checksum mismatch')
    return json.loads((directory / 'state.json').read_text())


def restore_checkpoint(directory, model, optimizer, configuration):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = verify_checkpoint(directory)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if state['configuration'] != configuration or names != state['trainable_names']:
        raise ValueError('Checkpoint configuration or parameter names changed')
    set_peft_model_state_dict(model, load_file(str(directory / 'adapter_model.safetensors')))
    # Own the buffers so Windows does not retain a mapped checkpoint-file handle.
    tensors = {key: value.clone() for key, value in load_file(str(directory / 'optimizer.safetensors')).items()}
    values = {}
    for key, value in tensors.items():
        identity, field = key.split('.', 1)
        values.setdefault(int(identity), {})[field] = value
    optimizer.load_state_dict({'state': values, 'param_groups': state['optimizer_groups']})
    rng = load_file(str(directory / 'rng.safetensors'))
    torch.set_rng_state(rng['cpu'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng[f'cuda_{i}'] for i in range(torch.cuda.device_count())])
    return state
