import json
import os
from pathlib import Path

import pytest

from repairbench.data import digest
from repairbench.training import TrainingConfig, encode_example, training_order
from repairbench.training_data import MBPP_REVISION, MBPP_SHA256, fingerprint, load_examples, official_split, similarity, split_controls, task_payload, write_json


def test_official_partitions_and_invalid_ids():
    assert [official_split(i) for i in (1, 10, 11, 510, 511, 600, 601, 974)] == ['prompt','prompt','test','test','validation','validation','train','train']
    with pytest.raises(ValueError):
        official_split(975)


def test_clone_guard_prioritizes_holdout_and_excludes_benchmark():
    code = 'def f(x):\n    return sorted(x)\n'
    renamed = 'def different(y):\n    return sorted(y)\n'
    assert similarity(fingerprint(code), fingerprint(renamed)) == 1
    rows = [{'task_id': i, 'code': s} for i, s in [(601, code), (511, renamed), (11,code), (602,'def g(n):\n    return n ** 2\n')]]
    controls = split_controls(rows, ['def square(a):\n    return a ** 2\n'])
    assert controls['partitions'] == {'train': [], 'validation': [], 'test': [11]}
    assert controls['excluded']['601']['reason'] == 'duplicate_or_near_clone'
    assert controls['excluded']['602']['reason'] == 'quixbugs_clone'


def test_training_loader_detects_tampering_and_refuses_test_ids(tmp_path):
    config = {'mbpp_revision': MBPP_REVISION, 'mbpp_sha256': MBPP_SHA256, 'controls': {'partitions': {'train': [601], 'validation': [511]}}}
    write_json(tmp_path/'configuration.json', config)
    for split, i in [('train',601),('validation',511)]:
        (tmp_path/f'{split}.jsonl').write_text(json.dumps({'split':split,'task_id':i})+'\n')
    manifest = {'configuration_sha256': digest((tmp_path/'configuration.json').read_bytes()), 'files': {s+'.jsonl': digest((tmp_path/f'{s}.jsonl').read_bytes()) for s in ('train','validation')}}
    write_json(tmp_path/'manifest.json', manifest)
    # The loader does not need or open the test partition.
    (tmp_path/'test.jsonl').write_text('must not parse')
    assert load_examples(tmp_path)[0]['train'][0]['task_id'] == 601
    (tmp_path/'train.jsonl').write_text(json.dumps({'split':'train','task_id':11})+'\n')
    with pytest.raises(ValueError,match='checksum'):
        load_examples(tmp_path)
    manifest['files']['train.jsonl'] = digest((tmp_path/'train.jsonl').read_bytes())
    write_json(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError,match='contamination'):
        load_examples(tmp_path)


def test_prompt_tokens_are_masked_and_long_examples_rejected():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return ''.join(m['content'] for m in messages) + ('A' if add_generation_prompt else '')
        def __call__(self, text, add_special_tokens=False): return {'input_ids':list(text.encode())}
    row = {'id':'x','task_id':601,'messages':[{'role':'user','content':'prompt'}],'target':'Answer'}
    result = encode_example(Tokenizer(), row, 30)
    assert result['labels'][:7] == [-100]*7
    assert result['labels'][7:] == list(b'nswer')
    assert encode_example(Tokenizer(), row, 5) is None


def test_training_order_can_resume_without_reusing_seen_examples():
    order = training_order(11,3,17)
    assert len(order)==33 and all(sorted(order[i:i+11])==list(range(11)) for i in (0,11,22))
    assert order[:8] + training_order(11,3,17)[8:] == order
    assert order != training_order(11,3,18)
    with pytest.raises(ValueError): TrainingConfig(epochs=0)


def test_task_builder_never_executes_source(tmp_path):
    marker=tmp_path/'not-created'
    code=f'from pathlib import Path\nPath({str(marker)!r}).touch()\ndef f(): return 1'
    row={'test_imports':[],'test_list':['assert f() == 1','assert f() != 0']}
    task=task_payload(row,code)
    assert not marker.exists()
    assert task.files['test_task.py'].count(b'def test_case_')==2


@pytest.mark.docker
def test_real_mutation_and_reference_in_isolated_runner():
    image=os.environ.get('REPAIR_IMAGE_FILE')
    if not image: pytest.skip('Set REPAIR_IMAGE_FILE for isolated execution checks')
    from repairbench.sandbox import DockerRunner
    runner=DockerRunner(Path(image).read_text().strip())
    row={'test_imports':[],'test_list':['assert f(2)==3','assert f(0)==1']}
    reference=task_payload(row,'def f(x): return x+1')
    mutated=task_payload(row,'def f(x): return x-1')
    good=runner.run(reference.files,'test_task.py')
    bad=runner.run(mutated.files,'test_task.py')
    assert good['status']=='passed' and good['evidence']['passed']==2
    assert bad['status']=='failed' and bad['evidence']['failed']==2


@pytest.mark.docker
def test_dataset_names_and_imports_do_not_confuse_pytest_collection():
    image=os.environ.get('REPAIR_IMAGE_FILE')
    if not image: pytest.skip('Set REPAIR_IMAGE_FILE for isolated execution checks')
    from repairbench.sandbox import DockerRunner
    runner=DockerRunner(Path(image).read_text().strip())
    row={'test_imports':['import math'],'test_list':['assert _private(4)==2','assert test_helper(4)==3']}
    source='def _private(x): return math.sqrt(x)\ndef test_helper(x): return _private(x)+1'
    task=task_payload(row,source)
    result=runner.run(task.files,'test_task.py')
    assert result['status']=='passed',result
    assert result['evidence']['passed']==2 and result['evidence']['errors']==0


def test_lora_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    torch=pytest.importorskip('torch')
    pytest.importorskip('peft')
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from repairbench.checkpoints import save_checkpoint, restore_checkpoint, verify_checkpoint
    torch.set_num_threads(1)
    def setup():
        torch.manual_seed(27)
        base=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
        model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=0.1,target_modules=['q_proj','v_proj']))
        return model,torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.01)
    inputs=torch.tensor([[1,5,8,3,2]])
    def step(model,optimizer):
        optimizer.zero_grad();loss=model(input_ids=inputs,labels=inputs).loss
        loss.backward();optimizer.step()
        return float(loss.detach())
    model,opt=setup();step(model,opt)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    save_checkpoint(tmp_path/'checkpoint',model,opt,{'step':1},{'seed':27})
    expected_loss=step(model,opt)
    expected={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    recovered,optimizer=setup()
    state=restore_checkpoint(tmp_path/'checkpoint',recovered,optimizer,{'seed':27})
    assert state['step']==1 and step(recovered,optimizer)==expected_loss
    for n,p in recovered.named_parameters():
        assert torch.equal(p.detach(), expected[n] if p.requires_grad else frozen[n])
    with pytest.raises(ValueError,match='configuration'):
        restore_checkpoint(tmp_path/'checkpoint',recovered,optimizer,{'seed':28})
    with pytest.raises(ValueError,match='already exists'):
        save_checkpoint(tmp_path/'checkpoint',model,opt,{}, {})
    path=tmp_path/'checkpoint'/'optimizer.safetensors'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):
        verify_checkpoint(tmp_path/'checkpoint')
