import hashlib
import json
import os
from pathlib import Path

MODELS = {
    "code": {
        "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
        "license": "Apache-2.0",
        "files": ["config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE", "README.md"],
    },
    "embedding": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "license": "Apache-2.0",
        "files": ["config.json", "config_sentence_transformers.json", "model.safetensors", "modules.json", "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json", "README.md"],
    },
}


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def offline_environment():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def download_models(root: Path):
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download

    for key, spec in MODELS.items():
        destination = root / key
        for name in spec["files"]:
            hf_hub_download(spec["id"], name, revision=spec["revision"], local_dir=destination, token=False)
        record = {**spec, "sha256": {name: file_digest(destination / name) for name in spec["files"]}}
        (destination / "verified-model.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def verify_model(root: Path, key: str) -> tuple[Path, dict]:
    path = root / key
    record = json.loads((path / "verified-model.json").read_text(encoding="utf-8"))
    if any(record.get(k) != v for k, v in MODELS[key].items()) or set(record["sha256"]) != set(MODELS[key]["files"]):
        raise ValueError("Model identity differs from the pinned configuration")
    for name, expected in record["sha256"].items():
        if file_digest(path / name) != expected:
            raise ValueError(f"Model file changed: {key}/{name}")
    return path, record
