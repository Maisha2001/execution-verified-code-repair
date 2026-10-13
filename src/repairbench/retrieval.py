from dataclasses import asdict, dataclass

from .data import Task, digest, safe_path
from .models import offline_environment, verify_model


@dataclass(frozen=True)
class Chunk:
    path: str
    start_line: int
    text: str

    def evidence(self, score: float) -> dict:
        return {**asdict(self), "sha256": digest(self.text.encode()), "score": float(score)}


def context_chunks(task: Task) -> list[Chunk]:
    allowed = {f"python_testcases/test_{task.name}.py", f"json_testcases/{task.name}.json", "python_programs/node.py", "python_testcases/node.py", "python_testcases/load_testdata.py"}
    chunks = []
    for path, raw in sorted(task.files.items()):
        if path not in allowed or not safe_path(path):
            continue
        text = raw.decode("utf-8")
        # Long fixture lines are bounded before embedding, as are ordinary lines.
        lines = text.splitlines()
        for start in range(0, len(lines), 16):
            block = "\n".join(lines[start:start + 20])
            for offset in range(0, len(block), 1200):
                chunks.append(Chunk(path, start + 1, block[offset:offset + 1200]))
    return chunks


class ContextRetriever:
    def __init__(self, models):
        offline_environment()
        from sentence_transformers import SentenceTransformer

        path, self.record = verify_model(models, "embedding")
        self.encoder = SentenceTransformer(str(path), device="cpu", local_files_only=True, trust_remote_code=False, model_kwargs={"use_safetensors": True})
        self.encoder.max_seq_length = 256

    def retrieve(self, task: Task, k: int = 3) -> list[dict]:
        if not 1 <= k <= 8:
            raise ValueError("Context budget must be between one and eight chunks")
        import faiss
        import numpy as np

        chunks = context_chunks(task)
        if not chunks:
            return []
        corpus = [f"{c.path}\n{c.text}" for c in chunks]
        vectors = np.asarray(self.encoder.encode(corpus, normalize_embeddings=True, show_progress_bar=False), dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        query = self.encoder.encode([f"Repair {task.name}\n{task.source}"], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        scores, ids = index.search(query, min(k, len(chunks)))
        return [chunks[int(i)].evidence(score) for i, score in zip(ids[0], scores[0])]
