import os
import hashlib
from pathlib import Path
import pickle
import threading

from earnings_qa.config import WORKSPACE_ROOT

CACHE_DIR = WORKSPACE_ROOT / ".earnings_qa_cache"


class CacheManager:
    """Manages persistent caching for embeddings, retrieval, and final answers."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_cache_file = self.cache_dir / "embeddings.pkl"
        self.retrieval_cache_file = self.cache_dir / "retrieval.pkl"
        self.answers_cache_file = self.cache_dir / "answers.pkl"

        self._lock = threading.Lock()

        self.embeddings = self._load(self.embeddings_cache_file)
        self.retrieval = self._load(self.retrieval_cache_file)
        self.answers = self._load(self.answers_cache_file)

    def _load(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _save(self, path: Path, data: dict):
        with self._lock:
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
            temp_path.replace(path)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def get_embedding(self, text: str):
        return self.embeddings.get(self._hash(text))

    def set_embedding(self, text: str, embedding):
        self.embeddings[self._hash(text)] = embedding
        self._save(self.embeddings_cache_file, self.embeddings)

    def get_retrieval(self, query: str, company_filter, quarter_filter, index_version: str):
        key = f"{self._hash(query)}_{company_filter}_{quarter_filter}_{index_version}"
        return self.retrieval.get(key)

    def set_retrieval(self, query: str, company_filter, quarter_filter, index_version: str, results):
        key = f"{self._hash(query)}_{company_filter}_{quarter_filter}_{index_version}"
        self.retrieval[key] = results
        self._save(self.retrieval_cache_file, self.retrieval)

    def get_answer(self, query: str, company_filter, quarter_filter, index_version: str, prompt_version: str, history_hash: str):
        key = f"{self._hash(query)}_{company_filter}_{quarter_filter}_{index_version}_{prompt_version}_{history_hash}"
        return self.answers.get(key)

    def set_answer(self, query: str, company_filter, quarter_filter, index_version: str, prompt_version: str, history_hash: str, answer: dict):
        key = f"{self._hash(query)}_{company_filter}_{quarter_filter}_{index_version}_{prompt_version}_{history_hash}"
        self.answers[key] = answer
        self._save(self.answers_cache_file, self.answers)

    def clear(self):
        with self._lock:
            self.embeddings = {}
            self.retrieval = {}
            self.answers = {}
        self._save(self.embeddings_cache_file, self.embeddings)
        self._save(self.retrieval_cache_file, self.retrieval)
        self._save(self.answers_cache_file, self.answers)


# Global singleton
cache = CacheManager()
