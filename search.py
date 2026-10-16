"""Core search logic. Depends only on numpy and on ANY object with an
`encode_text(str) -> (1, D) array` method, so it is easy to test and reuse."""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, encoder, embeddings: np.ndarray, filenames: List[str],
                 dataset_dir: Path, max_top_k: int = 50):
        if embeddings.ndim != 2 or embeddings.shape[0] != len(filenames):
            raise ValueError("embeddings must be (N, D) and match the number of filenames")
        self.encoder = encoder
        self.embeddings = embeddings
        self.filenames = filenames
        self.dataset_dir = Path(dataset_dir)
        self.max_top_k = max_top_k
        self._basename_map: Optional[Dict[str, Path]] = None  # lazy, for legacy Week 4 names

    def _resolve_path(self, name: str) -> Path:
        direct = self.dataset_dir / name
        if direct.exists():
            return direct
        if self._basename_map is None:  # legacy index stored basenames only
            self._basename_map = {p.name: p for p in self.dataset_dir.rglob("*") if p.is_file()}
        return self._basename_map.get(Path(name).name, direct)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Return the top_k most similar images as JSON-friendly dicts:
        {rank, filename, path, score}, best first."""
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must be a non-empty string")

        top_k = max(1, min(int(top_k), self.max_top_k, len(self.filenames)))
        q = self.encoder.encode_text(query)[0]
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError(f"Dimension mismatch: text={q.shape[0]} vs images={self.embeddings.shape[1]}. "
                             "Wrong model for this index?")

        scores = self.embeddings @ q                         # cosine similarity (both unit-norm)
        idx = np.argpartition(-scores, top_k - 1)[:top_k]    # O(N) top-k selection
        idx = idx[np.argsort(-scores[idx])]                  # sort only the k winners

        return [
            {"rank": r + 1, "filename": self.filenames[i],
             "path": str(self._resolve_path(self.filenames[i])), "score": float(scores[i])}
            for r, i in enumerate(idx)
        ]