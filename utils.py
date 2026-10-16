"""Small helpers with NO heavy dependencies (no torch), so they are easy to test."""
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def list_images(dataset_dir: Path, extensions: Iterable[str]) -> List[Path]:
    """Recursively list image files in a stable, sorted order."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    exts = tuple(e.lower() for e in extensions)
    return sorted(p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization so that a dot product equals cosine similarity."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def display_results(results: List[dict], query: str) -> None:
    """Show results with matplotlib. Imported lazily."""
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4.5), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_title(f"#{r['rank']}  {Path(r['filename']).name}\nscore={r['score']:.3f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f'Query: "{query}"', fontsize=14)
    plt.tight_layout()
    plt.show()