"""Build, save and load the image-embedding index."""
import json
import logging
import pickle
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import Config
from utils import l2_normalize, list_images

logger = logging.getLogger(__name__)


def build_index(encoder, cfg: Config) -> Tuple[np.ndarray, List[str]]:
    """Embed every image in cfg.dataset_dir. Filenames are stored RELATIVE to the dataset
    dir (posix style), so images in different subfolders with the same name don't collide.
    Unreadable images are skipped and logged, keeping embeddings and names aligned."""
    paths = list_images(cfg.dataset_dir, cfg.image_extensions)
    logger.info("Found %d images in %s", len(paths), cfg.dataset_dir)

    chunks, names, failed = [], [], []
    for i in tqdm(range(0, len(paths), cfg.batch_size), desc="Embedding"):
        imgs, ok = [], []
        for p in paths[i:i + cfg.batch_size]:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ok.append(p)
            except Exception as e:  # corrupt / unreadable file
                failed.append((str(p), str(e)))
        if imgs:
            chunks.append(encoder.encode_images(imgs))
            names.extend(p.relative_to(cfg.dataset_dir).as_posix() for p in ok)

    if failed:
        logger.warning("%d images failed: %s", len(failed), failed[:3])
    if not chunks:
        raise RuntimeError("No images could be embedded.")
    return np.vstack(chunks), names


def save_index(embeddings: np.ndarray, names: List[str], cfg: Config) -> None:
    if embeddings.shape[0] != len(names):
        raise ValueError("embeddings and names have different lengths")
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.embeddings_path, embeddings)
    cfg.filenames_path.write_text(json.dumps(names))
    logger.info("Saved %d embeddings to %s", len(names), cfg.index_dir)


def load_index(cfg: Config) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings + filenames. Accepts the Week 4 pickle as a fallback."""
    if not cfg.embeddings_path.exists():
        raise FileNotFoundError(f"No embeddings at {cfg.embeddings_path}. Run the 'build' command first.")
    embeddings = l2_normalize(np.load(cfg.embeddings_path))

    if cfg.filenames_path.exists():
        names = json.loads(cfg.filenames_path.read_text())
    elif cfg.legacy_filenames_path.exists():
        with open(cfg.legacy_filenames_path, "rb") as f:
            names = pickle.load(f)
    else:
        raise FileNotFoundError(f"No filenames file in {cfg.index_dir}")

    if embeddings.shape[0] != len(names):
        raise ValueError(f"Misaligned index: {embeddings.shape[0]} embeddings vs {len(names)} names")
    return embeddings, names