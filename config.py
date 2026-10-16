"""Central configuration. Paths are relative to the project folder; override with env vars."""
import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # Must be the SAME model used to create the stored image embeddings.
    model_name: str = "openai/clip-vit-base-patch32"

    dataset_dir: Path = Path(os.getenv("DATASET_DIR", BASE_DIR / "dataset"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", BASE_DIR / "data"))

    batch_size: int = 32
    default_top_k: int = 5
    max_top_k: int = 50
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "image_embeddings.npy"

    @property
    def filenames_path(self) -> Path:
        return self.index_dir / "image_filenames.json"

    @property
    def legacy_filenames_path(self) -> Path:
        """Week 4 saved filenames as a pickle; still supported for loading."""
        return self.index_dir / "image_filenames.pkl"