"""Wiring layer: the single entry point the Week 7 web interface should call."""
from config import Config
from embedding import CLIPEncoder
from indexer import load_index
from search import SearchEngine


def create_engine(cfg: Config = None) -> SearchEngine:
    """Load model + index once. Create this ONCE at server startup and reuse it."""
    cfg = cfg or Config()
    encoder = CLIPEncoder(cfg.model_name)
    embeddings, names = load_index(cfg)
    if embeddings.shape[1] != encoder.dim:
        raise ValueError(f"Index dim {embeddings.shape[1]} != model dim {encoder.dim}; "
                         "rebuild the index or fix cfg.model_name")
    return SearchEngine(encoder, embeddings, names, cfg.dataset_dir, cfg.max_top_k)