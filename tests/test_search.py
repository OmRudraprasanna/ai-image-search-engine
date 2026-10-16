import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from indexer import load_index, save_index
from search import SearchEngine
from utils import l2_normalize


class FakeEncoder:
    """Maps known query strings to fixed 3-D vectors; no model download needed."""
    VECS = {"cat": [1, 0, 0], "dog": [0, 1, 0], "bird": [0, 0, 1], "catdog": [1, 1, 0]}

    def encode_text(self, text):
        return l2_normalize(np.array([self.VECS[text]], dtype=np.float32))


@pytest.fixture
def engine(tmp_path):
    emb = l2_normalize(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32))
    names = ["cat.jpg", "dog.jpg", "bird.jpg", "both.jpg"]
    return SearchEngine(FakeEncoder(), emb, names, tmp_path)


def test_top1_is_correct(engine):
    assert engine.search("cat", 1)[0]["filename"] == "cat.jpg"


def test_results_sorted_and_ranked(engine):
    res = engine.search("catdog", 4)
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in res] == [1, 2, 3, 4]
    assert res[0]["filename"] == "both.jpg"


def test_top_k_clamped_to_dataset_size(engine):
    assert len(engine.search("cat", 999)) == 4


def test_top_k_minimum_is_one(engine):
    assert len(engine.search("cat", 0)) == 1


def test_empty_query_rejected(engine):
    with pytest.raises(ValueError):
        engine.search("   ")


def test_result_is_json_serializable(engine):
    json.dumps(engine.search("dog", 2))


def test_dimension_mismatch_detected(tmp_path):
    emb = l2_normalize(np.ones((2, 5), dtype=np.float32))
    eng = SearchEngine(FakeEncoder(), emb, ["a", "b"], tmp_path)
    with pytest.raises(ValueError):
        eng.search("cat")


def test_misaligned_inputs_rejected(tmp_path):
    with pytest.raises(ValueError):
        SearchEngine(FakeEncoder(), np.ones((3, 3), dtype=np.float32), ["a"], tmp_path)


def test_index_roundtrip(tmp_path):
    cfg = Config(index_dir=tmp_path)
    emb = l2_normalize(np.random.rand(5, 4).astype(np.float32))
    names = [f"{i}.jpg" for i in range(5)]
    save_index(emb, names, cfg)
    emb2, names2 = load_index(cfg)
    assert names2 == names and np.allclose(emb, emb2, atol=1e-6)


def test_legacy_pickle_names_load(tmp_path):
    import pickle
    cfg = Config(index_dir=tmp_path)
    np.save(cfg.embeddings_path, np.ones((2, 3), dtype=np.float32))
    pickle.dump(["a.jpg", "b.jpg"], open(cfg.legacy_filenames_path, "wb"))
    _, names = load_index(cfg)
    assert names == ["a.jpg", "b.jpg"]


def test_legacy_basename_resolves_in_subfolder(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "0001.jpg").write_bytes(b"x")
    emb = l2_normalize(np.array([[1, 0, 0]], dtype=np.float32))
    eng = SearchEngine(FakeEncoder(), emb, ["0001.jpg"], tmp_path)
    assert Path(eng.search("cat", 1)[0]["path"]) == tmp_path / "sub" / "0001.jpg"


def test_build_index_skips_corrupt_and_keeps_alignment(tmp_path):
    from PIL import Image
    from indexer import build_index

    ds = tmp_path / "ds"
    (ds / "sub").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(ds / "a.jpg")
    Image.new("RGB", (8, 8), "blue").save(ds / "sub" / "a.jpg")   # same basename, different folder
    (ds / "bad.jpg").write_bytes(b"not an image")

    class ImgEnc:
        def encode_images(self, imgs):
            return l2_normalize(np.ones((len(imgs), 3), dtype=np.float32))

    emb, names = build_index(ImgEnc(), Config(dataset_dir=ds, index_dir=tmp_path))
    assert names == ["a.jpg", "sub/a.jpg"]      # corrupt file skipped, no collision
    assert emb.shape[0] == len(names)