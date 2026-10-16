"""Streamlit front-end for the AI-powered image search engine (Week 7).

Run from the project root:   streamlit run streamlit_app.py
Uses the Week 6 backend: app.create_engine() -> SearchEngine.search(query, top_k)
"""
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make Week 6 modules importable

st.set_page_config(page_title="AI Image Search", page_icon="🔎", layout="wide")

EXAMPLE_QUERIES = ["a dog playing in the park", "a red car", "people eating food", "a sunset over the ocean"]
COLUMNS_PER_ROW = 5


@st.cache_resource(show_spinner="Loading CLIP model and image index (first run only)...")
def load_engine():
    """Heavy objects are created ONCE per server process, not on every interaction."""
    from app import create_engine
    return create_engine()


def set_query(text: str) -> None:
    st.session_state["query"] = text


def render_results(results, show_scores: bool) -> None:
    for start in range(0, len(results), COLUMNS_PER_ROW):
        cols = st.columns(COLUMNS_PER_ROW)
        for col, r in zip(cols, results[start:start + COLUMNS_PER_ROW]):
            with col:
                try:
                    img = Image.open(r["path"]).convert("RGB")
                    img.thumbnail((512, 512))          # keep the page light
                    caption = f"#{r['rank']}  {Path(r['filename']).name}"
                    if show_scores:
                        caption += f"\nsimilarity: {r['score']:.3f}"
                    st.image(img, caption=caption)
                except Exception:
                    st.warning(f"Could not load {r['filename']}")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results (Top-K)", min_value=1, max_value=20, value=5)
    show_scores = st.checkbox("Show similarity scores", value=True)
    st.divider()
    st.caption("Model: CLIP ViT-B/32 · Ranking: cosine similarity")

# ---------------- Main page ----------------
st.title("🔎 AI-Powered Image Search")
st.write("Describe what you are looking for in plain English and the engine finds matching images.")

try:
    engine = load_engine()
except FileNotFoundError as e:
    st.error(f"Index or dataset not found: {e}\n\nRun `python main.py build` or check your paths in config.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not start the search engine: {e}")
    st.stop()

with st.sidebar:
    st.caption(f"Indexed images: **{len(engine.filenames):,}**")

query = st.text_input("Search query", key="query", placeholder="e.g. a dog playing in the park")

st.caption("Try an example:")
example_cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(example_cols, EXAMPLE_QUERIES):
    col.button(example, on_click=set_query, args=(example,))

if not query.strip():
    st.info("Type a query above, or click an example, to start searching.")
else:
    try:
        t0 = time.perf_counter()
        with st.spinner("Searching..."):
            results = engine.search(query, top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Search failed: {e}")
    else:
        st.subheader(f'Top {len(results)} results for "{query.strip()}"')
        st.caption(f"Searched {len(engine.filenames):,} images in {elapsed_ms:.0f} ms")
        render_results(results, show_scores)