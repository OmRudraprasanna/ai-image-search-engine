# AI-Powered Image Search Engine

A semantic image search engine. Type a description in plain English, and it returns the most relevant images from a dataset using OpenAI's CLIP model and cosine similarity.

## Features
- Natural-language image search (no tags or captions needed)
- Adjustable Top-K results and optional similarity scores
- Modular backend, reusable from scripts, a CLI, or a web UI
- Streamlit web interface
- Unit tests that run without downloading the model

## How it works
1. **Indexing (once):** every dataset image is encoded by CLIP's image encoder into a 512-dimensional vector, then L2-normalized and saved.
2. **Query:** the text query is encoded by CLIP's text encoder into the same vector space.
3. **Ranking:** cosine similarity between the query vector and all image vectors (a dot product, since vectors are normalized). The K highest scores are returned.

Model: `openai/clip-vit-base-patch32`. The index and the query encoder must use the same model.

## Project structure
```
├── streamlit_app.py   # Streamlit web interface (Week 7)
├── app.py             # create_engine(): loads model + index
├── search.py          # SearchEngine: cosine-similarity Top-K
├── embedding.py       # CLIPEncoder: text/image -> embeddings
├── indexer.py         # build / save / load the embedding index
├── config.py          # paths and settings
├── utils.py           # helpers
├── main.py            # command-line interface
├── tests/             # pytest tests
└── requirements.txt
```
`dataset/` and `data/` are not included in this repository.

## Installation
```bash
git clone https://github.com/OmRudraprasanna/ai-image-search-engine.git
cd ai-image-search-engine 
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Prepare the data
1. Put your images in a folder named `dataset/` in the project root.
2. Build the embedding index (one time; takes a few minutes on CPU):
```bash
python main.py build
```
This creates `data/image_embeddings.npy` and `data/image_filenames.json`.

## Usage

**Web interface**
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`.

**Command line**
```bash
python main.py search "a dog playing in the park" -k 5
python main.py search "a red car" --json
python main.py interactive
```

**Python**
```python
from app import create_engine
engine = create_engine()          # load once
results = engine.search("a red car", top_k=5)
# [{"rank": 1, "filename": "...", "path": "...", "score": 0.31}, ...]
```

## Tests
```bash
python -m pytest -q
```

## Results


![Good query](docs/screenshots/query1.png)
![Failing query](docs/screenshots/query2.png)

| Query | Result quality | Notes |
|-------|----------------|-------|
| [query 1] | [e.g. 5/5 relevant] | [...] |
| [query 2] | [...] | [...] |
| [failing query] | [...] | [...] |

## Limitations
- CLIP is weak at counting ("three dogs"), negation ("no cars"), and spatial relations ("cat left of dog").
- It can only retrieve what exists in the dataset.
- Search is brute-force over all embeddings. This is fine for thousands of images; larger collections would need an index such as FAISS.
- Scores for text-to-image search are typically low (about 0.20 to 0.35); the ranking matters, not the absolute value.

## Future work
- Image-to-image search and combined text+image queries
- FAISS for large datasets
- Fine-tuning or a larger CLIP model

## Tech stack
Python, PyTorch, Hugging Face Transformers, NumPy, Streamlit

## Author
Om Rudraprasanna Mohanty · [Intern IN2517 , IIT BBSR, 2026]
