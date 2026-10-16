"""Command-line interface.

  python main.py build                       # embed the dataset and save the index
  python main.py search "a red car" -k 5     # one-off query
  python main.py search "a red car" --json   # machine-readable output
  python main.py search "a red car" --show   # also display images
  python main.py interactive                 # keep asking for queries
"""
import argparse
import json
import logging

from config import Config
from utils import display_results, setup_logging


def cmd_build(cfg: Config, args) -> None:
    from embedding import CLIPEncoder
    from indexer import build_index, save_index
    embeddings, names = build_index(CLIPEncoder(cfg.model_name), cfg)
    save_index(embeddings, names, cfg)
    print(f"Index built: {embeddings.shape[0]} images, dim {embeddings.shape[1]}")


def _print(results, query, as_json, show):
    if as_json:
        print(json.dumps(results, indent=2))
    else:
        print(f'\nTop {len(results)} for "{query}":')
        for r in results:
            print(f"  {r['rank']}. {r['score']:.4f}  {r['filename']}")
    if show:
        display_results(results, query)


def cmd_search(cfg: Config, args) -> None:
    from app import create_engine
    engine = create_engine(cfg)
    _print(engine.search(args.query, args.top_k), args.query, args.json, args.show)


def cmd_interactive(cfg: Config, args) -> None:
    from app import create_engine
    engine = create_engine(cfg)
    print("Type a query (empty line to quit).")
    while True:
        q = input("query> ").strip()
        if not q:
            break
        _print(engine.search(q, args.top_k), q, False, args.show)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI-powered image search engine")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="embed the dataset and save the index").set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="search with a text query")
    s.add_argument("query")
    s.add_argument("-k", "--top-k", type=int, default=Config().default_top_k)
    s.add_argument("--json", action="store_true")
    s.add_argument("--show", action="store_true")
    s.set_defaults(func=cmd_search)

    i = sub.add_parser("interactive", help="query repeatedly")
    i.add_argument("-k", "--top-k", type=int, default=Config().default_top_k)
    i.add_argument("--show", action="store_true")
    i.set_defaults(func=cmd_interactive)
    return p


def main() -> None:
    setup_logging(logging.WARNING)
    args = build_parser().parse_args()
    args.func(Config(), args)


if __name__ == "__main__":
    main()