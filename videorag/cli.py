"""Command line: `winter` (or `python -m videorag`) ingest|ask|search|list|serve."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from videorag.chunking import format_ts
from videorag.config import settings


def _services():
    from videorag.pipeline import Ingestor
    from videorag.retrieval import Retriever
    from videorag.store import Store

    store = Store(settings.db_path)
    return store, Ingestor(store, settings), Retriever(store, settings)


def cmd_ingest(args) -> int:
    store, ingestor, _ = _services()
    failed = 0
    for item in args.inputs:
        if item.startswith(("http://", "https://")):
            vid = ingestor.register_url(item)
        else:
            p = Path(item)
            if not p.exists():
                print(f"skip: {item} does not exist", file=sys.stderr)
                failed += 1
                continue
            vid = ingestor.register_file(p, title=args.title)
        print(f"[{vid}] ingesting {item} ...", flush=True)
        ingestor.run(vid)
        v = store.get_video(vid)
        if v.status == "ready":
            print(f"[{vid}] ready: {v.title} ({format_ts(v.duration or 0)}, language={v.language})")
        else:
            failed += 1
            print(f"[{vid}] FAILED: {v.error}", file=sys.stderr)
    return 1 if failed else 0


def cmd_list(_args) -> int:
    store, _, _ = _services()
    for v in store.list_videos():
        print(f"{v.id}  {v.status:<10} {format_ts(v.duration or 0):>8}  {v.title}")
    return 0


def cmd_search(args) -> int:
    _, _, retriever = _services()
    for s in retriever.search(args.query, k=args.k, video_ids=args.video).sources:
        print(f"[{s.n}] {s.video_title} @ {format_ts(s.start)}  ({s.kind}, score {s.score})\n    {s.text[:200]}\n")
    return 0


def cmd_ask(args) -> int:
    from videorag.llm import stream_answer

    _, _, retriever = _services()
    result = retriever.search(args.question, k=args.k, video_ids=args.video)
    for text in stream_answer(args.question, result.sources, settings):
        print(text, end="", flush=True)
    print("\n\nSources:")
    for s in result.sources:
        print(f"  [{s.n}] {s.video_title} @ {format_ts(s.start)}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("videorag.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="winter", description="Winter: ask questions about your videos.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="index video files or URLs")
    s.add_argument("inputs", nargs="+", help="file paths or YouTube/HTTP URLs")
    s.add_argument("--title", help="title for a single local file")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("list", help="list indexed videos")
    s.set_defaults(fn=cmd_list)

    for name, fn, arg in (("search", cmd_search, "query"), ("ask", cmd_ask, "question")):
        s = sub.add_parser(name, help=f"{name} across indexed videos")
        s.add_argument(arg)
        s.add_argument("-k", type=int, default=6)
        s.add_argument("--video", action="append", help="restrict to a video id (repeatable)")
        s.set_defaults(fn=fn)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    sys.exit(args.fn(args))
