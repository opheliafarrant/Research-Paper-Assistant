"""Command-line entry point for the research assistant."""

import argparse
import os
import sys

from research_assistant import config


def main() -> None:
    os.environ.setdefault("HF_HOME", str(config.BASE_DIR / ".hf-cache"))
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="论文阅读助手命令行")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="查看已收录论文")
    commands.add_parser("fetch", help="查询 arXiv 上的新论文，不写入索引")
    commands.add_parser("ingest", help="获取并导入新论文")
    ask = commands.add_parser("ask", help="针对已收录论文提问")
    ask.add_argument("question", help="中文或英文问题")
    ask.add_argument("--paper-id", help="限定单篇论文")
    summary = commands.add_parser("summary", help="生成单篇论文摘要")
    summary.add_argument("paper_id", help="已收录论文的 arXiv 编号")
    args = parser.parse_args()
    from research_assistant.bootstrap import build_application
    from research_assistant.infrastructure.arxiv import fetch_new_papers

    app = build_application()

    if args.command == "list":
        for paper in app.list_papers():
            print(f"{paper['paper_id']}  {paper['title']}")
    elif args.command == "fetch":
        for paper in fetch_new_papers():
            print(f"{paper['paper_id']}  {paper['title']}")
    elif args.command == "ingest":
        app.ingest()
    elif args.command == "ask":
        result = app.answer(args.question, paper_id=args.paper_id)
        print(result["answer"])
        for source in result["sources"]:
            print(f"[{source['paper_id']}, p.{source['page']}] {source['title']}")
    elif args.command == "summary":
        paper = next((p for p in app.list_papers() if p["paper_id"] == args.paper_id), None)
        if paper is None:
            parser.error(f"论文未收录: {args.paper_id}")
        print(app.summarise(paper["paper_id"], paper["title"]))


if __name__ == "__main__":
    main()
