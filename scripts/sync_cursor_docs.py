#!/usr/bin/env python3
"""同步官方 Cursor 文档源并刷新本地知识库。"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge import KnowledgeManager, KnowledgeStorage

OFFICIAL_LLMS_URL = "https://cursor.com/llms.txt"
DEFAULT_CACHE_PATH = ".cursor/cache/llms.txt"
DEFAULT_CLI_DOCS_PATH = "cursor_cli_docs.txt"
DEFAULT_AGENT_DOCS_PATH = "cursor_agent_docs.txt"
DEFAULT_FULL_DOCS_PATH = "cursor_docs_full.txt"

DOCS_PREFIX = "https://cursor.com/docs"
CN_DOCS_PREFIX = "https://cursor.com/cn/docs"

URL_RE = re.compile(r"https://cursor\.com/docs[^\s)]+")

CLI_PREFIXES = (f"{CN_DOCS_PREFIX}/cli/",)
CLI_EXTRA_URLS = {
    f"{CN_DOCS_PREFIX}/get-started/concepts.md",
    f"{CN_DOCS_PREFIX}/models.md",
    f"{CN_DOCS_PREFIX}/configuration/shell.md",
    f"{CN_DOCS_PREFIX}/settings/api-keys.md",
    f"{CN_DOCS_PREFIX}/integrations/git.md",
    f"{CN_DOCS_PREFIX}/integrations/github.md",
}

AGENT_PREFIXES = (
    f"{CN_DOCS_PREFIX}/agent/",
    f"{CN_DOCS_PREFIX}/context/",
    f"{CN_DOCS_PREFIX}/cloud-agent",
    f"{CN_DOCS_PREFIX}/plugins",
    f"{CN_DOCS_PREFIX}/tab/",
    f"{CN_DOCS_PREFIX}/inline-edit/",
)
AGENT_EXTRA_PREFIXES = (
    f"{CN_DOCS_PREFIX}/bugbot",
    f"{CN_DOCS_PREFIX}/shared-transcripts",
)
AGENT_EXTRA_URLS = {
    f"{CN_DOCS_PREFIX}/cookbook/agent-workflows.md",
    f"{CN_DOCS_PREFIX}/cookbook/building-mcp-server.md",
    f"{CN_DOCS_PREFIX}/cookbook/large-codebases.md",
    f"{CN_DOCS_PREFIX}/cookbook/bugbot-rules.md",
    f"{CN_DOCS_PREFIX}/get-started/concepts.md",
    f"{CN_DOCS_PREFIX}/models.md",
}


@dataclass
class SyncResult:
    source_count: int
    cli_count: int
    agent_count: int
    deleted_count: int
    saved_count: int
    failed_count: int


def stable_unique(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def extract_cursor_doc_urls(llms_text: str) -> list[str]:
    return stable_unique(URL_RE.findall(llms_text))


def to_cn_url(url: str) -> str:
    if url.startswith(DOCS_PREFIX):
        return url.replace(DOCS_PREFIX, CN_DOCS_PREFIX, 1)
    return url


def build_cli_doc_urls(cn_urls: list[str]) -> list[str]:
    return [url for url in cn_urls if any(url.startswith(prefix) for prefix in CLI_PREFIXES) or url in CLI_EXTRA_URLS]


def build_agent_doc_urls(cn_urls: list[str]) -> list[str]:
    return [
        url
        for url in cn_urls
        if any(url.startswith(prefix) for prefix in AGENT_PREFIXES)
        or any(url.startswith(prefix) for prefix in AGENT_EXTRA_PREFIXES)
        or url in AGENT_EXTRA_URLS
    ]


def render_url_list(title: str, urls: list[str], updated_on: str, section_title: str) -> str:
    lines = [
        title,
        f"# 基于官方 llms.txt 自动同步，更新日期: {updated_on}",
        f"# 共 {len(urls)} 个 URL",
        "",
        section_title,
    ]
    lines.extend(urls)
    return "\n".join(lines) + "\n"


def render_full_url_list(urls: list[str], updated_on: str) -> str:
    lines = [
        "# Cursor 完整文档列表 (中文版)",
        "# 自动生成自 https://cursor.com/llms.txt",
        f"# 更新日期: {updated_on}",
        "",
    ]
    lines.extend(urls)
    return "\n".join(lines) + "\n"


def write_text_file(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fetch_llms_text(llms_url: str, cache_path: Path, offline: bool, write_cache: bool) -> str:
    if offline:
        return cache_path.read_text(encoding="utf-8")

    with urlopen(llms_url) as response:
        content = response.read().decode("utf-8")
    if write_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
    return content


def build_sync_payload(llms_text: str, updated_on: str) -> tuple[list[str], list[str], list[str], dict[Path, str]]:
    source_urls = extract_cursor_doc_urls(llms_text)
    cn_urls = [to_cn_url(url) for url in source_urls]
    cli_urls = build_cli_doc_urls(cn_urls)
    agent_urls = build_agent_doc_urls(cn_urls)

    files = {
        PROJECT_ROOT / DEFAULT_CLI_DOCS_PATH: render_url_list(
            "# Cursor CLI 相关文档",
            cli_urls,
            updated_on,
            "# ========== CLI 核心文档 ==========",
        ),
        PROJECT_ROOT / DEFAULT_AGENT_DOCS_PATH: render_url_list(
            "# Cursor Agent 相关文档",
            agent_urls,
            updated_on,
            "# ========== Agent / Context / Cloud Agent ==========",
        ),
        PROJECT_ROOT / DEFAULT_FULL_DOCS_PATH: render_full_url_list(cn_urls, updated_on),
    }
    return cn_urls, cli_urls, agent_urls, files


def is_cursor_docs_url(url: str) -> bool:
    return url.startswith(CN_DOCS_PREFIX) or url.startswith(DOCS_PREFIX)


def find_obsolete_cursor_doc_ids(entries: Iterable[object], desired_urls: set[str]) -> list[str]:
    obsolete_ids: list[str] = []
    for entry in entries:
        url = getattr(entry, "url", "")
        doc_id = getattr(entry, "doc_id", "")
        if is_cursor_docs_url(url) and url not in desired_urls:
            obsolete_ids.append(doc_id)
    return obsolete_ids


async def refresh_knowledge_base(workspace_root: Path, desired_urls: list[str], dry_run: bool) -> tuple[int, int, int]:
    storage = KnowledgeStorage(workspace_root=str(workspace_root))
    await storage.initialize()

    desired_url_set = set(desired_urls)
    obsolete_doc_ids = find_obsolete_cursor_doc_ids(storage._index.values(), desired_url_set)

    if dry_run:
        return len(obsolete_doc_ids), 0, 0

    deleted_count = 0
    for doc_id in obsolete_doc_ids:
        if await storage.delete_document(doc_id):
            deleted_count += 1

    manager = KnowledgeManager()
    await manager.initialize()
    docs = await manager.add_urls(
        desired_urls,
        metadata={
            "source": "web",
            "category": "cursor-docs",
            "synced_at": datetime.now().isoformat(),
        },
        force_refresh=True,
    )

    saved_count = 0
    for doc in docs:
        saved, _ = await storage.save_document(doc, force=True)
        if saved:
            saved_count += 1

    failed_count = max(len(desired_urls) - len(docs), 0)
    return deleted_count, saved_count, failed_count


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="同步官方 Cursor 文档源并刷新本地知识库")
    parser.add_argument("--llms-url", default=OFFICIAL_LLMS_URL, help=f"llms.txt 来源 (默认: {OFFICIAL_LLMS_URL})")
    parser.add_argument(
        "--cache-path",
        default=DEFAULT_CACHE_PATH,
        help=f"llms.txt 缓存路径 (默认: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument("--offline", action="store_true", help="仅使用本地缓存，不发起网络请求")
    parser.add_argument("--skip-knowledge-sync", action="store_true", help="只更新源文件，不刷新 .cursor/knowledge")
    parser.add_argument("--dry-run", action="store_true", help="仅输出计划，不写文件、不修改知识库")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="工作区根目录 (默认: 当前仓库)",
    )
    return parser


async def run_sync(args: argparse.Namespace) -> SyncResult:
    workspace_root = Path(args.workspace_root).resolve()
    cache_path = (workspace_root / args.cache_path).resolve()
    llms_text = fetch_llms_text(
        args.llms_url,
        cache_path,
        offline=args.offline,
        write_cache=not args.dry_run,
    )

    updated_on = date.today().isoformat()
    source_urls, cli_urls, agent_urls, files = build_sync_payload(llms_text, updated_on)

    for path, content in files.items():
        write_text_file(path, content, args.dry_run)

    deleted_count = 0
    saved_count = 0
    failed_count = 0
    if not args.skip_knowledge_sync:
        desired_urls = stable_unique(cli_urls + agent_urls)
        deleted_count, saved_count, failed_count = await refresh_knowledge_base(
            workspace_root=workspace_root,
            desired_urls=desired_urls,
            dry_run=args.dry_run,
        )

    return SyncResult(
        source_count=len(source_urls),
        cli_count=len(cli_urls),
        agent_count=len(agent_urls),
        deleted_count=deleted_count,
        saved_count=saved_count,
        failed_count=failed_count,
    )


async def main() -> int:
    args = create_parser().parse_args()
    result = await run_sync(args)

    print(f"同步完成: full={result.source_count}, cli={result.cli_count}, agent={result.agent_count}")
    if args.skip_knowledge_sync:
        print("已跳过本地知识库刷新")
    elif args.dry_run:
        print(f"dry-run: 将删除 {result.deleted_count} 个旧条目")
    else:
        print(f"知识库刷新: 删除 {result.deleted_count} 个旧条目，保存 {result.saved_count} 个文档")
        if result.failed_count:
            print(f"警告: {result.failed_count} 个文档抓取失败")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
