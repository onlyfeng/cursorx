from __future__ import annotations

from types import SimpleNamespace

from scripts.sync_cursor_docs import (
    build_agent_doc_urls,
    build_cli_doc_urls,
    extract_cursor_doc_urls,
    find_obsolete_cursor_doc_ids,
    render_full_url_list,
    render_url_list,
    to_cn_url,
)


SAMPLE_LLMS_TXT = """
# Cursor Documentation
- https://cursor.com/docs/cli/overview.md
- https://cursor.com/docs/cli/acp.md
- https://cursor.com/docs/cli/reference/terminal-setup.md
- https://cursor.com/docs/agent/overview.md
- https://cursor.com/docs/agent/third-party-hooks.md
- https://cursor.com/docs/context/mcp/install-links.md
- https://cursor.com/docs/cloud-agent/api/endpoints.md
- https://cursor.com/docs/bugbot.md
- https://cursor.com/docs/plugins.md
- https://cursor.com/docs/get-started/concepts.md
- https://cursor.com/docs/models.md
- https://cursor.com/docs/cli/overview.md
"""


def test_extract_cursor_doc_urls_deduplicates_and_preserves_order() -> None:
    urls = extract_cursor_doc_urls(SAMPLE_LLMS_TXT)

    assert urls == [
        "https://cursor.com/docs/cli/overview.md",
        "https://cursor.com/docs/cli/acp.md",
        "https://cursor.com/docs/cli/reference/terminal-setup.md",
        "https://cursor.com/docs/agent/overview.md",
        "https://cursor.com/docs/agent/third-party-hooks.md",
        "https://cursor.com/docs/context/mcp/install-links.md",
        "https://cursor.com/docs/cloud-agent/api/endpoints.md",
        "https://cursor.com/docs/bugbot.md",
        "https://cursor.com/docs/plugins.md",
        "https://cursor.com/docs/get-started/concepts.md",
        "https://cursor.com/docs/models.md",
    ]


def test_build_doc_lists_include_latest_cli_and_agent_pages() -> None:
    cn_urls = [to_cn_url(url) for url in extract_cursor_doc_urls(SAMPLE_LLMS_TXT)]

    cli_urls = build_cli_doc_urls(cn_urls)
    agent_urls = build_agent_doc_urls(cn_urls)

    assert "https://cursor.com/cn/docs/cli/acp.md" in cli_urls
    assert "https://cursor.com/cn/docs/cli/reference/terminal-setup.md" in cli_urls
    assert "https://cursor.com/cn/docs/agent/third-party-hooks.md" in agent_urls
    assert "https://cursor.com/cn/docs/cloud-agent/api/endpoints.md" in agent_urls
    assert "https://cursor.com/cn/docs/bugbot.md" in agent_urls
    assert all("/cli/modes/" not in url for url in cli_urls)


def test_render_helpers_emit_updated_md_paths() -> None:
    cli_output = render_url_list(
        "# Cursor CLI 相关文档",
        [
            "https://cursor.com/cn/docs/cli/overview.md",
            "https://cursor.com/cn/docs/cli/acp.md",
        ],
        "2026-03-04",
        "# ========== CLI 核心文档 ==========",
    )
    full_output = render_full_url_list(
        ["https://cursor.com/cn/docs/cloud-agent/api/endpoints.md"],
        "2026-03-04",
    )

    assert "# 基于官方 llms.txt 自动同步，更新日期: 2026-03-04" in cli_output
    assert "https://cursor.com/cn/docs/cli/acp.md" in cli_output
    assert "# 更新日期: 2026-03-04" in full_output
    assert full_output.rstrip().endswith("https://cursor.com/cn/docs/cloud-agent/api/endpoints.md")


def test_find_obsolete_cursor_doc_ids_filters_removed_or_legacy_docs() -> None:
    entries = [
        SimpleNamespace(doc_id="keep-cli", url="https://cursor.com/cn/docs/cli/overview.md"),
        SimpleNamespace(doc_id="legacy", url="https://cursor.com/cn/docs/cli/overview"),
        SimpleNamespace(doc_id="removed", url="https://cursor.com/cn/docs/agent/review.md"),
        SimpleNamespace(doc_id="other", url="https://example.com/docs/page"),
    ]

    obsolete_ids = find_obsolete_cursor_doc_ids(
        entries,
        {"https://cursor.com/cn/docs/cli/overview.md"},
    )

    assert obsolete_ids == ["legacy", "removed"]
