#!/usr/bin/env python3
"""
check_all.sh JSON 输出渲染器

将 check_all.sh --json 的输出渲染为 GitHub Step Summary 格式的 Markdown。

用法:
    python scripts/render_check_all_summary.py /tmp/check_all.json
    bash scripts/check_all.sh --json | python scripts/render_check_all_summary.py -

输出:
    按 section 分组的 Markdown，包含:
    - 统计摘要（pass/fail/warn/skip）
    - 失败项详情（附 log_file, command, last_test）
    - 警告项列表
    - 跳过项列表
    - 通过项折叠展示
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# 确保项目根目录在 Python 路径中（支持直接运行脚本）
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 从统一契约模块导入 status_emoji
from core.check_all_contract import status_emoji  # noqa: E402


def load_json(source: str) -> dict[str, Any]:
    """从文件或 stdin 加载 JSON"""
    if source == "-":
        return json.load(sys.stdin)
    with open(source, encoding="utf-8") as f:
        return json.load(f)


def render_check_item(check: dict[str, Any], show_details: bool = True) -> list[str]:
    """渲染单个检查项"""
    lines = []
    emoji = status_emoji(check.get("status", ""))
    name = check.get("name", "未知检查")
    message = check.get("message", "")

    # 基本信息
    if message:
        lines.append(f"- {emoji} **{name}**: {message}")
    else:
        lines.append(f"- {emoji} **{name}**")

    # 详细信息（仅对失败/警告项显示）
    if show_details:
        details = []

        # 耗时
        duration_ms = check.get("duration_ms")
        if duration_ms is not None:
            if duration_ms < 1000:
                details.append(f"耗时: {duration_ms}ms")
            else:
                details.append(f"耗时: {duration_ms / 1000:.2f}s")

        # 最后一个测试用例
        last_test = check.get("last_test")
        if last_test:
            details.append(f"最后测试: `{last_test}`")

        # 日志文件
        log_file = check.get("log_file")
        if log_file:
            details.append(f"日志: `{log_file}`")

        # 复现命令
        command = check.get("command")
        if command:
            details.append(f"复现: `{command}`")

        if details:
            for detail in details:
                lines.append(f"  - {detail}")

    return lines


def group_checks_by_section(checks: list[dict]) -> dict[str, list[dict]]:
    """按 section 分组检查项"""
    grouped = defaultdict(list)
    for check in checks:
        section = check.get("section", "其他")
        grouped[section].append(check)
    return dict(grouped)


def group_checks_by_status(checks: list[dict]) -> dict[str, list[dict]]:
    """按 status 分组检查项"""
    grouped = defaultdict(list)
    for check in checks:
        status = check.get("status", "unknown")
        grouped[status].append(check)
    return dict(grouped)


def render_summary_table(data: dict[str, Any]) -> list[str]:
    """渲染统计摘要表格"""
    lines = []
    summary = data.get("summary", {})

    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    warnings = summary.get("warnings", 0)
    skipped = summary.get("skipped", 0)
    total = summary.get("total", passed + failed + warnings + skipped)

    success = data.get("success", failed == 0)
    data.get("exit_code", 0 if success else 1)

    # 标题
    if success:
        lines.append("## ✅ 项目健康检查通过")
    else:
        lines.append("## ❌ 项目健康检查失败")

    lines.append("")

    # 统计表格
    lines.append("| 类型 | 数量 |")
    lines.append("|------|------|")
    lines.append(f"| ✅ 通过 | {passed} |")
    lines.append(f"| ❌ 失败 | {failed} |")
    lines.append(f"| ⚠️ 警告 | {warnings} |")
    lines.append(f"| ⏭️ 跳过 | {skipped} |")
    lines.append(f"| **总计** | **{total}** |")
    lines.append("")

    # 元信息
    timestamp = data.get("timestamp", "")
    if timestamp:
        lines.append(f"> 检查时间: {timestamp}")

    log_dir = data.get("log_dir", "")
    if log_dir:
        lines.append(f"> 日志目录: `{log_dir}`")

    lines.append("")

    return lines


def render_section_failures(section_name: str, checks: list[dict], status_filter: str) -> list[str]:
    """渲染某个 section 中特定状态的检查项"""
    filtered = [c for c in checks if c.get("status") == status_filter]
    if not filtered:
        return []

    lines = []
    show_details = status_filter in ("fail", "warn")

    for check in filtered:
        lines.extend(render_check_item(check, show_details=show_details))

    return lines


def render_by_section(data: dict[str, Any]) -> list[str]:
    """按 section 分组渲染"""
    lines = []
    checks = data.get("checks", [])

    if not checks:
        lines.append("_没有检查结果_")
        return lines

    # 按 section 分组
    by_section = group_checks_by_section(checks)

    # 先渲染失败项（按 section）
    fail_sections = []
    for section_name, section_checks in by_section.items():
        fail_items = render_section_failures(section_name, section_checks, "fail")
        if fail_items:
            fail_sections.append((section_name, fail_items))

    if fail_sections:
        lines.append("### ❌ 失败项")
        lines.append("")
        for section_name, items in fail_sections:
            lines.append(f"#### {section_name}")
            lines.append("")
            lines.extend(items)
            lines.append("")

    # 渲染警告项（按 section）
    warn_sections = []
    for section_name, section_checks in by_section.items():
        warn_items = render_section_failures(section_name, section_checks, "warn")
        if warn_items:
            warn_sections.append((section_name, warn_items))

    if warn_sections:
        lines.append("### ⚠️ 警告项")
        lines.append("")
        for section_name, items in warn_sections:
            lines.append(f"#### {section_name}")
            lines.append("")
            lines.extend(items)
            lines.append("")

    # 渲染跳过项（简化列表）
    skip_items = []
    for section_name, section_checks in by_section.items():
        for check in section_checks:
            if check.get("status") == "skip":
                skip_items.append(f"- ⏭️ [{section_name}] {check.get('name', '未知')}")

    if skip_items:
        lines.append("<details>")
        lines.append(f"<summary>⏭️ 跳过项 ({len(skip_items)})</summary>")
        lines.append("")
        lines.extend(skip_items)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # 渲染通过项（折叠）
    pass_items = []
    for section_name, section_checks in by_section.items():
        for check in section_checks:
            if check.get("status") == "pass":
                pass_items.append(f"- ✅ [{section_name}] {check.get('name', '未知')}")

    if pass_items:
        lines.append("<details>")
        lines.append(f"<summary>✅ 通过项 ({len(pass_items)})</summary>")
        lines.append("")
        lines.extend(pass_items)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return lines


def render_durations(data: dict[str, Any]) -> list[str]:
    """渲染耗时统计"""
    lines: list[str] = []
    durations = data.get("durations", [])

    if not durations:
        return lines

    # 按耗时排序（降序）
    sorted_durations = sorted(durations, key=lambda d: d.get("duration_ms", 0), reverse=True)

    # 只显示前 10 个最慢的
    top_n = sorted_durations[:10]
    if not top_n:
        return lines

    lines.append("<details>")
    lines.append("<summary>⏱️ 耗时统计 (Top 10)</summary>")
    lines.append("")
    lines.append("| 检查项 | 耗时 |")
    lines.append("|--------|------|")

    for d in top_n:
        name = d.get("name", "未知")
        ms = d.get("duration_ms", 0)
        time_str = f"{ms}ms" if ms < 1000 else f"{ms / 1000:.2f}s"
        lines.append(f"| {name} | {time_str} |")

    lines.append("")
    lines.append("</details>")
    lines.append("")

    return lines


def render_tips(data: dict[str, Any]) -> list[str]:
    """渲染修复提示"""
    lines = []
    success = data.get("success", True)

    lines.append("### 💡 提示")
    lines.append("")

    if not success:
        lines.append("- 本地运行 `bash scripts/check_all.sh --full` 复现问题")
        lines.append("- 查看上方失败项的**日志**和**复现命令**")

    lines.append("- 运行 `bash scripts/check_all.sh --full --json` 获取 JSON 输出")
    lines.append("- 运行 `python scripts/render_check_all_summary.py /tmp/check_all.json` 生成 Markdown")
    lines.append("")

    return lines


def render_markdown(data: dict[str, Any]) -> str:
    """生成完整的 Markdown 输出"""
    lines = []

    # 统计摘要
    lines.extend(render_summary_table(data))

    # 按 section 分组的检查结果
    lines.extend(render_by_section(data))

    # 耗时统计
    lines.extend(render_durations(data))

    # 修复提示
    lines.extend(render_tips(data))

    return "\n".join(lines)


def main() -> int:
    """主入口"""
    if len(sys.argv) < 2:
        print("用法: python scripts/render_check_all_summary.py <json_file>")
        print("      python scripts/render_check_all_summary.py -  # 从 stdin 读取")
        return 1

    source = sys.argv[1]

    try:
        data = load_json(source)
    except FileNotFoundError:
        print(f"错误: 文件不存在: {source}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败: {e}", file=sys.stderr)
        return 1

    markdown = render_markdown(data)
    print(markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
