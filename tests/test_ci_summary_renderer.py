"""render_check_all_summary.py 单元测试

测试 check_all.sh JSON 输出渲染器的各项功能：
- 按 section 分组
- fail/warn/skip/pass 统计
- 复现命令字段
- 日志文件字段
- 耗时统计
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 从统一契约模块导入常量
from core.check_all_contract import (  # noqa: E402
    STATUS_EMOJI_MAP,
    VALID_STATUSES,
    status_emoji,
)
from scripts.render_check_all_summary import (  # noqa: E402
    group_checks_by_section,
    group_checks_by_status,
    render_check_item,
    render_markdown,
    render_summary_table,
)

# ==================== 测试 Fixtures ====================


@pytest.fixture
def sample_json_data() -> dict:
    """固定的测试 JSON 样例数据"""
    return {
        "success": False,
        "exit_code": 1,
        "summary": {
            "passed": 5,
            "failed": 2,
            "warnings": 1,
            "skipped": 2,
            "total": 10,
        },
        "ci_mode": True,
        "fail_fast": False,
        "full_check": True,
        "diagnose_hang": False,
        "timeout_backend": "timeout",
        "log_dir": "/tmp/check_all_logs_20250129",
        "project_root": "/Users/test/project",
        "timestamp": "2025-01-29T10:30:00+08:00",
        "durations": [
            {"name": "pytest-unit", "duration_ms": 5000},
            {"name": "mypy", "duration_ms": 3000},
            {"name": "ruff", "duration_ms": 500},
        ],
        "checks": [
            {
                "section": "代码风格",
                "name": "ruff-check",
                "status": "pass",
                "message": "代码风格检查通过",
                "duration_ms": 250,
            },
            {
                "section": "代码风格",
                "name": "ruff-format",
                "status": "pass",
                "message": "格式检查通过",
                "duration_ms": 200,
            },
            {
                "section": "类型检查",
                "name": "mypy",
                "status": "fail",
                "message": "发现 3 个类型错误",
                "duration_ms": 3000,
                "log_file": "/tmp/check_all_logs_20250129/mypy.log",
                "command": "mypy core/ --strict",
                "last_test": "core/config.py:45",
            },
            {
                "section": "测试",
                "name": "pytest-unit",
                "status": "fail",
                "message": "2 个测试失败",
                "duration_ms": 5000,
                "log_file": "/tmp/check_all_logs_20250129/pytest.log",
                "command": "pytest tests/ -v",
                "last_test": "tests/test_config.py::test_load_config",
            },
            {
                "section": "测试",
                "name": "pytest-integration",
                "status": "skip",
                "message": "跳过集成测试（无 API Key）",
            },
            {
                "section": "安全检查",
                "name": "pip-audit",
                "status": "warn",
                "message": "发现 1 个低危漏洞",
                "duration_ms": 1500,
                "log_file": "/tmp/check_all_logs_20250129/pip-audit.log",
                "command": "pip-audit --strict",
            },
            {
                "section": "安全检查",
                "name": "bandit",
                "status": "pass",
                "message": "未发现安全问题",
                "duration_ms": 800,
            },
            {
                "section": "依赖检查",
                "name": "dep-check",
                "status": "pass",
                "message": "依赖一致",
                "duration_ms": 300,
            },
            {
                "section": "依赖检查",
                "name": "outdated-check",
                "status": "skip",
                "message": "跳过过期检查",
            },
            {
                "section": "文档",
                "name": "doc-build",
                "status": "pass",
                "message": "文档构建成功",
                "duration_ms": 1200,
            },
        ],
    }


@pytest.fixture
def success_json_data() -> dict:
    """全部通过的测试 JSON 样例"""
    return {
        "success": True,
        "exit_code": 0,
        "summary": {
            "passed": 5,
            "failed": 0,
            "warnings": 0,
            "skipped": 0,
            "total": 5,
        },
        "timestamp": "2025-01-29T10:30:00+08:00",
        "checks": [
            {"section": "代码风格", "name": "ruff", "status": "pass"},
            {"section": "类型检查", "name": "mypy", "status": "pass"},
            {"section": "测试", "name": "pytest", "status": "pass"},
            {"section": "安全检查", "name": "bandit", "status": "pass"},
            {"section": "依赖检查", "name": "deps", "status": "pass"},
        ],
        "durations": [],
    }


@pytest.fixture
def empty_json_data() -> dict:
    """空检查结果的 JSON 样例"""
    return {
        "success": True,
        "exit_code": 0,
        "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
        "checks": [],
        "durations": [],
    }


# ==================== status_emoji 测试 ====================


class TestStatusEmoji:
    """status_emoji 函数测试"""

    def test_pass_emoji(self):
        """测试 pass 状态的 emoji"""
        assert status_emoji("pass") == STATUS_EMOJI_MAP["pass"]

    def test_fail_emoji(self):
        """测试 fail 状态的 emoji"""
        assert status_emoji("fail") == STATUS_EMOJI_MAP["fail"]

    def test_warn_emoji(self):
        """测试 warn 状态的 emoji"""
        assert status_emoji("warn") == STATUS_EMOJI_MAP["warn"]

    def test_skip_emoji(self):
        """测试 skip 状态的 emoji"""
        assert status_emoji("skip") == STATUS_EMOJI_MAP["skip"]

    def test_info_emoji(self):
        """测试 info 状态的 emoji"""
        assert status_emoji("info") == STATUS_EMOJI_MAP["info"]

    def test_unknown_status(self):
        """测试未知状态的 emoji"""
        assert status_emoji("unknown") == "❓"
        assert status_emoji("") == "❓"

    def test_all_valid_statuses_have_emoji(self):
        """测试所有有效状态都有对应的 emoji"""
        for status in VALID_STATUSES:
            assert status in STATUS_EMOJI_MAP, f"状态 {status} 缺少 emoji 定义"
            assert status_emoji(status) == STATUS_EMOJI_MAP[status]

    def test_various_unknown_statuses(self):
        """测试各种未知状态都回退到 ❓"""
        unknown_values = [
            "error",  # 常见但不在枚举中
            "pending",
            "running",
            "cancelled",
            "timeout",
            "PASS",  # 大写
            "Fail",  # 混合大小写
            "123",
            "?",
            "❌",  # emoji 本身作为状态值
        ]
        for status in unknown_values:
            assert status_emoji(status) == "❓", f"status={status!r} 应返回 ❓"


# ==================== group_checks_by_section 测试 ====================


class TestGroupChecksBySection:
    """group_checks_by_section 函数测试"""

    def test_basic_grouping(self, sample_json_data: dict):
        """测试基本按 section 分组"""
        checks = sample_json_data["checks"]
        grouped = group_checks_by_section(checks)

        assert "代码风格" in grouped
        assert "类型检查" in grouped
        assert "测试" in grouped
        assert "安全检查" in grouped
        assert "依赖检查" in grouped
        assert "文档" in grouped

    def test_section_count(self, sample_json_data: dict):
        """测试每个 section 的检查项数量"""
        checks = sample_json_data["checks"]
        grouped = group_checks_by_section(checks)

        assert len(grouped["代码风格"]) == 2
        assert len(grouped["类型检查"]) == 1
        assert len(grouped["测试"]) == 2
        assert len(grouped["安全检查"]) == 2
        assert len(grouped["依赖检查"]) == 2
        assert len(grouped["文档"]) == 1

    def test_empty_checks(self):
        """测试空检查列表"""
        grouped = group_checks_by_section([])
        assert grouped == {}

    def test_missing_section_field(self):
        """测试缺少 section 字段的检查项"""
        checks = [
            {"name": "test1", "status": "pass"},
            {"name": "test2", "status": "fail", "section": "测试"},
        ]
        grouped = group_checks_by_section(checks)

        assert "其他" in grouped
        assert "测试" in grouped
        assert len(grouped["其他"]) == 1
        assert len(grouped["测试"]) == 1


# ==================== group_checks_by_status 测试 ====================


class TestGroupChecksByStatus:
    """group_checks_by_status 函数测试"""

    def test_basic_grouping(self, sample_json_data: dict):
        """测试基本按 status 分组"""
        checks = sample_json_data["checks"]
        grouped = group_checks_by_status(checks)

        assert "pass" in grouped
        assert "fail" in grouped
        assert "warn" in grouped
        assert "skip" in grouped

    def test_status_count(self, sample_json_data: dict):
        """测试每个 status 的检查项数量"""
        checks = sample_json_data["checks"]
        grouped = group_checks_by_status(checks)

        assert len(grouped["pass"]) == 5
        assert len(grouped["fail"]) == 2
        assert len(grouped["warn"]) == 1
        assert len(grouped["skip"]) == 2


# ==================== render_check_item 测试 ====================


class TestRenderCheckItem:
    """render_check_item 函数测试"""

    def test_basic_render(self):
        """测试基本渲染"""
        check = {"name": "测试项", "status": "pass", "message": "通过"}
        lines = render_check_item(check)

        assert len(lines) >= 1
        assert STATUS_EMOJI_MAP["pass"] in lines[0]
        assert "**测试项**" in lines[0]
        assert "通过" in lines[0]

    def test_render_with_log_file(self):
        """测试包含日志文件的渲染"""
        check = {
            "name": "失败项",
            "status": "fail",
            "message": "失败",
            "log_file": "/tmp/test.log",
        }
        lines = render_check_item(check, show_details=True)

        output = "\n".join(lines)
        assert "日志:" in output
        assert "`/tmp/test.log`" in output

    def test_render_with_command(self):
        """测试包含复现命令的渲染"""
        check = {
            "name": "失败项",
            "status": "fail",
            "message": "失败",
            "command": "pytest tests/ -v",
        }
        lines = render_check_item(check, show_details=True)

        output = "\n".join(lines)
        assert "复现:" in output
        assert "`pytest tests/ -v`" in output

    def test_render_with_last_test(self):
        """测试包含最后测试用例的渲染"""
        check = {
            "name": "失败项",
            "status": "fail",
            "message": "失败",
            "last_test": "tests/test_foo.py::test_bar",
        }
        lines = render_check_item(check, show_details=True)

        output = "\n".join(lines)
        assert "最后测试:" in output
        assert "`tests/test_foo.py::test_bar`" in output

    def test_render_with_duration_ms(self):
        """测试包含耗时的渲染（毫秒）"""
        check = {"name": "快速检查", "status": "pass", "duration_ms": 500}
        lines = render_check_item(check, show_details=True)

        output = "\n".join(lines)
        assert "耗时:" in output
        assert "500ms" in output

    def test_render_with_duration_seconds(self):
        """测试包含耗时的渲染（秒）"""
        check = {"name": "慢检查", "status": "pass", "duration_ms": 5000}
        lines = render_check_item(check, show_details=True)

        output = "\n".join(lines)
        assert "耗时:" in output
        assert "5.00s" in output

    def test_render_no_details(self):
        """测试不显示详情"""
        check = {
            "name": "测试项",
            "status": "pass",
            "log_file": "/tmp/test.log",
            "command": "pytest",
        }
        lines = render_check_item(check, show_details=False)

        output = "\n".join(lines)
        assert "日志:" not in output
        assert "复现:" not in output


# ==================== render_summary_table 测试 ====================


class TestRenderSummaryTable:
    """render_summary_table 函数测试"""

    def test_failed_summary(self, sample_json_data: dict):
        """测试失败情况的统计表格"""
        lines = render_summary_table(sample_json_data)
        output = "\n".join(lines)

        # 标题应显示失败
        assert "## ❌ 项目健康检查失败" in output

        # 统计表格
        assert "| ✅ 通过 | 5 |" in output
        assert "| ❌ 失败 | 2 |" in output
        assert "| ⚠️ 警告 | 1 |" in output
        assert "| ⏭️ 跳过 | 2 |" in output
        assert "| **总计** | **10** |" in output

    def test_success_summary(self, success_json_data: dict):
        """测试成功情况的统计表格"""
        lines = render_summary_table(success_json_data)
        output = "\n".join(lines)

        assert "## ✅ 项目健康检查通过" in output

    def test_timestamp_display(self, sample_json_data: dict):
        """测试时间戳显示"""
        lines = render_summary_table(sample_json_data)
        output = "\n".join(lines)

        assert "检查时间:" in output

    def test_log_dir_display(self, sample_json_data: dict):
        """测试日志目录显示"""
        lines = render_summary_table(sample_json_data)
        output = "\n".join(lines)

        assert "日志目录:" in output
        assert "`/tmp/check_all_logs_20250129`" in output


# ==================== render_markdown 完整渲染测试 ====================


class TestRenderMarkdown:
    """render_markdown 完整渲染测试"""

    def test_full_render_structure(self, sample_json_data: dict):
        """测试完整渲染的结构"""
        output = render_markdown(sample_json_data)

        # 应包含统计摘要
        assert "## ❌ 项目健康检查失败" in output

        # 应包含失败项章节
        assert "### ❌ 失败项" in output

        # 应包含警告项章节
        assert "### ⚠️ 警告项" in output

        # 应包含跳过项（折叠）
        assert "⏭️ 跳过项" in output

        # 应包含通过项（折叠）
        assert "✅ 通过项" in output

        # 应包含提示
        assert "### 💡 提示" in output

    def test_section_grouping_in_failures(self, sample_json_data: dict):
        """测试失败项按 section 分组"""
        output = render_markdown(sample_json_data)

        # 失败项应按 section 分组
        assert "#### 类型检查" in output
        assert "#### 测试" in output

    def test_failure_details(self, sample_json_data: dict):
        """测试失败项包含详细信息"""
        output = render_markdown(sample_json_data)

        # mypy 失败项应包含详情
        assert "mypy" in output
        assert "发现 3 个类型错误" in output
        assert "复现:" in output
        assert "`mypy core/ --strict`" in output

        # pytest 失败项应包含详情
        assert "pytest-unit" in output
        assert "`pytest tests/ -v`" in output
        assert "最后测试:" in output

    def test_skip_items_count(self, sample_json_data: dict):
        """测试跳过项数量显示"""
        output = render_markdown(sample_json_data)

        # 跳过项应显示数量
        assert "⏭️ 跳过项 (2)" in output

    def test_pass_items_count(self, sample_json_data: dict):
        """测试通过项数量显示"""
        output = render_markdown(sample_json_data)

        # 通过项应显示数量
        assert "✅ 通过项 (5)" in output

    def test_empty_checks(self, empty_json_data: dict):
        """测试空检查结果"""
        output = render_markdown(empty_json_data)

        assert "_没有检查结果_" in output

    def test_success_render(self, success_json_data: dict):
        """测试全部成功的渲染"""
        output = render_markdown(success_json_data)

        assert "## ✅ 项目健康检查通过" in output
        # 不应有失败项章节
        assert "### ❌ 失败项" not in output
        # 不应有警告项章节
        assert "### ⚠️ 警告项" not in output

    def test_fix_tips_on_failure(self, sample_json_data: dict):
        """测试失败时的修复提示"""
        output = render_markdown(sample_json_data)

        assert "本地运行 `bash scripts/check_all.sh --full` 复现问题" in output


# ==================== 边缘情况测试 ====================


class TestEdgeCases:
    """边缘情况测试（字段缺失、未知值等）"""

    def test_durations_missing(self):
        """测试 durations 字段缺失时渲染不抛异常"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 2, "failed": 0, "warnings": 0, "skipped": 0, "total": 2},
            "checks": [
                {"section": "测试", "name": "test1", "status": "pass"},
                {"section": "测试", "name": "test2", "status": "pass"},
            ],
            # durations 字段完全缺失
        }
        output = render_markdown(data)

        # 不应抛异常，应包含基本结构
        assert "## ✅ 项目健康检查通过" in output
        assert "| 类型 | 数量 |" in output
        # 不应包含耗时统计折叠块
        assert "⏱️ 耗时统计" not in output

    def test_summary_total_missing_inferred_from_counts(self):
        """测试 summary.total 缺失时由 passed/failed/... 推导"""
        data = {
            "success": False,
            "exit_code": 1,
            "summary": {
                "passed": 3,
                "failed": 1,
                "warnings": 2,
                "skipped": 1,
                # total 字段缺失，应由 3+1+2+1=7 推导
            },
            "checks": [
                {"section": "测试", "name": "test1", "status": "pass"},
                {"section": "测试", "name": "test2", "status": "pass"},
                {"section": "测试", "name": "test3", "status": "pass"},
                {"section": "测试", "name": "test4", "status": "fail"},
                {"section": "安全", "name": "sec1", "status": "warn"},
                {"section": "安全", "name": "sec2", "status": "warn"},
                {"section": "其他", "name": "other1", "status": "skip"},
            ],
            "durations": [],
        }
        output = render_markdown(data)

        # 不应抛异常
        assert "## ❌ 项目健康检查失败" in output
        # total 应被推导为 7
        assert "| **总计** | **7** |" in output

    def test_checks_with_extra_unknown_fields(self):
        """测试 checks 中存在额外未知字段时渲染正常"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 1, "failed": 0, "warnings": 0, "skipped": 0, "total": 1},
            "checks": [
                {
                    "section": "测试",
                    "name": "test1",
                    "status": "pass",
                    "message": "通过",
                    # 额外未知字段
                    "unknown_field_1": "some_value",
                    "unknown_field_2": 12345,
                    "extra_metadata": {"nested": "data"},
                    "legacy_code": None,
                },
            ],
            "durations": [],
        }
        output = render_markdown(data)

        # 不应抛异常，应正常渲染
        assert "## ✅ 项目健康检查通过" in output
        assert "test1" in output
        assert "通过" in output

    def test_unknown_status_renders_question_mark(self):
        """测试未知 status 值时 render_check_item 使用 ❓ emoji"""
        # 注意：render_by_section 只渲染 pass/fail/warn/skip 状态的检查项
        # 未知状态的检查项不会出现在 render_markdown 的输出中
        # 但 render_check_item 函数本身能正确处理未知状态

        # 测试 render_check_item 对未知状态的处理
        unknown_checks = [
            {"section": "测试", "name": "invalid_status", "status": "invalid_value"},
            {"section": "测试", "name": "empty_status", "status": ""},
            {"section": "测试", "name": "error_status", "status": "error"},
        ]

        for check in unknown_checks:
            lines = render_check_item(check)
            output = "\n".join(lines)

            # 不应抛异常
            assert check["name"] in output, f"检查项名称应在输出中: {check['name']}"
            # 应使用 ❓ emoji
            assert "❓" in output, f"未知状态应使用 ❓ emoji: {check['status']}"

    def test_render_markdown_with_unknown_status_no_exception(self):
        """测试 render_markdown 处理未知状态时不抛异常"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 2},
            "checks": [
                {
                    "section": "测试",
                    "name": "unknown_status_test",
                    "status": "invalid_status_value",  # 未知状态
                    "message": "测试未知状态",
                },
                {
                    "section": "测试",
                    "name": "empty_status_test",
                    "status": "",  # 空状态
                },
            ],
            "durations": [],
        }

        # 不应抛异常，应正常渲染基本结构
        output = render_markdown(data)

        # 基本结构应存在
        assert "## ✅ 项目健康检查通过" in output
        assert "| 类型 | 数量 |" in output
        # 注意：未知状态的检查项不会出现在 pass/fail/warn/skip 分类中
        # 这是 render_by_section 的预期行为

    def test_render_with_all_optional_fields_missing(self):
        """测试所有可选字段都缺失时渲染正常"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 1, "failed": 0, "warnings": 0, "skipped": 0, "total": 1},
            "checks": [
                {
                    "section": "测试",
                    "name": "minimal_check",
                    "status": "pass",
                    # message, duration_ms, log_file, command, last_test 都缺失
                },
            ],
            # timestamp, log_dir, ci_mode 等都缺失
        }
        output = render_markdown(data)

        # 不应抛异常
        assert "## ✅ 项目健康检查通过" in output
        assert "minimal_check" in output

    def test_render_markdown_basic_structure_always_present(self):
        """测试 render_markdown 输出始终包含基本标题/表格结构"""
        test_cases = [
            # 成功场景
            {
                "success": True,
                "exit_code": 0,
                "summary": {"passed": 1, "failed": 0, "warnings": 0, "skipped": 0, "total": 1},
                "checks": [{"section": "测试", "name": "t", "status": "pass"}],
            },
            # 失败场景
            {
                "success": False,
                "exit_code": 1,
                "summary": {"passed": 0, "failed": 1, "warnings": 0, "skipped": 0, "total": 1},
                "checks": [{"section": "测试", "name": "t", "status": "fail"}],
            },
            # 空检查
            {
                "success": True,
                "exit_code": 0,
                "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
                "checks": [],
            },
            # durations 缺失
            {
                "success": True,
                "exit_code": 0,
                "summary": {"passed": 1, "failed": 0, "warnings": 0, "skipped": 0},
                "checks": [{"section": "测试", "name": "t", "status": "pass"}],
            },
        ]

        for i, data in enumerate(test_cases):
            output = render_markdown(data)

            # 基本结构检查
            assert "| 类型 | 数量 |" in output, f"用例 {i}: 缺少表格头"
            assert "|------|------|" in output, f"用例 {i}: 缺少表格分隔符"
            # 标题（成功或失败）
            assert "## ✅" in output or "## ❌" in output, f"用例 {i}: 缺少标题"
            # 提示章节
            assert "### 💡 提示" in output, f"用例 {i}: 缺少提示章节"


# ==================== 集成测试 ====================


class TestIntegration:
    """集成测试"""

    def test_markdown_is_valid(self, sample_json_data: dict):
        """测试生成的 Markdown 是有效的"""
        output = render_markdown(sample_json_data)

        # 基本 Markdown 结构检查
        # 表格头
        assert "| 类型 | 数量 |" in output
        assert "|------|------|" in output

        # 折叠块
        assert "<details>" in output
        assert "</details>" in output
        assert "<summary>" in output
        assert "</summary>" in output

    def test_all_checks_accounted(self, sample_json_data: dict):
        """测试所有检查项都被统计"""
        output = render_markdown(sample_json_data)

        # 所有检查项名称应出现在输出中
        for check in sample_json_data["checks"]:
            assert check["name"] in output
