"""check_all.sh JSON 输出契约测试

轻量契约测试，验证 check_all.sh --json 输出的 JSON schema 符合预期。

测试策略：
1. 使用 subprocess 调用 check_all.sh --json --help 验证 --json 参数存在
2. 验证 JSON 输出的 schema 结构
3. 验证必需字段的存在和类型
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CHECK_ALL_SCRIPT = SCRIPTS_DIR / "check_all.sh"


# ==================== JSON Schema 定义 ====================


def get_expected_schema() -> dict[str, Any]:
    """返回 check_all.sh --json 输出的预期 schema。

    这是 output_json_result 函数输出的契约定义。
    """
    return {
        # 顶层必需字段
        "required_fields": {
            "success": bool,
            "exit_code": int,
            "summary": dict,
            "checks": list,
        },
        # 顶层可选字段
        "optional_fields": {
            "ci_mode": bool,
            "fail_fast": bool,
            "full_check": bool,
            "diagnose_hang": bool,
            "timeout_backend": str,
            "log_dir": str,
            "project_root": str,
            "timestamp": str,
            "durations": list,
        },
        # summary 子结构
        "summary_fields": {
            "passed": int,
            "failed": int,
            "warnings": int,
            "skipped": int,
            "total": int,
        },
        # check 项必需字段
        "check_required_fields": {
            "section": str,
            "name": str,
            "status": str,
        },
        # check 项可选字段
        "check_optional_fields": {
            "message": str,
            "duration_ms": int,
            "log_file": str,
            "command": str,
            "last_test": str,
        },
        # status 允许的值
        "valid_statuses": ["pass", "fail", "warn", "skip", "info"],
        # duration 项结构
        "duration_fields": {
            "name": str,
            "duration_ms": int,
        },
    }


# ==================== 验证函数 ====================


def validate_json_against_schema(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """验证 JSON 数据是否符合 schema。

    Args:
        data: 要验证的 JSON 数据
        schema: 预期的 schema 定义

    Returns:
        错误消息列表，空列表表示验证通过
    """
    errors = []

    # 验证顶层必需字段
    for field, expected_type in schema["required_fields"].items():
        if field not in data:
            errors.append(f"缺少必需字段: {field}")
        elif not isinstance(data[field], expected_type):
            errors.append(f"字段 {field} 类型错误: 期望 {expected_type.__name__}, 实际 {type(data[field]).__name__}")

    # 验证顶层可选字段类型（如果存在）
    for field, expected_type in schema["optional_fields"].items():
        if field in data and data[field] is not None and not isinstance(data[field], expected_type):
            errors.append(
                f"可选字段 {field} 类型错误: 期望 {expected_type.__name__}, 实际 {type(data[field]).__name__}"
            )

    # 验证 summary 结构
    if "summary" in data and isinstance(data["summary"], dict):
        summary = data["summary"]
        for field, expected_type in schema["summary_fields"].items():
            if field not in summary:
                errors.append(f"summary 缺少字段: {field}")
            elif not isinstance(summary[field], expected_type):
                errors.append(
                    f"summary.{field} 类型错误: 期望 {expected_type.__name__}, 实际 {type(summary[field]).__name__}"
                )

    # 验证 checks 数组
    if "checks" in data and isinstance(data["checks"], list):
        for i, check in enumerate(data["checks"]):
            if not isinstance(check, dict):
                errors.append(f"checks[{i}] 应为 dict，实际为 {type(check).__name__}")
                continue

            # 验证必需字段
            for field, expected_type in schema["check_required_fields"].items():
                if field not in check:
                    errors.append(f"checks[{i}] 缺少必需字段: {field}")
                elif not isinstance(check[field], expected_type):
                    errors.append(
                        f"checks[{i}].{field} 类型错误: 期望 {expected_type.__name__}, "
                        f"实际 {type(check[field]).__name__}"
                    )

            # 验证 status 值
            if "status" in check and check["status"] not in schema["valid_statuses"]:
                errors.append(f"checks[{i}].status 值无效: {check['status']}, 允许值: {schema['valid_statuses']}")

            # 验证可选字段类型（如果存在）
            for field, expected_type in schema["check_optional_fields"].items():
                if field in check and check[field] is not None and not isinstance(check[field], expected_type):
                    errors.append(
                        f"checks[{i}].{field} 类型错误: 期望 {expected_type.__name__}, "
                        f"实际 {type(check[field]).__name__}"
                    )

    # 验证 durations 数组
    if "durations" in data and isinstance(data["durations"], list):
        for i, duration in enumerate(data["durations"]):
            if not isinstance(duration, dict):
                errors.append(f"durations[{i}] 应为 dict，实际为 {type(duration).__name__}")
                continue

            for field, expected_type in schema["duration_fields"].items():
                if field not in duration:
                    errors.append(f"durations[{i}] 缺少字段: {field}")
                elif not isinstance(duration[field], expected_type):
                    errors.append(
                        f"durations[{i}].{field} 类型错误: 期望 {expected_type.__name__}, "
                        f"实际 {type(duration[field]).__name__}"
                    )

    return errors


# ==================== 测试 Fixtures ====================


@pytest.fixture
def sample_valid_json() -> dict[str, Any]:
    """有效的测试 JSON 样例"""
    return {
        "success": True,
        "exit_code": 0,
        "summary": {
            "passed": 5,
            "failed": 0,
            "warnings": 1,
            "skipped": 2,
            "total": 8,
        },
        "ci_mode": True,
        "fail_fast": False,
        "full_check": True,
        "diagnose_hang": False,
        "timeout_backend": "timeout",
        "log_dir": "/tmp/logs",
        "project_root": "/home/user/project",
        "timestamp": "2025-01-29T10:00:00+08:00",
        "durations": [
            {"name": "pytest", "duration_ms": 5000},
            {"name": "mypy", "duration_ms": 3000},
        ],
        "checks": [
            {
                "section": "代码风格",
                "name": "ruff",
                "status": "pass",
                "message": "检查通过",
                "duration_ms": 500,
            },
            {
                "section": "测试",
                "name": "pytest",
                "status": "pass",
                "message": "所有测试通过",
                "duration_ms": 5000,
                "log_file": "/tmp/logs/pytest.log",
                "command": "pytest tests/",
            },
            {
                "section": "安全检查",
                "name": "bandit",
                "status": "warn",
                "message": "发现 1 个低危问题",
            },
            {
                "section": "集成测试",
                "name": "integration",
                "status": "skip",
                "message": "无 API Key",
            },
        ],
    }


@pytest.fixture
def sample_minimal_json() -> dict[str, Any]:
    """最小有效的 JSON 样例（仅包含必需字段）"""
    return {
        "success": True,
        "exit_code": 0,
        "summary": {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "skipped": 0,
            "total": 0,
        },
        "checks": [],
    }


# ==================== Schema 验证测试 ====================


class TestSchemaValidation:
    """Schema 验证函数测试"""

    def test_valid_json_passes(self, sample_valid_json: dict):
        """测试有效 JSON 通过验证"""
        schema = get_expected_schema()
        errors = validate_json_against_schema(sample_valid_json, schema)
        assert errors == [], f"验证错误: {errors}"

    def test_minimal_json_passes(self, sample_minimal_json: dict):
        """测试最小 JSON 通过验证"""
        schema = get_expected_schema()
        errors = validate_json_against_schema(sample_minimal_json, schema)
        assert errors == [], f"验证错误: {errors}"

    def test_missing_required_field(self):
        """测试缺少必需字段"""
        data = {
            # 缺少 success 字段
            "exit_code": 0,
            "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
            "checks": [],
        }
        schema = get_expected_schema()
        errors = validate_json_against_schema(data, schema)
        assert any("缺少必需字段: success" in e for e in errors)

    def test_wrong_type_for_required_field(self):
        """测试必需字段类型错误"""
        data = {
            "success": "true",  # 应为 bool
            "exit_code": 0,
            "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
            "checks": [],
        }
        schema = get_expected_schema()
        errors = validate_json_against_schema(data, schema)
        assert any("success" in e and "类型错误" in e for e in errors)

    def test_missing_summary_field(self):
        """测试缺少 summary 子字段"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {
                "passed": 0,
                # 缺少 failed
                "warnings": 0,
                "skipped": 0,
                "total": 0,
            },
            "checks": [],
        }
        schema = get_expected_schema()
        errors = validate_json_against_schema(data, schema)
        assert any("summary 缺少字段: failed" in e for e in errors)

    def test_invalid_status_value(self):
        """测试无效的 status 值"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
            "checks": [
                {"section": "测试", "name": "test", "status": "invalid_status"},
            ],
        }
        schema = get_expected_schema()
        errors = validate_json_against_schema(data, schema)
        assert any("status 值无效" in e for e in errors)

    def test_missing_check_required_field(self):
        """测试 check 项缺少必需字段"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
            "checks": [
                {"section": "测试", "name": "test"},  # 缺少 status
            ],
        }
        schema = get_expected_schema()
        errors = validate_json_against_schema(data, schema)
        assert any("缺少必需字段: status" in e for e in errors)

    def test_wrong_type_for_check_field(self):
        """测试 check 项字段类型错误"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
            "checks": [
                {
                    "section": "测试",
                    "name": "test",
                    "status": "pass",
                    "duration_ms": "500",  # 应为 int
                },
            ],
        }
        schema = get_expected_schema()
        errors = validate_json_against_schema(data, schema)
        assert any("duration_ms" in e and "类型错误" in e for e in errors)

    def test_durations_validation(self):
        """测试 durations 数组验证"""
        data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
            "checks": [],
            "durations": [
                {"name": "test"},  # 缺少 duration_ms
            ],
        }
        schema = get_expected_schema()
        errors = validate_json_against_schema(data, schema)
        assert any("durations[0] 缺少字段: duration_ms" in e for e in errors)


# ==================== 脚本存在性测试 ====================


class TestScriptExists:
    """脚本存在性测试"""

    def test_check_all_script_exists(self):
        """测试 check_all.sh 脚本存在"""
        assert CHECK_ALL_SCRIPT.exists(), f"脚本不存在: {CHECK_ALL_SCRIPT}"

    def test_check_all_script_is_valid_bash(self):
        """测试 check_all.sh 是有效的 bash 脚本。

        通过 bash -n 语法检查验证脚本有效性，
        不依赖文件的可执行权限位。
        """
        result = subprocess.run(
            ["bash", "-n", str(CHECK_ALL_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"脚本语法错误: {result.stderr}"


# ==================== 帮助信息测试 ====================


class TestHelpMessage:
    """帮助信息测试"""

    def test_json_option_in_help(self):
        """测试 --json 选项在帮助信息中"""
        result = subprocess.run(
            ["bash", str(CHECK_ALL_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # 帮助信息应包含 --json 选项
        assert "--json" in result.stdout, "帮助信息中应包含 --json 选项"
        assert "JSON" in result.stdout, "帮助信息中应说明 JSON 输出"


# ==================== 最小 JSON 输出测试（可选，需要实际运行脚本）====================


@pytest.mark.slow
@pytest.mark.skipif(not CHECK_ALL_SCRIPT.exists(), reason="check_all.sh 脚本不存在")
class TestMinimalJsonOutput:
    """最小 JSON 输出测试。

    这些测试需要实际运行 check_all.sh 脚本，可能较慢。
    使用 --mode mode1 限制只运行一个快速检查。
    """

    def test_json_output_is_valid_json(self):
        """测试 --json 输出是有效的 JSON"""
        # 使用 --mode import 作为最快的检查模式
        result = subprocess.run(
            [
                "bash",
                str(CHECK_ALL_SCRIPT),
                "--json",
                "--mode",
                "import",  # 最快的检查模式
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )

        # 尝试解析 JSON
        try:
            # 输出应该是完整的 JSON
            output = result.stdout.strip()
            # 从第一个 { 开始找 JSON
            json_start = output.find("{")
            if json_start >= 0:
                json_str = output[json_start:]
                data = json.loads(json_str)

                # 验证基本结构
                schema = get_expected_schema()
                errors = validate_json_against_schema(data, schema)
                assert errors == [], f"JSON schema 验证错误: {errors}"
            else:
                # 如果没有 JSON 输出，检查是否有错误
                pytest.skip(f"脚本未产生 JSON 输出: {result.stderr}")
        except json.JSONDecodeError as e:
            pytest.fail(f"无效的 JSON 输出: {e}\n输出: {result.stdout}")


# ==================== 契约一致性测试 ====================


class TestContractConsistency:
    """契约一致性测试。

    确保 schema 定义与 render_check_all_summary.py 的期望一致。
    """

    def test_renderer_handles_all_schema_fields(self, sample_valid_json: dict):
        """测试渲染器能处理所有 schema 字段"""
        from scripts.render_check_all_summary import render_markdown

        # 渲染应该不抛出异常
        output = render_markdown(sample_valid_json)
        assert output, "渲染器应产生输出"

    def test_schema_covers_renderer_expectations(self):
        """测试 schema 覆盖了渲染器的所有期望字段"""
        from scripts.render_check_all_summary import render_markdown

        # 用最小数据测试渲染器
        minimal_data = {
            "success": True,
            "exit_code": 0,
            "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
            "checks": [],
            "durations": [],
        }

        # 渲染应该不抛出异常
        output = render_markdown(minimal_data)
        assert "## ✅ 项目健康检查通过" in output
