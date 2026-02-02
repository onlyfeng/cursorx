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

# 从统一契约模块导入
from core.check_all_contract import (  # noqa: E402
    VALID_STATUSES,
    get_expected_schema,
    validate_json_against_schema,
)

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CHECK_ALL_SCRIPT = SCRIPTS_DIR / "check_all.sh"


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


# ==================== 轻量 JSON 契约测试（CI 必跑，防止 PR 解析失败）====================


@pytest.mark.skipif(not CHECK_ALL_SCRIPT.exists(), reason="check_all.sh 脚本不存在")
class TestMinimalJsonContract:
    """轻量 JSON 契约测试（非 slow，CI 必跑）。

    使用 --mode minimal 运行最轻量的检查（仅 Python 版本 + 基础验证）。
    直接使用 json.loads(stdout) 验证输出是纯 JSON，不使用宽松的 find('{') 策略。
    这确保 PR workflow 能够正确解析 JSON 输出。

    耗时约 2-3 秒，无额外依赖要求。
    """

    def test_json_output_is_pure_json(self):
        """测试 --json --mode minimal 输出是纯 JSON（严格解析）

        关键点：
        1. 直接 json.loads(stdout)，不使用 find('{') 宽松策略
        2. 确保输出无前缀杂质（echo/debug 输出等）
        3. 验证 schema 符合契约定义
        """
        result = subprocess.run(
            [
                "bash",
                str(CHECK_ALL_SCRIPT),
                "--json",
                "--mode",
                "minimal",  # 最轻量的检查模式（仅 Python 版本 + 基础验证）
            ],
            capture_output=True,
            text=True,
            timeout=30,  # minimal 模式通常 2-3 秒完成
            cwd=str(PROJECT_ROOT),
        )

        # 严格解析：直接 json.loads(stdout)，不使用 find('{') 宽松策略
        # 如果输出有任何前缀杂质，这里会失败并暴露问题
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            # 提供详细的错误信息帮助调试
            stdout_preview = result.stdout[:500] if len(result.stdout) > 500 else result.stdout
            pytest.fail(
                f"JSON 解析失败（stdout 应为纯 JSON，无前缀杂质）:\n"
                f"错误: {e}\n"
                f"stdout 前 500 字符: {stdout_preview}\n"
                f"stderr: {result.stderr}"
            )

        # 验证 schema 符合契约定义
        schema = get_expected_schema()
        errors = validate_json_against_schema(data, schema)
        assert errors == [], f"JSON schema 验证错误: {errors}"

        # 验证 minimal 模式的基本结构
        assert "success" in data, "缺少 success 字段"
        assert "exit_code" in data, "缺少 exit_code 字段"
        assert isinstance(data["checks"], list), "checks 应为列表"
        assert isinstance(data["summary"], dict), "summary 应为字典"

        # 验证 status 枚举值（使用统一常量）
        for check in data["checks"]:
            assert check.get("status") in VALID_STATUSES, (
                f"无效的 status 值: {check.get('status')}，允许值: {VALID_STATUSES}"
            )

    def test_json_output_exit_code_consistency(self):
        """测试 JSON 输出的 exit_code 与实际退出码一致"""
        result = subprocess.run(
            [
                "bash",
                str(CHECK_ALL_SCRIPT),
                "--json",
                "--mode",
                "minimal",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )

        # 严格解析
        data = json.loads(result.stdout)

        # exit_code 应与实际退出码一致
        assert data["exit_code"] == result.returncode, (
            f"exit_code 不一致: JSON 中为 {data['exit_code']}，实际退出码为 {result.returncode}"
        )

        # success 字段应与 exit_code 一致
        if data["exit_code"] == 0:
            assert data["success"] is True, "exit_code=0 时 success 应为 True"
        else:
            assert data["success"] is False, "exit_code!=0 时 success 应为 False"


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
