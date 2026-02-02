#!/usr/bin/env python3
"""端到端导入测试

验证所有核心模块可以正确导入，run.py 可运行，且无循环导入问题。
这些测试确保项目的基本可用性和依赖完整性。

测试内容:
1. test_all_core_modules_import - 测试所有核心模块可导入
2. test_entry_script_runnable - 测试 run.py 入口脚本可执行
3. test_no_circular_imports - 检测循环导入问题
"""

import importlib
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# 核心模块列表
# ============================================================

# 项目的核心模块目录
CORE_MODULES = [
    "agents",
    "coordinator",
    "core",
    "cursor",
    "indexing",
    "knowledge",
    "process",
    "tasks",
]

# 特定的关键模块（必须能导入）
CRITICAL_MODULES = [
    "agents.planner",
    "agents.worker",
    "agents.reviewer",
    "agents.committer",
    "coordinator.orchestrator",
    "core.base",
    "core.state",
    "core.message",
    "cursor.client",
    "cursor.executor",
    "cursor.streaming",
    "tasks.task",
    "tasks.queue",
]


# ============================================================
# 测试函数
# ============================================================


class TestCoreModulesImport:
    """测试核心模块导入"""

    def test_all_core_modules_import(self):
        """测试所有核心模块可以正确导入

        遍历 CORE_MODULES 中的所有包，递归导入其子模块，
        确保没有导入错误（如缺少依赖、语法错误等）。
        """
        imported_modules = []
        failed_modules = []

        def import_submodules(package_name: str):
            """递归导入包中的所有子模块"""
            try:
                package = importlib.import_module(package_name)
                imported_modules.append(package_name)
            except Exception as e:
                failed_modules.append((package_name, str(e)))
                return

            # 检查是否为包（有 __path__ 属性）
            if not hasattr(package, "__path__"):
                return

            # 递归导入子模块
            for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
                try:
                    importlib.import_module(modname)
                    imported_modules.append(modname)
                except Exception as e:
                    failed_modules.append((modname, str(e)))

        # 导入所有顶级模块
        for module_name in CORE_MODULES:
            import_submodules(module_name)

        # 报告结果
        print(f"\n成功导入 {len(imported_modules)} 个模块")
        for mod in imported_modules:
            print(f"  [OK] {mod}")

        if failed_modules:
            print(f"\n导入失败 {len(failed_modules)} 个模块:")
            for mod, err in failed_modules:
                print(f"  [FAIL] {mod}: {err}")

            # CI 环境中某些可选依赖可能未安装，仅报告但不失败
            if os.environ.get("CI"):
                # 检查是否有关键模块失败
                failed_names = {m[0] for m in failed_modules}
                critical_failed = [m for m in CRITICAL_MODULES if m in failed_names]
                if critical_failed:
                    pytest.fail(f"关键模块导入失败: {critical_failed}")
                else:
                    print("  (CI 环境，非关键模块导入失败可忽略)")
            else:
                pytest.fail(f"模块导入失败: {failed_modules}")

        assert len(imported_modules) > 0, "没有成功导入任何模块"
        print(f"\n所有 {len(imported_modules)} 个模块导入成功")

    def test_critical_modules_import(self):
        """测试关键模块可以正确导入

        确保项目核心功能所依赖的关键模块都能正常导入。
        """
        failed = []

        for module_name in CRITICAL_MODULES:
            try:
                importlib.import_module(module_name)
                print(f"  [OK] {module_name}")
            except Exception as e:
                failed.append((module_name, str(e)))
                print(f"  [FAIL] {module_name}: {e}")

        if failed:
            pytest.fail(f"关键模块导入失败: {failed}")

        print(f"\n所有 {len(CRITICAL_MODULES)} 个关键模块导入成功")


class TestEntryScriptRunnable:
    """测试入口脚本可运行"""

    def test_entry_script_runnable(self):
        """测试 run.py 入口脚本可执行

        验证 run.py 可以被 Python 解释器加载而不出错。
        使用 --help 参数测试脚本可以正常启动。
        """
        run_py_path = project_root / "run.py"

        assert run_py_path.exists(), f"run.py 不存在: {run_py_path}"

        # 测试语法正确性（编译模块）
        try:
            with open(run_py_path, "rb") as f:
                source = f.read()
            compile(source, str(run_py_path), "exec")
            print("  [OK] run.py 语法检查通过")
        except SyntaxError as e:
            pytest.fail(f"run.py 语法错误: {e}")

        # 测试可以导入 run 模块
        try:
            import run  # noqa: F401

            print("  [OK] run 模块可导入")
        except Exception as e:
            pytest.fail(f"run 模块导入失败: {e}")

        # 测试 run.py --help 可以正常执行
        try:
            result = subprocess.run(
                [sys.executable, str(run_py_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root),
            )

            # --help 应该返回 0 或显示帮助信息
            if result.returncode == 0:
                print("  [OK] run.py --help 执行成功")
                assert "usage" in result.stdout.lower() or "cursorx" in result.stdout.lower(), "帮助信息不完整"
            else:
                # 有些脚本可能因为缺少参数而返回非零
                # 但只要不是语法/导入错误就可以接受
                if "SyntaxError" in result.stderr or "ImportError" in result.stderr:
                    pytest.fail(f"run.py 执行错误: {result.stderr}")
                print(f"  [WARN] run.py --help 返回码: {result.returncode}")
                print(f"         stdout: {result.stdout[:200]}")
                print(f"         stderr: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            pytest.fail("run.py --help 执行超时")
        except Exception as e:
            pytest.fail(f"run.py 执行异常: {e}")

    def test_run_module_components(self):
        """测试 run.py 模块的关键组件"""
        from run import (
            MODE_ALIASES,
            RunMode,
            TaskAnalyzer,
        )

        # 验证 RunMode 枚举
        assert RunMode.BASIC.value == "basic"
        assert RunMode.MP.value == "mp"
        assert RunMode.KNOWLEDGE.value == "knowledge"
        assert RunMode.ITERATE.value == "iterate"
        assert RunMode.AUTO.value == "auto"
        print("  [OK] RunMode 枚举定义正确")

        # 验证模式别名映射
        assert MODE_ALIASES["basic"] == RunMode.BASIC
        assert MODE_ALIASES["mp"] == RunMode.MP
        assert MODE_ALIASES["parallel"] == RunMode.MP
        assert MODE_ALIASES["knowledge"] == RunMode.KNOWLEDGE
        assert MODE_ALIASES["iterate"] == RunMode.ITERATE
        print("  [OK] MODE_ALIASES 映射正确")

        # 验证 TaskAnalyzer 可实例化
        analyzer = TaskAnalyzer(use_agent=False)
        assert analyzer is not None
        print("  [OK] TaskAnalyzer 可实例化")


class TestNoCircularImports:
    """测试无循环导入"""

    def test_no_circular_imports(self):
        """检测循环导入问题

        通过在新的子进程中逐个导入模块来检测循环导入。
        如果存在循环导入，导入会失败或挂起。
        """
        # 需要检测的模块组合（容易产生循环依赖的模块对）
        module_pairs = [
            ("agents", "core"),
            ("agents", "coordinator"),
            ("coordinator", "agents"),
            ("core", "tasks"),
            ("cursor", "agents"),
            ("knowledge", "indexing"),
            ("process", "coordinator"),
        ]

        failed_pairs = []

        for mod1, mod2 in module_pairs:
            # 在子进程中测试导入顺序
            test_code = f"""
import sys
sys.path.insert(0, '{project_root}')
try:
    import {mod1}
    import {mod2}
    print('OK')
except ImportError as e:
    print(f'ImportError: {{e}}')
    sys.exit(1)
except RecursionError as e:
    print(f'RecursionError (circular import): {{e}}')
    sys.exit(2)
except Exception as e:
    print(f'Error: {{e}}')
    sys.exit(3)
"""
            try:
                result = subprocess.run(
                    [sys.executable, "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(project_root),
                )

                if result.returncode == 0:
                    print(f"  [OK] {mod1} -> {mod2}")
                elif result.returncode == 2:
                    # RecursionError - 明确的循环导入
                    failed_pairs.append((mod1, mod2, "循环导入"))
                    print(f"  [FAIL] {mod1} -> {mod2}: 循环导入")
                else:
                    # 其他错误，在 CI 环境可能是依赖问题
                    error_msg = result.stdout.strip() or result.stderr.strip()
                    if os.environ.get("CI"):
                        print(f"  [WARN] {mod1} -> {mod2}: {error_msg[:100]}")
                    else:
                        failed_pairs.append((mod1, mod2, error_msg))
                        print(f"  [FAIL] {mod1} -> {mod2}: {error_msg}")

            except subprocess.TimeoutExpired:
                failed_pairs.append((mod1, mod2, "超时（可能是循环导入死锁）"))
                print(f"  [FAIL] {mod1} -> {mod2}: 超时")

        if failed_pairs:
            pytest.fail(f"检测到循环导入问题: {failed_pairs}")

        print(f"\n所有 {len(module_pairs)} 个模块对检查通过，无循环导入")

    def test_deep_import_chain(self):
        """测试深层导入链

        验证常见的深层导入链不会产生循环依赖。
        """
        # 典型的导入链
        import_chains = [
            # 从入口到核心
            ["run", "coordinator", "agents", "core"],
            # 从 coordinator 到各子系统
            ["coordinator.orchestrator", "agents.planner", "cursor.client"],
            # 知识库链
            ["knowledge", "indexing", "cursor"],
        ]

        for chain in import_chains:
            chain_str = " -> ".join(chain)
            try:
                for module_name in chain:
                    importlib.import_module(module_name)
                print(f"  [OK] 导入链: {chain_str}")
            except Exception as e:
                # CI 环境可能缺少某些依赖
                if os.environ.get("CI") and "ModuleNotFoundError" in str(type(e).__name__):
                    print(f"  [WARN] 导入链 {chain_str}: {e}")
                else:
                    pytest.fail(f"导入链失败 {chain_str}: {e}")


class TestModuleIntegrity:
    """测试模块完整性"""

    def test_all_init_files_exist(self):
        """验证所有包都有 __init__.py 文件"""
        missing_init = []

        for module_name in CORE_MODULES:
            module_path = project_root / module_name
            init_file = module_path / "__init__.py"

            if module_path.is_dir():
                if not init_file.exists():
                    missing_init.append(module_name)
                else:
                    print(f"  [OK] {module_name}/__init__.py")

        if missing_init:
            pytest.fail(f"缺少 __init__.py: {missing_init}")

        print(f"\n所有 {len(CORE_MODULES)} 个模块都有 __init__.py")

    def test_no_syntax_errors(self):
        """检查所有 Python 文件语法正确"""
        syntax_errors = []

        for module_name in CORE_MODULES:
            module_path = project_root / module_name

            if not module_path.is_dir():
                continue

            for py_file in module_path.rglob("*.py"):
                try:
                    with open(py_file, "rb") as f:
                        source = f.read()
                    compile(source, str(py_file), "exec")
                except SyntaxError as e:
                    syntax_errors.append((str(py_file), str(e)))

        if syntax_errors:
            for path, error in syntax_errors:
                print(f"  [FAIL] {path}: {error}")
            pytest.fail(f"语法错误: {len(syntax_errors)} 个文件")

        print("  [OK] 所有 Python 文件语法正确")


# ============================================================
# 主函数（直接运行支持）
# ============================================================


def main():
    """直接运行测试"""
    print("\n" + "=" * 60)
    print("端到端导入测试")
    print("=" * 60 + "\n")

    # 运行测试
    exit_code = pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-x",  # 遇到第一个失败就停止
        ]
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
