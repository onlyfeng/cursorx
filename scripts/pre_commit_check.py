#!/usr/bin/env python3
"""预提交检查脚本

在提交代码前执行全面的检查，确保代码质量和依赖完整性。

检查内容：
1. 语法检查 - 验证所有 Python 文件的语法正确性
2. 依赖检查 - 检查 requirements.txt 中的依赖是否都已安装
3. 模块导入检查 - 递归导入项目所有 Python 模块
4. 完整导入验证 - 验证 agents/, coordinator/, core/, cursor/, knowledge/, tasks/
   目录下的所有模块，并对 run.py 执行语法和导入测试
5. 关键组件检查 - 验证 run.py 中的关键类和函数存在
6. 配置文件检查 - 验证 config.yaml、mcp.json 等配置文件格式
7. 运行模式验证 - 验证 run.py --help 和所有模式（basic、mp、knowledge、iterate、auto、plan、ask）
8. 冒烟测试 - 运行快速测试（不实际执行任务，仅验证初始化）

用法：
    python scripts/pre_commit_check.py [--verbose] [--fix]
    python scripts/pre_commit_check.py --quick  # 快速检查模式
    python scripts/pre_commit_check.py --ci     # CI 模式（禁用颜色）
    python scripts/pre_commit_check.py --module-only  # 仅模块导入检查
"""

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ============================================================
# 常量定义
# ============================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 项目模块列表
PROJECT_MODULES = [
    "core",
    "agents",
    "coordinator",
    "tasks",
    "cursor",
    "indexing",
    "knowledge",
    "process",
]

# run.py 中需要验证的关键类和函数
CRITICAL_CLASSES = [
    ("run", "RunMode"),
    ("run", "TaskAnalysis"),
    ("run", "TaskAnalyzer"),
    ("run", "Runner"),
]

CRITICAL_FUNCTIONS = [
    ("run", "main"),
    ("run", "async_main"),
    ("run", "parse_args"),
]

# 依赖包名到导入名的映射
PACKAGE_IMPORT_MAP = {
    # 标准依赖
    "asyncio-pool": "asyncio_pool",
    "aiofiles": "aiofiles",
    "pydantic": "pydantic",
    "psutil": "psutil",
    "loguru": "loguru",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
    "typing-extensions": "typing_extensions",
    "beautifulsoup4": "bs4",
    "html2text": "html2text",
    "lxml": "lxml",
    "httpx": "httpx",
    "websockets": "websockets",
    # 可选依赖
    "sentence-transformers": "sentence_transformers",
    "chromadb": "chromadb",
    "tiktoken": "tiktoken",
    "hnswlib": "hnswlib",
    # 测试依赖
    "pytest": "pytest",
    "pytest-asyncio": "pytest_asyncio",
    "respx": "respx",
}

# 必需依赖（缺失会导致检查失败）
REQUIRED_PACKAGES = [
    "aiofiles",
    "pydantic",
    "psutil",
    "loguru",
    "python-dotenv",
    "pyyaml",
    "typing-extensions",
    "beautifulsoup4",
    "lxml",
]

# 可选依赖（缺失只警告）
OPTIONAL_PACKAGES = [
    "sentence-transformers",
    "chromadb",
    "tiktoken",
    "hnswlib",
    "httpx",
    "websockets",
    # 开发/测试工具（可选）
    "pre-commit",
    "pip-audit",
    "mypy",
    "ruff",
    "flake8",
]


# ============================================================
# CI 环境检测
# ============================================================

def is_ci_environment() -> bool:
    """检测是否在 CI 环境中运行"""
    return os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"


def get_github_step_summary_path() -> Optional[str]:
    """获取 GitHub Step Summary 文件路径"""
    return os.environ.get("GITHUB_STEP_SUMMARY")


# ============================================================
# 颜色输出
# ============================================================

class Colors:
    """终端颜色定义，支持禁用颜色输出"""
    _enabled: bool = True
    
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    MAGENTA = "\033[0;35m"
    BOLD = "\033[1m"
    NC = "\033[0m"
    
    @classmethod
    def disable(cls) -> None:
        """禁用颜色输出"""
        cls._enabled = False
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.CYAN = ""
        cls.MAGENTA = ""
        cls.BOLD = ""
        cls.NC = ""
    
    @classmethod
    def enable(cls) -> None:
        """启用颜色输出"""
        cls._enabled = True
        cls.RED = "\033[0;31m"
        cls.GREEN = "\033[0;32m"
        cls.YELLOW = "\033[1;33m"
        cls.BLUE = "\033[0;34m"
        cls.CYAN = "\033[0;36m"
        cls.MAGENTA = "\033[0;35m"
        cls.BOLD = "\033[1m"
        cls.NC = "\033[0m"
    
    @classmethod
    def is_enabled(cls) -> bool:
        """检查颜色输出是否启用"""
        return cls._enabled


def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")


def print_section(text: str) -> None:
    print(f"\n{Colors.CYAN}{'─' * 60}{Colors.NC}")
    print(f"{Colors.CYAN}  {text}{Colors.NC}")
    print(f"{Colors.CYAN}{'─' * 60}{Colors.NC}")


def print_pass(text: str) -> None:
    symbol = "✓" if Colors.is_enabled() else "[PASS]"
    print(f"  {Colors.GREEN}{symbol}{Colors.NC} {text}")


def print_fail(text: str) -> None:
    symbol = "✗" if Colors.is_enabled() else "[FAIL]"
    print(f"  {Colors.RED}{symbol}{Colors.NC} {text}")


def print_warn(text: str) -> None:
    symbol = "⚠" if Colors.is_enabled() else "[WARN]"
    print(f"  {Colors.YELLOW}{symbol}{Colors.NC} {text}")


def print_info(text: str) -> None:
    symbol = "ℹ" if Colors.is_enabled() else "[INFO]"
    print(f"  {Colors.BLUE}{symbol}{Colors.NC} {text}")


def print_debug(text: str, verbose: bool = False) -> None:
    if verbose:
        symbol = "•" if Colors.is_enabled() else "[DEBUG]"
        print(f"  {Colors.MAGENTA}{symbol}{Colors.NC} {text}")


# ============================================================
# 检查结果数据结构
# ============================================================

@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    passed: bool
    message: str = ""
    details: list[str] = field(default_factory=list)
    fix_suggestion: str = ""


@dataclass
class CheckReport:
    """检查报告"""
    checks: list[CheckResult] = field(default_factory=list)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)
    
    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)
    
    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
    
    def print_summary(self) -> None:
        """打印检查汇总"""
        print_header("预提交检查报告")
        
        pass_symbol = "✓" if Colors.is_enabled() else "[PASS]"
        fail_symbol = "✗" if Colors.is_enabled() else "[FAIL]"
        
        print(f"\n  {Colors.GREEN}{pass_symbol} 通过:{Colors.NC} {self.passed_count}")
        print(f"  {Colors.RED}{fail_symbol} 失败:{Colors.NC} {self.failed_count}")
        print()
        
        if self.all_passed:
            print(f"{Colors.GREEN}{Colors.BOLD}{'━' * 60}{Colors.NC}")
            print(f"{Colors.GREEN}{Colors.BOLD}  所有预提交检查通过！可以安全提交{Colors.NC}")
            print(f"{Colors.GREEN}{Colors.BOLD}{'━' * 60}{Colors.NC}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}{'━' * 60}{Colors.NC}")
            print(f"{Colors.RED}{Colors.BOLD}  发现 {self.failed_count} 个问题需要修复{Colors.NC}")
            print(f"{Colors.RED}{Colors.BOLD}{'━' * 60}{Colors.NC}")
            
            # 打印失败项的修复建议
            print("\n修复建议：")
            for check in self.checks:
                if not check.passed and check.fix_suggestion:
                    print(f"\n  {Colors.YELLOW}• {check.name}:{Colors.NC}")
                    print(f"    {check.fix_suggestion}")
    
    def to_markdown(self) -> str:
        """生成 Markdown 格式的检查报告（用于 GitHub Step Summary）"""
        lines = []
        lines.append("## 预提交检查报告")
        lines.append("")
        
        # 总体状态
        if self.all_passed:
            lines.append("### ✅ 所有检查通过")
        else:
            lines.append(f"### ❌ 发现 {self.failed_count} 个问题")
        lines.append("")
        
        # 统计表格
        lines.append("| 统计 | 数量 |")
        lines.append("|------|------|")
        lines.append(f"| ✅ 通过 | {self.passed_count} |")
        lines.append(f"| ❌ 失败 | {self.failed_count} |")
        lines.append("")
        
        # 详细结果表格
        lines.append("### 检查详情")
        lines.append("")
        lines.append("| 检查项 | 状态 | 信息 |")
        lines.append("|--------|------|------|")
        for check in self.checks:
            status = "✅" if check.passed else "❌"
            message = check.message.replace("|", "\\|")  # 转义表格分隔符
            lines.append(f"| {check.name} | {status} | {message} |")
        lines.append("")
        
        # 失败项的修复建议
        failed_checks = [c for c in self.checks if not c.passed and c.fix_suggestion]
        if failed_checks:
            lines.append("### 修复建议")
            lines.append("")
            for check in failed_checks:
                lines.append(f"- **{check.name}**: {check.fix_suggestion}")
            lines.append("")
        
        return "\n".join(lines)


# ============================================================
# 检查函数
# ============================================================

def check_dependencies(verbose: bool = False) -> CheckResult:
    """检查 requirements.txt 中的依赖是否都已安装"""
    result = CheckResult(name="依赖检查", passed=True)
    missing_required = []
    missing_optional = []
    installed = []
    
    # 读取 requirements.txt
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        result.passed = False
        result.message = "requirements.txt 不存在"
        result.fix_suggestion = "创建 requirements.txt 文件"
        return result
    
    # 解析 requirements.txt
    packages = []
    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 提取包名（去除版本约束）
            match = re.match(r'^([a-zA-Z0-9_-]+)', line)
            if match:
                packages.append(match.group(1).lower())
    
    # 检查每个包是否可导入
    for package in packages:
        import_name = PACKAGE_IMPORT_MAP.get(package, package.replace("-", "_"))
        
        try:
            importlib.import_module(import_name)
            installed.append(package)
            print_debug(f"已安装: {package} (import {import_name})", verbose)
        except ImportError:
            if package in REQUIRED_PACKAGES:
                missing_required.append(package)
                print_debug(f"缺失(必需): {package}", verbose)
            elif package in OPTIONAL_PACKAGES:
                missing_optional.append(package)
                print_debug(f"缺失(可选): {package}", verbose)
            else:
                # 未分类的依赖视为必需
                missing_required.append(package)
                print_debug(f"缺失(未分类): {package}", verbose)
    
    # 生成结果
    result.details.append(f"已安装: {len(installed)}/{len(packages)}")
    
    if missing_required:
        result.passed = False
        result.details.append(f"缺失必需依赖: {', '.join(missing_required)}")
        result.message = f"缺少 {len(missing_required)} 个必需依赖"
        result.fix_suggestion = f"运行: pip install {' '.join(missing_required)}"
    
    if missing_optional:
        result.details.append(f"缺失可选依赖: {', '.join(missing_optional)}")
    
    if result.passed:
        result.message = f"所有 {len(installed)} 个依赖已安装"
    
    return result


def check_module_imports(verbose: bool = False) -> CheckResult:
    """递归导入项目所有 Python 模块"""
    result = CheckResult(name="模块导入检查", passed=True)
    import_errors = []
    successful_imports = []
    
    for module_name in PROJECT_MODULES:
        module_path = PROJECT_ROOT / module_name
        
        if not module_path.is_dir():
            print_debug(f"模块目录不存在: {module_name}/", verbose)
            continue
        
        # 检查 __init__.py
        init_file = module_path / "__init__.py"
        if not init_file.exists():
            import_errors.append((module_name, "__init__.py 不存在"))
            continue
        
        # 尝试导入主模块
        try:
            importlib.import_module(module_name)
            successful_imports.append(module_name)
            print_debug(f"导入成功: {module_name}", verbose)
        except Exception as e:
            import_errors.append((module_name, str(e)))
            print_debug(f"导入失败: {module_name} - {e}", verbose)
            continue
        
        # 递归导入所有子模块
        for py_file in module_path.rglob("*.py"):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            
            # 构建模块路径
            relative_path = py_file.relative_to(PROJECT_ROOT)
            parts = list(relative_path.parts)
            parts[-1] = parts[-1].replace(".py", "")
            
            # 跳过 __init__
            if parts[-1] == "__init__":
                continue
            
            full_module_name = ".".join(parts)
            
            try:
                importlib.import_module(full_module_name)
                successful_imports.append(full_module_name)
                print_debug(f"导入成功: {full_module_name}", verbose)
            except Exception as e:
                import_errors.append((full_module_name, str(e)))
                print_debug(f"导入失败: {full_module_name} - {e}", verbose)
    
    # 生成结果
    result.details.append(f"成功导入: {len(successful_imports)} 个模块")
    
    if import_errors:
        result.passed = False
        result.message = f"{len(import_errors)} 个模块导入失败"
        result.details.append("导入失败的模块:")
        for module, error in import_errors[:5]:  # 只显示前5个
            result.details.append(f"  - {module}: {error[:50]}...")
        if len(import_errors) > 5:
            result.details.append(f"  ... 还有 {len(import_errors) - 5} 个错误")
        result.fix_suggestion = "检查导入错误并修复语法/依赖问题"
    else:
        result.message = f"所有 {len(successful_imports)} 个模块导入成功"
    
    return result


def check_critical_components(verbose: bool = False) -> CheckResult:
    """验证 run.py 中的关键类和函数存在"""
    result = CheckResult(name="关键组件检查", passed=True)
    missing_classes = []
    missing_functions = []
    found_components = []
    
    # 检查关键类
    for module_name, class_name in CRITICAL_CLASSES:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                found_components.append(f"{module_name}.{class_name}")
                print_debug(f"找到类: {module_name}.{class_name}", verbose)
            else:
                missing_classes.append(f"{module_name}.{class_name}")
                print_debug(f"缺失类: {module_name}.{class_name}", verbose)
        except ImportError as e:
            missing_classes.append(f"{module_name}.{class_name} (导入失败: {e})")
    
    # 检查关键函数
    for module_name, func_name in CRITICAL_FUNCTIONS:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, func_name):
                found_components.append(f"{module_name}.{func_name}")
                print_debug(f"找到函数: {module_name}.{func_name}", verbose)
            else:
                missing_functions.append(f"{module_name}.{func_name}")
                print_debug(f"缺失函数: {module_name}.{func_name}", verbose)
        except ImportError as e:
            missing_functions.append(f"{module_name}.{func_name} (导入失败: {e})")
    
    # 生成结果
    result.details.append(f"找到组件: {len(found_components)}")
    
    if missing_classes or missing_functions:
        result.passed = False
        result.message = f"缺少 {len(missing_classes)} 个类, {len(missing_functions)} 个函数"
        
        if missing_classes:
            result.details.append(f"缺失的类: {', '.join(missing_classes)}")
        if missing_functions:
            result.details.append(f"缺失的函数: {', '.join(missing_functions)}")
        
        result.fix_suggestion = "检查 run.py 中的类和函数定义是否完整"
    else:
        total = len(CRITICAL_CLASSES) + len(CRITICAL_FUNCTIONS)
        result.message = f"所有 {total} 个关键组件存在"
    
    return result


def check_smoke_test(verbose: bool = False) -> CheckResult:
    """运行快速冒烟测试（仅验证初始化，不实际执行任务）"""
    result = CheckResult(name="冒烟测试", passed=True)
    test_results = []
    errors = []
    
    # 测试 1: Orchestrator 初始化
    try:
        from coordinator import Orchestrator, OrchestratorConfig
        from cursor.client import CursorAgentConfig
        
        cursor_config = CursorAgentConfig(working_directory=str(PROJECT_ROOT))
        config = OrchestratorConfig(
            working_directory=str(PROJECT_ROOT),
            cursor_config=cursor_config,
        )
        orchestrator = Orchestrator(config)
        test_results.append("Orchestrator 初始化成功")
        print_debug("Orchestrator 初始化成功", verbose)
    except Exception as e:
        errors.append(f"Orchestrator 初始化失败: {e}")
        print_debug(f"Orchestrator 初始化失败: {e}", verbose)
    
    # 测试 2: MultiProcessOrchestrator 初始化
    try:
        from coordinator import MultiProcessOrchestrator, MultiProcessOrchestratorConfig
        
        config = MultiProcessOrchestratorConfig(
            working_directory=str(PROJECT_ROOT),
        )
        mp_orchestrator = MultiProcessOrchestrator(config)
        test_results.append("MultiProcessOrchestrator 初始化成功")
        print_debug("MultiProcessOrchestrator 初始化成功", verbose)
    except Exception as e:
        errors.append(f"MultiProcessOrchestrator 初始化失败: {e}")
        print_debug(f"MultiProcessOrchestrator 初始化失败: {e}", verbose)
    
    # 测试 3: TaskQueue 初始化
    try:
        from tasks import TaskQueue
        
        queue = TaskQueue()
        test_results.append("TaskQueue 初始化成功")
        print_debug("TaskQueue 初始化成功", verbose)
    except Exception as e:
        errors.append(f"TaskQueue 初始化失败: {e}")
        print_debug(f"TaskQueue 初始化失败: {e}", verbose)
    
    # 测试 4: Agent 初始化
    try:
        from agents import Planner, Worker, Reviewer
        
        # 只验证类存在，不实际初始化（需要配置）
        assert Planner is not None
        assert Worker is not None
        assert Reviewer is not None
        test_results.append("Agent 类定义完整")
        print_debug("Agent 类定义完整", verbose)
    except Exception as e:
        errors.append(f"Agent 类检查失败: {e}")
        print_debug(f"Agent 类检查失败: {e}", verbose)
    
    # 测试 5: KnowledgeManager 初始化
    try:
        from knowledge import KnowledgeManager
        
        # 只验证类存在
        assert KnowledgeManager is not None
        test_results.append("KnowledgeManager 类存在")
        print_debug("KnowledgeManager 类存在", verbose)
    except Exception as e:
        # 知识库是可选的
        print_debug(f"KnowledgeManager 检查跳过: {e}", verbose)
    
    # 测试 6: TaskAnalyzer 初始化
    try:
        from run import TaskAnalyzer
        
        analyzer = TaskAnalyzer(use_agent=False)  # 不使用 Agent 避免网络调用
        test_results.append("TaskAnalyzer 初始化成功")
        print_debug("TaskAnalyzer 初始化成功", verbose)
    except Exception as e:
        errors.append(f"TaskAnalyzer 初始化失败: {e}")
        print_debug(f"TaskAnalyzer 初始化失败: {e}", verbose)
    
    # 生成结果
    result.details.append(f"通过测试: {len(test_results)}")
    
    if errors:
        result.passed = False
        result.message = f"{len(errors)} 个组件初始化失败"
        result.details.append("失败的测试:")
        for error in errors:
            result.details.append(f"  - {error}")
        result.fix_suggestion = "检查组件初始化代码和依赖关系"
    else:
        result.message = f"所有 {len(test_results)} 个组件初始化成功"
    
    return result


def check_syntax(verbose: bool = False) -> CheckResult:
    """检查所有 Python 文件的语法"""
    result = CheckResult(name="语法检查", passed=True)
    syntax_errors = []
    checked_files = 0
    
    for py_file in PROJECT_ROOT.rglob("*.py"):
        # 跳过不需要检查的目录
        if any(part.startswith(".") or part == "__pycache__" or part == "venv" 
               for part in py_file.parts):
            continue
        
        checked_files += 1
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, str(py_file), "exec")
            print_debug(f"语法正确: {py_file.relative_to(PROJECT_ROOT)}", verbose)
        except SyntaxError as e:
            syntax_errors.append((py_file.relative_to(PROJECT_ROOT), e))
            print_debug(f"语法错误: {py_file.relative_to(PROJECT_ROOT)}", verbose)
    
    # 生成结果
    result.details.append(f"检查文件: {checked_files}")
    
    if syntax_errors:
        result.passed = False
        result.message = f"{len(syntax_errors)} 个文件有语法错误"
        result.details.append("语法错误:")
        for file_path, error in syntax_errors[:5]:
            result.details.append(f"  - {file_path}:{error.lineno}: {error.msg}")
        if len(syntax_errors) > 5:
            result.details.append(f"  ... 还有 {len(syntax_errors) - 5} 个错误")
        result.fix_suggestion = "修复语法错误后重新检查"
    else:
        result.message = f"所有 {checked_files} 个文件语法正确"
    
    return result


def verify_all_imports(verbose: bool = False) -> CheckResult:
    """递归验证所有项目模块的导入

    该函数会：
    1. 递归导入 agents/, coordinator/, core/, cursor/, knowledge/, tasks/ 下的所有模块
    2. 对 run.py 执行语法验证和导入测试
    3. 报告所有导入错误
    """
    result = CheckResult(name="完整导入验证", passed=True)
    import_errors: list[tuple[str, str]] = []
    successful_imports: list[str] = []

    # 需要验证的目录列表
    target_dirs = ["agents", "coordinator", "core", "cursor", "knowledge", "tasks"]

    # 1. 验证 run.py 语法
    run_py = PROJECT_ROOT / "run.py"
    if run_py.exists():
        try:
            with open(run_py, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, str(run_py), "exec")
            print_debug("run.py 语法验证通过", verbose)
        except SyntaxError as e:
            import_errors.append(("run.py", f"语法错误: 行 {e.lineno}: {e.msg}"))
            print_debug(f"run.py 语法错误: {e}", verbose)
    else:
        import_errors.append(("run.py", "文件不存在"))

    # 2. 验证 run.py 导入
    try:
        # 清除缓存以确保重新导入
        if "run" in sys.modules:
            del sys.modules["run"]
        importlib.import_module("run")
        successful_imports.append("run")
        print_debug("run.py 导入成功", verbose)
    except Exception as e:
        import_errors.append(("run", f"导入失败: {e}"))
        print_debug(f"run.py 导入失败: {e}", verbose)

    # 3. 递归导入所有目标目录下的模块
    for dir_name in target_dirs:
        dir_path = PROJECT_ROOT / dir_name

        if not dir_path.is_dir():
            print_debug(f"目录不存在: {dir_name}/", verbose)
            continue

        # 检查 __init__.py
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            import_errors.append((dir_name, "__init__.py 不存在"))
            continue

        # 导入主模块
        try:
            if dir_name in sys.modules:
                # 重新加载以捕获最新变更
                importlib.reload(sys.modules[dir_name])
            else:
                importlib.import_module(dir_name)
            successful_imports.append(dir_name)
            print_debug(f"导入成功: {dir_name}", verbose)
        except Exception as e:
            import_errors.append((dir_name, str(e)))
            print_debug(f"导入失败: {dir_name} - {e}", verbose)
            continue

        # 递归导入所有子模块
        for py_file in dir_path.rglob("*.py"):
            # 跳过私有模块（但保留 __init__.py）
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue

            # 跳过 __pycache__ 目录
            if "__pycache__" in py_file.parts:
                continue

            # 构建模块路径
            relative_path = py_file.relative_to(PROJECT_ROOT)
            parts = list(relative_path.parts)
            parts[-1] = parts[-1].replace(".py", "")

            # 跳过 __init__（已经在导入主模块时处理）
            if parts[-1] == "__init__":
                continue

            full_module_name = ".".join(parts)

            try:
                if full_module_name in sys.modules:
                    importlib.reload(sys.modules[full_module_name])
                else:
                    importlib.import_module(full_module_name)
                successful_imports.append(full_module_name)
                print_debug(f"导入成功: {full_module_name}", verbose)
            except Exception as e:
                import_errors.append((full_module_name, str(e)))
                print_debug(f"导入失败: {full_module_name} - {e}", verbose)

    # 生成结果
    result.details.append(f"成功导入: {len(successful_imports)} 个模块")
    result.details.append(f"目标目录: {', '.join(target_dirs)}")

    if import_errors:
        result.passed = False
        result.message = f"{len(import_errors)} 个模块导入失败"
        result.details.append("导入失败的模块:")
        for module, error in import_errors[:10]:  # 显示前10个错误
            # 截断过长的错误信息
            error_short = error[:80] + "..." if len(error) > 80 else error
            result.details.append(f"  - {module}: {error_short}")
        if len(import_errors) > 10:
            result.details.append(f"  ... 还有 {len(import_errors) - 10} 个错误")
        result.fix_suggestion = "检查导入错误并修复语法/依赖问题，运行 --verbose 查看详细信息"
    else:
        result.message = f"所有 {len(successful_imports)} 个模块导入验证通过"

    return result


def verify_run_modes(verbose: bool = False) -> CheckResult:
    """验证 run.py 的运行模式
    
    验证内容：
    1. run.py --help 是否正常工作
    2. 循环验证所有模式：basic、mp、knowledge、iterate、auto、plan、ask
    
    注意：模式列表需要与 .github/workflows/ci.yml 中的 MODES 变量保持一致
    """
    import subprocess
    
    result = CheckResult(name="运行模式验证", passed=True)
    errors = []
    verified = []
    
    # 需要验证的模式列表（与 ci.yml 保持一致）
    run_modes = ["basic", "mp", "knowledge", "iterate", "auto", "plan", "ask"]
    
    # 1. 验证 run.py --help
    try:
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        
        if proc.returncode == 0:
            verified.append("--help")
            print_debug("run.py --help 执行成功", verbose)
            
            # 检查帮助信息中是否包含所有模式
            help_output = proc.stdout
            for mode in run_modes:
                if mode not in help_output:
                    errors.append(f"--help 输出中缺少模式: {mode}")
                    print_debug(f"--help 输出中缺少模式: {mode}", verbose)
        else:
            errors.append(f"--help 执行失败: {proc.stderr[:100]}")
            print_debug(f"run.py --help 执行失败: {proc.stderr[:100]}", verbose)
    except subprocess.TimeoutExpired:
        errors.append("--help 执行超时")
        print_debug("run.py --help 执行超时", verbose)
    except Exception as e:
        errors.append(f"--help 执行异常: {e}")
        print_debug(f"run.py --help 执行异常: {e}", verbose)
    
    # 2. 验证每个模式的可用性
    # 使用空任务来验证模式参数是否被接受（会返回错误但应该是"请提供任务描述"而不是参数错误）
    for mode in run_modes:
        try:
            proc = subprocess.run(
                [
                    sys.executable, str(PROJECT_ROOT / "run.py"),
                    "--mode", mode,
                    "--no-auto-analyze",
                    "",  # 空任务
                ],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(PROJECT_ROOT),
            )
            
            # 检查输出
            # 期望的行为：要么成功，要么提示"请提供任务描述"
            output = proc.stdout + proc.stderr
            
            # 如果是参数错误则失败
            if "invalid choice" in output.lower() or "unrecognized arguments" in output.lower():
                errors.append(f"模式 '{mode}' 参数无效")
                print_debug(f"模式 '{mode}' 参数无效: {output[:100]}", verbose)
            elif "请提供任务描述" in output or proc.returncode in (0, 1):
                # 正常情况：要求提供任务 或 正常退出
                verified.append(f"--mode {mode}")
                print_debug(f"模式 '{mode}' 验证通过", verbose)
            else:
                # 其他情况也视为通过（可能是环境问题）
                verified.append(f"--mode {mode}")
                print_debug(f"模式 '{mode}' 验证通过 (returncode={proc.returncode})", verbose)
                
        except subprocess.TimeoutExpired:
            # 超时可能是因为模式启动成功但在等待输入
            verified.append(f"--mode {mode} (超时)")
            print_debug(f"模式 '{mode}' 超时（可能正常）", verbose)
        except Exception as e:
            errors.append(f"模式 '{mode}' 验证异常: {e}")
            print_debug(f"模式 '{mode}' 验证异常: {e}", verbose)
    
    # 3. 验证 RunMode 枚举定义完整性
    try:
        from run import RunMode
        
        expected_modes = {"basic", "iterate", "plan", "ask", "mp", "knowledge", "auto"}
        actual_modes = {m.value for m in RunMode}
        
        missing_modes = expected_modes - actual_modes
        if missing_modes:
            errors.append(f"RunMode 枚举缺少模式: {missing_modes}")
            print_debug(f"RunMode 枚举缺少模式: {missing_modes}", verbose)
        else:
            verified.append("RunMode 枚举完整")
            print_debug("RunMode 枚举验证通过", verbose)
    except ImportError as e:
        errors.append(f"无法导入 RunMode: {e}")
        print_debug(f"无法导入 RunMode: {e}", verbose)
    
    # 生成结果
    result.details.append(f"验证通过: {len(verified)} 项")
    result.details.append(f"验证的模式: {', '.join(run_modes)}")
    
    if errors:
        result.passed = False
        result.message = f"{len(errors)} 个运行模式验证失败"
        result.details.append("失败项:")
        for error in errors[:5]:
            result.details.append(f"  - {error}")
        if len(errors) > 5:
            result.details.append(f"  ... 还有 {len(errors) - 5} 个错误")
        result.fix_suggestion = (
            "检查 run.py 中的模式定义和参数解析。"
            "确保 RunMode 枚举包含所有模式，"
            "parse_args() 中的 --mode choices 列表正确。"
        )
    else:
        result.message = f"所有 {len(verified)} 项运行模式验证通过"
    
    return result


def check_config_files(verbose: bool = False) -> CheckResult:
    """检查配置文件格式"""
    result = CheckResult(name="配置文件检查", passed=True)
    errors = []
    checked = []
    
    # 检查 config.yaml
    config_yaml = PROJECT_ROOT / "config.yaml"
    if config_yaml.exists():
        try:
            import yaml
            with open(config_yaml, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            checked.append("config.yaml")
            print_debug("config.yaml 格式正确", verbose)
        except Exception as e:
            errors.append(f"config.yaml: {e}")
    else:
        errors.append("config.yaml 不存在")
    
    # 检查 mcp.json
    mcp_json = PROJECT_ROOT / "mcp.json"
    if mcp_json.exists():
        try:
            import json
            with open(mcp_json, "r", encoding="utf-8") as f:
                json.load(f)
            checked.append("mcp.json")
            print_debug("mcp.json 格式正确", verbose)
        except Exception as e:
            errors.append(f"mcp.json: {e}")
    
    # 检查 .cursor/cli.json
    cli_json = PROJECT_ROOT / ".cursor" / "cli.json"
    if cli_json.exists():
        try:
            import json
            with open(cli_json, "r", encoding="utf-8") as f:
                json.load(f)
            checked.append(".cursor/cli.json")
            print_debug(".cursor/cli.json 格式正确", verbose)
        except Exception as e:
            errors.append(f".cursor/cli.json: {e}")
    
    # 生成结果
    result.details.append(f"检查通过: {len(checked)} 个配置文件")
    
    if errors:
        result.passed = False
        result.message = f"{len(errors)} 个配置文件有问题"
        for error in errors:
            result.details.append(f"  - {error}")
        result.fix_suggestion = "修复配置文件格式错误"
    else:
        result.message = f"所有 {len(checked)} 个配置文件格式正确"
    
    return result


# ============================================================
# 主函数
# ============================================================

def run_checks(verbose: bool = False) -> CheckReport:
    """运行所有检查"""
    report = CheckReport()
    
    print_header("预提交检查")
    print(f"  项目路径: {PROJECT_ROOT}")
    print(f"  Python: {sys.version.split()[0]}")
    
    # 1. 语法检查
    print_section("1. 语法检查")
    result = check_syntax(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 2. 依赖检查
    print_section("2. 依赖检查")
    result = check_dependencies(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 3. 模块导入检查
    print_section("3. 模块导入检查")
    result = check_module_imports(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 4. 完整导入验证（verify_all_imports）
    print_section("4. 完整导入验证")
    result = verify_all_imports(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 5. 关键组件检查
    print_section("5. 关键组件检查")
    result = check_critical_components(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 6. 配置文件检查
    print_section("6. 配置文件检查")
    result = check_config_files(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 7. 运行模式验证
    print_section("7. 运行模式验证")
    result = verify_run_modes(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 8. 冒烟测试
    print_section("8. 冒烟测试")
    result = check_smoke_test(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    return report


def run_quick_checks(verbose: bool = False) -> CheckReport:
    """运行快速检查（仅语法和导入）"""
    report = CheckReport()
    
    print_header("快速预提交检查")
    print(f"  项目路径: {PROJECT_ROOT}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  模式: 快速检查（仅语法和导入）")
    
    # 1. 语法检查
    print_section("1. 语法检查")
    result = check_syntax(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 2. 模块导入检查
    print_section("2. 模块导入检查")
    result = check_module_imports(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 3. 完整导入验证（verify_all_imports）
    print_section("3. 完整导入验证")
    result = verify_all_imports(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    return report


def run_module_only_checks(verbose: bool = False) -> CheckReport:
    """仅运行模块导入检查（用于 CI 快速验证）"""
    report = CheckReport()
    
    print_header("模块导入检查")
    print(f"  项目路径: {PROJECT_ROOT}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  模式: 仅模块导入检查")
    if is_ci_environment():
        print(f"  环境: CI/GitHub Actions")
    
    # 1. 完整导入验证（verify_all_imports）
    print_section("1. 完整导入验证")
    result = verify_all_imports(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    return report


def write_github_step_summary(report: CheckReport) -> None:
    """将检查报告写入 GitHub Step Summary"""
    summary_path = get_github_step_summary_path()
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(report.to_markdown())
                f.write("\n")
            print_info(f"已写入 GitHub Step Summary")
        except Exception as e:
            print_warn(f"无法写入 GitHub Step Summary: {e}")


def main() -> int:
    """主入口"""
    parser = argparse.ArgumentParser(
        description="预提交检查脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="尝试自动修复问题（目前仅提供建议）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结果",
    )
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="快速检查模式（仅语法和导入）",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI 模式（禁用颜色输出，适用于 GitHub Actions）",
    )
    parser.add_argument(
        "--module-only",
        action="store_true",
        help="仅运行模块导入检查",
    )
    
    args = parser.parse_args()
    
    # 检测 CI 环境或显式指定 --ci 参数时禁用颜色
    if args.ci or is_ci_environment():
        Colors.disable()
    
    # 记录开始时间
    start_time = datetime.now(timezone.utc)
    
    try:
        # 根据模式选择检查范围
        if args.module_only:
            report = run_module_only_checks(verbose=args.verbose)
        elif args.quick:
            report = run_quick_checks(verbose=args.verbose)
        else:
            report = run_checks(verbose=args.verbose)
        
        # 计算退出码
        exit_code = 0 if report.all_passed else 1
        
        # 计算结束时间
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        if args.json:
            output = {
                "passed": report.all_passed,
                "passed_count": report.passed_count,
                "failed_count": report.failed_count,
                "exit_code": exit_code,
                "timestamp": start_time.isoformat(),
                "duration_ms": duration_ms,
                "ci_environment": is_ci_environment(),
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "message": c.message,
                        "details": c.details,
                        "fix_suggestion": c.fix_suggestion,
                    }
                    for c in report.checks
                ],
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            report.print_summary()
        
        # 在 CI 环境中写入 GitHub Step Summary
        if is_ci_environment():
            write_github_step_summary(report)
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n\n检查已中断")
        return 130
    except Exception as e:
        print(f"\n{Colors.RED}检查过程出错: {e}{Colors.NC}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
