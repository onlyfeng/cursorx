#!/usr/bin/env python3
"""预提交检查脚本

在提交代码前执行全面的检查，确保代码质量和依赖完整性。

检查内容：
1. 递归导入项目所有 Python 模块
2. 检查 requirements.txt 中的依赖是否都已安装
3. 尝试导入 run.py 并验证关键类和函数存在
4. 运行快速冒烟测试（不实际执行任务，仅验证初始化）

用法：
    python scripts/pre_commit_check.py [--verbose] [--fix]
"""

import argparse
import importlib
import importlib.util
import os
import pkgutil
import re
import sys
import traceback
from dataclasses import dataclass, field
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
]


# ============================================================
# 颜色输出
# ============================================================

class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    MAGENTA = "\033[0;35m"
    BOLD = "\033[1m"
    NC = "\033[0m"


def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")


def print_section(text: str) -> None:
    print(f"\n{Colors.CYAN}{'─' * 60}{Colors.NC}")
    print(f"{Colors.CYAN}  {text}{Colors.NC}")
    print(f"{Colors.CYAN}{'─' * 60}{Colors.NC}")


def print_pass(text: str) -> None:
    print(f"  {Colors.GREEN}✓{Colors.NC} {text}")


def print_fail(text: str) -> None:
    print(f"  {Colors.RED}✗{Colors.NC} {text}")


def print_warn(text: str) -> None:
    print(f"  {Colors.YELLOW}⚠{Colors.NC} {text}")


def print_info(text: str) -> None:
    print(f"  {Colors.BLUE}ℹ{Colors.NC} {text}")


def print_debug(text: str, verbose: bool = False) -> None:
    if verbose:
        print(f"  {Colors.MAGENTA}•{Colors.NC} {text}")


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
        
        print(f"\n  {Colors.GREEN}✓ 通过:{Colors.NC} {self.passed_count}")
        print(f"  {Colors.RED}✗ 失败:{Colors.NC} {self.failed_count}")
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
    
    # 4. 关键组件检查
    print_section("4. 关键组件检查")
    result = check_critical_components(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 5. 配置文件检查
    print_section("5. 配置文件检查")
    result = check_config_files(verbose)
    report.add(result)
    if result.passed:
        print_pass(result.message)
    else:
        print_fail(result.message)
        for detail in result.details:
            print_info(detail)
    
    # 6. 冒烟测试
    print_section("6. 冒烟测试")
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
    
    return report


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
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            report = run_quick_checks(verbose=args.verbose)
        else:
            report = run_checks(verbose=args.verbose)
        
        if args.json:
            import json
            output = {
                "passed": report.all_passed,
                "passed_count": report.passed_count,
                "failed_count": report.failed_count,
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
        
        return 0 if report.all_passed else 1
        
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
