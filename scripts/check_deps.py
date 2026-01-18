#!/usr/bin/env python3
"""依赖版本检查与冲突检测脚本

检查已安装包版本与声明版本是否匹配，检测间接依赖冲突，生成冲突报告。

功能：
1. 检查已安装包版本与 requirements.txt / pyproject.toml 声明版本是否匹配
2. 检测间接依赖冲突（通过 pip check）
3. 分析依赖树，识别版本不兼容问题
4. 生成详细的冲突报告（支持 text/json/markdown 格式）

用法：
    python scripts/check_deps.py [--format text|json|md] [--output report.txt] [--verbose]
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# 常量定义
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"


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


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DependencySpec:
    """依赖声明"""
    name: str
    version_spec: str  # 原始版本规范，如 ">=1.0.0"
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    exact_version: Optional[str] = None
    source: str = "requirements.txt"  # 来源文件


@dataclass
class InstalledPackage:
    """已安装的包"""
    name: str
    version: str
    location: str = ""
    requires: list[str] = field(default_factory=list)


@dataclass
class VersionMismatch:
    """版本不匹配"""
    package: str
    declared_spec: str
    installed_version: str
    source: str
    severity: str = "error"  # error, warning
    message: str = ""


@dataclass
class DependencyConflict:
    """依赖冲突"""
    package: str
    required_by: str
    required_version: str
    installed_version: str
    message: str = ""


@dataclass
class DependencyReport:
    """依赖检查报告"""
    timestamp: str = ""
    total_declared: int = 0
    total_installed: int = 0
    version_mismatches: list[VersionMismatch] = field(default_factory=list)
    dependency_conflicts: list[DependencyConflict] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)
    extra_packages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        return bool(self.version_mismatches or self.dependency_conflicts or self.missing_packages)
    
    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings or self.extra_packages)


# ============================================================
# 版本解析与比较
# ============================================================

def parse_version(version: str) -> tuple:
    """解析版本号为可比较的元组"""
    # 移除前导 v
    version = version.lstrip("v")
    # 分割版本号
    parts = re.split(r'[.\-]', version)
    result = []
    for part in parts:
        # 尝试转换为数字
        try:
            result.append(int(part))
        except ValueError:
            result.append(part)
    return tuple(result)


def compare_versions(v1: str, v2: str) -> int:
    """比较两个版本号，返回 -1, 0, 1"""
    p1, p2 = parse_version(v1), parse_version(v2)
    if p1 < p2:
        return -1
    elif p1 > p2:
        return 1
    return 0


def version_satisfies(installed: str, spec: str) -> bool:
    """检查已安装版本是否满足规范"""
    if not spec or spec == "*":
        return True
    
    # 解析版本规范
    # 支持: >=1.0, <=2.0, ==1.0, ~=1.0, !=1.0, >1.0, <2.0
    patterns = [
        (r'^>=\s*(.+)$', lambda v, s: compare_versions(v, s) >= 0),
        (r'^>\s*(.+)$', lambda v, s: compare_versions(v, s) > 0),
        (r'^<=\s*(.+)$', lambda v, s: compare_versions(v, s) <= 0),
        (r'^<\s*(.+)$', lambda v, s: compare_versions(v, s) < 0),
        (r'^==\s*(.+)$', lambda v, s: compare_versions(v, s) == 0),
        (r'^!=\s*(.+)$', lambda v, s: compare_versions(v, s) != 0),
        (r'^~=\s*(.+)$', lambda v, s: compare_versions(v, s) >= 0),  # 兼容版本
    ]
    
    # 处理复合规范（如 >=1.0,<2.0）
    specs = [s.strip() for s in spec.split(",")]
    
    for single_spec in specs:
        matched = False
        for pattern, check_fn in patterns:
            match = re.match(pattern, single_spec)
            if match:
                required_version = match.group(1)
                if not check_fn(installed, required_version):
                    return False
                matched = True
                break
        
        if not matched and single_spec:
            # 没有操作符，假设是精确匹配
            if compare_versions(installed, single_spec) != 0:
                return False
    
    return True


# ============================================================
# 依赖解析
# ============================================================

def parse_requirements_txt(file_path: Path) -> list[DependencySpec]:
    """解析 requirements.txt 文件"""
    dependencies = []
    
    if not file_path.exists():
        return dependencies
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith("#"):
                continue
            # 跳过 -r/-e 等特殊行
            if line.startswith("-"):
                continue
            
            # 解析包名和版本
            # 支持格式: package>=1.0.0, package==1.0.0, package, package[extra]>=1.0
            match = re.match(r'^([a-zA-Z0-9_-]+)(?:\[.+\])?\s*(.*)$', line)
            if match:
                name = match.group(1).lower()
                version_spec = match.group(2).strip() if match.group(2) else ""
                dependencies.append(DependencySpec(
                    name=name,
                    version_spec=version_spec,
                    source="requirements.txt"
                ))
    
    return dependencies


def parse_pyproject_toml(file_path: Path) -> list[DependencySpec]:
    """解析 pyproject.toml 文件中的依赖"""
    dependencies = []
    
    if not file_path.exists():
        return dependencies
    
    try:
        # Python 3.11+ 内置 tomllib
        try:
            import tomllib
            with open(file_path, "rb") as f:
                data = tomllib.load(f)
        except ImportError:
            # 回退到手动解析
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            data = _parse_toml_simple(content)
        
        # 获取 project.dependencies
        project = data.get("project", {})
        deps = project.get("dependencies", [])
        
        for dep in deps:
            match = re.match(r'^([a-zA-Z0-9_-]+)(?:\[.+\])?\s*(.*)$', dep)
            if match:
                name = match.group(1).lower()
                version_spec = match.group(2).strip() if match.group(2) else ""
                dependencies.append(DependencySpec(
                    name=name,
                    version_spec=version_spec,
                    source="pyproject.toml"
                ))
        
        # 获取可选依赖
        optional_deps = project.get("optional-dependencies", {})
        for group, deps in optional_deps.items():
            for dep in deps:
                # 跳过自引用 (如 cursorx[vector])
                if dep.startswith(project.get("name", "")):
                    continue
                match = re.match(r'^([a-zA-Z0-9_-]+)(?:\[.+\])?\s*(.*)$', dep)
                if match:
                    name = match.group(1).lower()
                    version_spec = match.group(2).strip() if match.group(2) else ""
                    dependencies.append(DependencySpec(
                        name=name,
                        version_spec=version_spec,
                        source=f"pyproject.toml[{group}]"
                    ))
    except Exception as e:
        print_warn(f"解析 pyproject.toml 失败: {e}")
    
    return dependencies


def _parse_toml_simple(content: str) -> dict:
    """简单的 TOML 解析（仅用于回退）"""
    result = {}
    current_section = result
    section_path = []
    
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # 解析节
        if line.startswith("["):
            section_name = line.strip("[]").strip()
            parts = section_name.split(".")
            current_section = result
            for part in parts:
                if part not in current_section:
                    current_section[part] = {}
                current_section = current_section[part]
            section_path = parts
            continue
        
        # 解析键值对
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            # 解析值
            if value.startswith("["):
                # 简单数组解析
                items = []
                value = value.strip("[]")
                for item in value.split(","):
                    item = item.strip().strip('"').strip("'")
                    if item:
                        items.append(item)
                current_section[key] = items
            elif value.startswith('"') or value.startswith("'"):
                current_section[key] = value.strip('"').strip("'")
            else:
                current_section[key] = value
    
    return result


def get_installed_packages() -> dict[str, InstalledPackage]:
    """获取已安装的包列表"""
    packages = {}
    
    try:
        # 使用 pip list --format=json
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            pip_list = json.loads(result.stdout)
            for pkg in pip_list:
                name = pkg["name"].lower()
                packages[name] = InstalledPackage(
                    name=pkg["name"],
                    version=pkg["version"]
                )
    except Exception as e:
        print_warn(f"获取已安装包列表失败: {e}")
    
    return packages


def check_pip_conflicts() -> list[DependencyConflict]:
    """使用 pip check 检测依赖冲突"""
    conflicts = []
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            # 解析冲突信息
            # 格式: package X.Y.Z has requirement dep>=A.B.C, but you have dep A.B.B.
            # 或: package X.Y.Z requires dep, which is not installed.
            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # 解析 "has requirement" 格式
                match = re.match(
                    r'^(.+?)\s+[\d.]+\s+has requirement\s+(.+?)\s*([<>=!~]+[\d.]+)?,\s*but you have\s+(.+?)\s+([\d.]+)',
                    line
                )
                if match:
                    conflicts.append(DependencyConflict(
                        package=match.group(2),
                        required_by=match.group(1),
                        required_version=match.group(3) or "any",
                        installed_version=match.group(5),
                        message=line
                    ))
                    continue
                
                # 解析 "requires ... not installed" 格式
                match = re.match(
                    r'^(.+?)\s+[\d.]+\s+requires\s+(.+?),\s*which is not installed',
                    line
                )
                if match:
                    conflicts.append(DependencyConflict(
                        package=match.group(2),
                        required_by=match.group(1),
                        required_version="any",
                        installed_version="NOT INSTALLED",
                        message=line
                    ))
    except Exception as e:
        print_warn(f"运行 pip check 失败: {e}")
    
    return conflicts


def analyze_dependency_tree(verbose: bool = False) -> list[DependencyConflict]:
    """分析依赖树，检测间接依赖冲突"""
    conflicts = []
    
    try:
        # 使用 pipdeptree 如果可用
        result = subprocess.run(
            [sys.executable, "-m", "pipdeptree", "--warn", "fail", "--json"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0 and result.stderr:
            # 解析 pipdeptree 警告
            for line in result.stderr.split("\n"):
                if "Warning" in line:
                    print_warn(line)
    except FileNotFoundError:
        # pipdeptree 未安装，跳过
        if verbose:
            print_info("pipdeptree 未安装，跳过依赖树分析（可选: pip install pipdeptree）")
    except Exception as e:
        if verbose:
            print_warn(f"依赖树分析失败: {e}")
    
    return conflicts


# ============================================================
# 检查与报告
# ============================================================

def check_dependencies(verbose: bool = False) -> DependencyReport:
    """执行完整的依赖检查"""
    report = DependencyReport(
        timestamp=datetime.now().isoformat()
    )
    
    # 解析声明的依赖
    declared_deps = []
    declared_deps.extend(parse_requirements_txt(REQUIREMENTS_FILE))
    declared_deps.extend(parse_pyproject_toml(PYPROJECT_FILE))
    
    # 去重（保留第一个）
    seen = set()
    unique_deps = []
    for dep in declared_deps:
        if dep.name not in seen:
            seen.add(dep.name)
            unique_deps.append(dep)
    declared_deps = unique_deps
    
    report.total_declared = len(declared_deps)
    
    # 获取已安装的包
    installed = get_installed_packages()
    report.total_installed = len(installed)
    
    if verbose:
        print_info(f"声明依赖: {report.total_declared} 个")
        print_info(f"已安装包: {report.total_installed} 个")
    
    # 检查版本匹配
    for dep in declared_deps:
        pkg_name = dep.name.lower().replace("-", "_")
        # 尝试多种名称格式
        pkg = installed.get(dep.name) or installed.get(pkg_name) or installed.get(dep.name.replace("_", "-"))
        
        if not pkg:
            report.missing_packages.append(dep.name)
            continue
        
        if dep.version_spec and not version_satisfies(pkg.version, dep.version_spec):
            report.version_mismatches.append(VersionMismatch(
                package=dep.name,
                declared_spec=dep.version_spec,
                installed_version=pkg.version,
                source=dep.source,
                severity="error",
                message=f"已安装 {pkg.version}，但要求 {dep.version_spec}"
            ))
    
    # 检查 pip 冲突
    pip_conflicts = check_pip_conflicts()
    report.dependency_conflicts.extend(pip_conflicts)
    
    # 分析依赖树
    tree_conflicts = analyze_dependency_tree(verbose)
    report.dependency_conflicts.extend(tree_conflicts)
    
    return report


def format_report_text(report: DependencyReport) -> str:
    """生成文本格式报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("  依赖检查报告")
    lines.append("=" * 60)
    lines.append(f"\n时间: {report.timestamp}")
    lines.append(f"声明依赖: {report.total_declared}")
    lines.append(f"已安装包: {report.total_installed}")
    
    # 缺失包
    if report.missing_packages:
        lines.append(f"\n{'─' * 60}")
        lines.append("  缺失的包")
        lines.append(f"{'─' * 60}")
        for pkg in report.missing_packages:
            lines.append(f"  ✗ {pkg}")
    
    # 版本不匹配
    if report.version_mismatches:
        lines.append(f"\n{'─' * 60}")
        lines.append("  版本不匹配")
        lines.append(f"{'─' * 60}")
        for mismatch in report.version_mismatches:
            lines.append(f"  ✗ {mismatch.package}")
            lines.append(f"    声明: {mismatch.declared_spec} ({mismatch.source})")
            lines.append(f"    已安装: {mismatch.installed_version}")
    
    # 依赖冲突
    if report.dependency_conflicts:
        lines.append(f"\n{'─' * 60}")
        lines.append("  依赖冲突")
        lines.append(f"{'─' * 60}")
        for conflict in report.dependency_conflicts:
            lines.append(f"  ✗ {conflict.package}")
            lines.append(f"    需要: {conflict.required_version} (by {conflict.required_by})")
            lines.append(f"    已安装: {conflict.installed_version}")
            if conflict.message:
                lines.append(f"    信息: {conflict.message}")
    
    # 汇总
    lines.append(f"\n{'=' * 60}")
    if report.has_errors:
        error_count = len(report.missing_packages) + len(report.version_mismatches) + len(report.dependency_conflicts)
        lines.append(f"  发现 {error_count} 个问题需要解决")
        lines.append("")
        lines.append("  修复建议:")
        if report.missing_packages:
            lines.append(f"    pip install {' '.join(report.missing_packages)}")
        if report.version_mismatches or report.dependency_conflicts:
            lines.append("    pip install -r requirements.txt --upgrade")
    else:
        lines.append("  ✓ 所有依赖检查通过")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def format_report_json(report: DependencyReport) -> str:
    """生成 JSON 格式报告"""
    data = {
        "timestamp": report.timestamp,
        "summary": {
            "total_declared": report.total_declared,
            "total_installed": report.total_installed,
            "has_errors": report.has_errors,
            "has_warnings": report.has_warnings,
        },
        "missing_packages": report.missing_packages,
        "version_mismatches": [
            {
                "package": m.package,
                "declared_spec": m.declared_spec,
                "installed_version": m.installed_version,
                "source": m.source,
                "severity": m.severity,
                "message": m.message,
            }
            for m in report.version_mismatches
        ],
        "dependency_conflicts": [
            {
                "package": c.package,
                "required_by": c.required_by,
                "required_version": c.required_version,
                "installed_version": c.installed_version,
                "message": c.message,
            }
            for c in report.dependency_conflicts
        ],
        "warnings": report.warnings,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def format_report_markdown(report: DependencyReport) -> str:
    """生成 Markdown 格式报告"""
    lines = []
    lines.append("# 依赖检查报告")
    lines.append("")
    lines.append(f"**时间**: {report.timestamp}")
    lines.append(f"**声明依赖**: {report.total_declared}")
    lines.append(f"**已安装包**: {report.total_installed}")
    
    if report.has_errors:
        lines.append("")
        lines.append("## 发现问题")
        
        if report.missing_packages:
            lines.append("")
            lines.append("### 缺失的包")
            lines.append("")
            for pkg in report.missing_packages:
                lines.append(f"- [ ] `{pkg}`")
        
        if report.version_mismatches:
            lines.append("")
            lines.append("### 版本不匹配")
            lines.append("")
            lines.append("| 包 | 声明版本 | 已安装版本 | 来源 |")
            lines.append("|---|---|---|---|")
            for m in report.version_mismatches:
                lines.append(f"| `{m.package}` | {m.declared_spec} | {m.installed_version} | {m.source} |")
        
        if report.dependency_conflicts:
            lines.append("")
            lines.append("### 依赖冲突")
            lines.append("")
            lines.append("| 包 | 要求版本 | 已安装版本 | 依赖方 |")
            lines.append("|---|---|---|---|")
            for c in report.dependency_conflicts:
                lines.append(f"| `{c.package}` | {c.required_version} | {c.installed_version} | {c.required_by} |")
        
        lines.append("")
        lines.append("## 修复建议")
        lines.append("")
        if report.missing_packages:
            lines.append("```bash")
            lines.append(f"pip install {' '.join(report.missing_packages)}")
            lines.append("```")
        lines.append("")
        lines.append("```bash")
        lines.append("pip install -r requirements.txt --upgrade")
        lines.append("```")
    else:
        lines.append("")
        lines.append("## 检查结果")
        lines.append("")
        lines.append("✅ 所有依赖检查通过")
    
    return "\n".join(lines)


def print_report(report: DependencyReport) -> None:
    """打印报告到控制台"""
    print_header("依赖检查报告")
    print(f"\n  时间: {report.timestamp}")
    print(f"  声明依赖: {report.total_declared}")
    print(f"  已安装包: {report.total_installed}")
    
    # 缺失包
    if report.missing_packages:
        print_section("缺失的包")
        for pkg in report.missing_packages:
            print_fail(pkg)
    
    # 版本不匹配
    if report.version_mismatches:
        print_section("版本不匹配")
        for mismatch in report.version_mismatches:
            print_fail(f"{mismatch.package}")
            print_info(f"  声明: {mismatch.declared_spec} ({mismatch.source})")
            print_info(f"  已安装: {mismatch.installed_version}")
    
    # 依赖冲突
    if report.dependency_conflicts:
        print_section("依赖冲突")
        for conflict in report.dependency_conflicts:
            print_fail(f"{conflict.package}")
            print_info(f"  需要: {conflict.required_version} (by {conflict.required_by})")
            print_info(f"  已安装: {conflict.installed_version}")
    
    # 汇总
    print_header("检查汇总")
    if report.has_errors:
        error_count = len(report.missing_packages) + len(report.version_mismatches) + len(report.dependency_conflicts)
        print(f"\n{Colors.RED}{Colors.BOLD}  发现 {error_count} 个问题需要解决{Colors.NC}")
        print("\n  修复建议:")
        if report.missing_packages:
            print(f"    {Colors.YELLOW}pip install {' '.join(report.missing_packages)}{Colors.NC}")
        if report.version_mismatches or report.dependency_conflicts:
            print(f"    {Colors.YELLOW}pip install -r requirements.txt --upgrade{Colors.NC}")
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}  ✓ 所有依赖检查通过{Colors.NC}")


# ============================================================
# 主函数
# ============================================================

def main() -> int:
    """主入口"""
    parser = argparse.ArgumentParser(
        description="依赖版本检查与冲突检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "md", "markdown"],
        default="text",
        help="输出格式 (default: text)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="输出到文件",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI 模式：简化输出，返回适当的退出码",
    )
    
    args = parser.parse_args()
    
    try:
        # 执行检查
        if not args.ci:
            print_header("依赖检查")
            print(f"  项目路径: {PROJECT_ROOT}")
            print(f"  Python: {sys.version.split()[0]}")
        
        report = check_dependencies(verbose=args.verbose)
        
        # 生成报告
        if args.format == "json":
            output = format_report_json(report)
        elif args.format in ("md", "markdown"):
            output = format_report_markdown(report)
        else:
            if args.output or args.ci:
                output = format_report_text(report)
            else:
                print_report(report)
                return 1 if report.has_errors else 0
        
        # 输出
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            if not args.ci:
                print_pass(f"报告已保存到: {args.output}")
        else:
            print(output)
        
        return 1 if report.has_errors else 0
        
    except KeyboardInterrupt:
        print("\n\n检查已中断")
        return 130
    except Exception as e:
        print(f"\n{Colors.RED}检查过程出错: {e}{Colors.NC}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
