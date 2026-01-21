"""目录驱动工程创建/扩展模块

提供目录状态探测、语言推断、脚手架生成、参考子工程发现等功能。

用于支持 `--directory` 参数的工程初始化逻辑：
- 已有工程：以该目录为工程根进行扩展/更新
- 空目录/仅文档目录：先创建最小可工作的工程骨架，再进入迭代流程
- 根目录含参考子工程：自动发现候选参考工程子目录，并将其信息写入最终 goal
"""
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger


# ============================================================
# 常量定义
# ============================================================

# 工程标记文件映射（文件名 -> 语言/类型）
PROJECT_MARKERS = {
    # Python
    "pyproject.toml": "python",
    "setup.py": "python",
    "requirements.txt": "python",
    "Pipfile": "python",
    "poetry.lock": "python",
    # Node.js / JavaScript / TypeScript
    "package.json": "node",
    "package-lock.json": "node",
    "yarn.lock": "node",
    "pnpm-lock.yaml": "node",
    "tsconfig.json": "typescript",
    # Go
    "go.mod": "go",
    "go.sum": "go",
    # Java
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "java",
    "settings.gradle": "java",
    "settings.gradle.kts": "java",
    # Rust
    "Cargo.toml": "rust",
    "Cargo.lock": "rust",
    # C/C++
    "CMakeLists.txt": "cpp",
    "Makefile": "c",
    # Ruby
    "Gemfile": "ruby",
    "Gemfile.lock": "ruby",
    # PHP
    "composer.json": "php",
    "composer.lock": "php",
    # .NET / C#
    "*.csproj": "csharp",
    "*.sln": "csharp",
    # Scala
    "build.sbt": "scala",
    # Kotlin
    "build.gradle.kts": "kotlin",  # 与 java 共用，但 .kt 文件可区分
}

# 源码文件扩展名映射（扩展名 -> 语言）
SOURCE_EXTENSIONS = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript / TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # Go
    ".go": "go",
    # Java
    ".java": "java",
    # Rust
    ".rs": "rust",
    # C/C++
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    # Ruby
    ".rb": "ruby",
    # PHP
    ".php": "php",
    # C#
    ".cs": "csharp",
    # Scala
    ".scala": "scala",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
}

# 语言关键词映射（用于从任务文本推断语言）
# 支持中文/英文关键词
LANGUAGE_KEYWORDS = {
    "python": [
        "python", "py", "django", "flask", "fastapi", "pytest",
        "用python", "python写", "python项目", "python工程",
        "用 python", "python 写", "python 项目", "python 工程",
    ],
    "node": [
        "node", "nodejs", "node.js", "npm", "yarn", "pnpm",
        "用node", "node写", "nodejs项目", "node项目",
        "用 node", "node 写", "nodejs 项目", "node 项目",
    ],
    "typescript": [
        "typescript", "ts", "tsx", "tsc",
        "用typescript", "typescript写", "ts项目", "typescript项目",
        "用 typescript", "typescript 写", "ts 项目", "typescript 项目",
    ],
    "javascript": [
        "javascript", "js", "jsx", "es6", "es2015",
        "用javascript", "javascript写", "js项目",
        "用 javascript", "javascript 写", "js 项目",
    ],
    "go": [
        "go", "golang", "go语言",
        "用go", "go写", "golang项目", "go项目",
        "用 go", "go 写", "golang 项目", "go 项目",
    ],
    "java": [
        "java", "jdk", "maven", "gradle", "spring", "springboot",
        "用java", "java写", "java项目", "java工程",
        "用 java", "java 写", "java 项目", "java 工程",
    ],
    "rust": [
        "rust", "rs", "cargo", "rustc",
        "用rust", "rust写", "rust项目", "rust工程",
        "用 rust", "rust 写", "rust 项目", "rust 工程",
    ],
    "cpp": [
        "c++", "cpp", "cmake",
        "用c++", "c++写", "c++项目", "cpp项目",
        "用 c++", "c++ 写", "c++ 项目", "cpp 项目",
    ],
    "c": [
        "c语言", "clang", "gcc",
        "用c写", "c项目", "c工程",
        "用 c 写", "c 项目", "c 工程",
    ],
}

# 语言推断优先级（更具体的关键词优先）
# TypeScript 优先于 JavaScript（ts 是 js 的超集）
LANGUAGE_PRIORITY_ORDER = [
    "typescript", "python", "go", "rust", "java",
    "cpp", "c", "node", "javascript",
]

# 仅文档类文件扩展名（不算作源码）
DOC_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".txt", ".adoc",
    ".pdf", ".doc", ".docx", ".odt",
}

# 配置/元数据类文件名（不算作源码）
CONFIG_FILES = {
    ".gitignore", ".gitattributes", ".editorconfig",
    "LICENSE", "LICENSE.md", "LICENSE.txt",
    "README", "README.md", "README.txt", "README.rst",
    "CHANGELOG", "CHANGELOG.md", "CONTRIBUTING.md",
    ".env", ".env.example", ".env.local",
}

# 扫描时忽略的目录
IGNORE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", ".cursor",
    "target", "build", "dist", "out",
    ".tox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "vendor", "packages", ".bundle",
}


# ============================================================
# 数据结构
# ============================================================

class ProjectState(str, Enum):
    """目录状态枚举"""
    EXISTING_PROJECT = "existing_project"  # 已有工程（存在工程标记文件）
    EMPTY_OR_DOCS_ONLY = "empty_or_docs_only"  # 空目录或仅文档目录
    HAS_SOURCE_FILES = "has_source_files"  # 有源码但无工程标记文件


@dataclass
class ProjectInfo:
    """工程信息"""
    state: ProjectState
    path: Path
    detected_language: Optional[str] = None
    marker_files: list[str] = field(default_factory=list)
    source_files_count: int = 0
    is_newly_initialized: bool = False


@dataclass
class ReferenceProject:
    """参考子工程信息"""
    path: Path
    relative_path: str
    detected_language: Optional[str] = None
    marker_files: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ScaffoldResult:
    """脚手架生成结果"""
    success: bool
    language: str
    created_files: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TaskAnalysis:
    """任务解析结果（由 Agent 推断）"""
    language: Optional[str] = None
    project_name: Optional[str] = None
    framework: Optional[str] = None
    params: dict[str, str] = field(default_factory=dict)
    raw_output: str = ""


# ============================================================
# 目录状态探测
# ============================================================

def inspect_project_state(
    target_dir: Path,
    max_depth: int = 3,
) -> ProjectInfo:
    """检测目录状态

    判断目录是否为已有工程、空目录/仅文档目录、或有源码但无工程标记。

    Args:
        target_dir: 目标目录路径
        max_depth: 最大扫描深度

    Returns:
        ProjectInfo 包含目录状态和相关信息
    """
    target_dir = Path(target_dir).resolve()

    if not target_dir.exists():
        logger.warning(f"目录不存在: {target_dir}")
        return ProjectInfo(
            state=ProjectState.EMPTY_OR_DOCS_ONLY,
            path=target_dir,
        )

    if not target_dir.is_dir():
        logger.warning(f"不是目录: {target_dir}")
        return ProjectInfo(
            state=ProjectState.EMPTY_OR_DOCS_ONLY,
            path=target_dir,
        )

    # 检测工程标记文件
    marker_files: list[str] = []
    detected_language: Optional[str] = None

    for marker, lang in PROJECT_MARKERS.items():
        if marker.startswith("*"):
            # 通配符模式（如 *.csproj）
            pattern = marker[1:]  # 去掉 *
            matches = list(target_dir.glob(f"*{pattern}"))
            if matches:
                marker_files.extend([m.name for m in matches])
                if not detected_language:
                    detected_language = lang
        else:
            marker_path = target_dir / marker
            if marker_path.exists():
                marker_files.append(marker)
                if not detected_language:
                    detected_language = lang

    if marker_files:
        logger.debug(f"检测到工程标记文件: {marker_files}, 语言: {detected_language}")
        return ProjectInfo(
            state=ProjectState.EXISTING_PROJECT,
            path=target_dir,
            detected_language=detected_language,
            marker_files=marker_files,
        )

    # 没有工程标记，检查是否有源码文件
    source_files_count = 0
    source_language: Optional[str] = None

    for item in _iter_files(target_dir, max_depth=max_depth):
        ext = item.suffix.lower()
        if ext in SOURCE_EXTENSIONS:
            source_files_count += 1
            if not source_language:
                source_language = SOURCE_EXTENSIONS[ext]

    if source_files_count > 0:
        logger.debug(f"检测到 {source_files_count} 个源码文件，语言: {source_language}")
        return ProjectInfo(
            state=ProjectState.HAS_SOURCE_FILES,
            path=target_dir,
            detected_language=source_language,
            source_files_count=source_files_count,
        )

    # 空目录或仅文档目录
    logger.debug(f"空目录或仅文档目录: {target_dir}")
    return ProjectInfo(
        state=ProjectState.EMPTY_OR_DOCS_ONLY,
        path=target_dir,
    )


def _iter_files(
    root: Path,
    max_depth: int = 3,
    current_depth: int = 0,
) -> list[Path]:
    """迭代目录下的文件（忽略特定目录）

    Args:
        root: 根目录
        max_depth: 最大深度
        current_depth: 当前深度

    Returns:
        文件路径列表
    """
    if current_depth > max_depth:
        return []

    files: list[Path] = []

    try:
        for item in root.iterdir():
            if item.is_file():
                # 跳过配置/元数据文件
                if item.name in CONFIG_FILES:
                    continue
                # 跳过纯文档文件
                if item.suffix.lower() in DOC_EXTENSIONS:
                    continue
                files.append(item)
            elif item.is_dir():
                # 跳过忽略目录
                if item.name in IGNORE_DIRS:
                    continue
                # 递归
                files.extend(_iter_files(item, max_depth, current_depth + 1))
    except PermissionError:
        logger.warning(f"无权限访问目录: {root}")

    return files


# ============================================================
# 语言推断
# ============================================================

def infer_language(task_text: str) -> Optional[str]:
    """从任务文本推断语言

    默认使用大模型推断，失败时回退到关键词规则。

    Args:
        task_text: 任务描述文本

    Returns:
        推断出的语言标识符，推断失败返回 None
    """
    if not task_text:
        return None

    infer_mode = os.getenv("LANGUAGE_INFER_MODE", "llm").lower()
    if infer_mode != "heuristic":
        llm_result = _infer_language_with_llm(task_text)
        if llm_result:
            return llm_result
        if infer_mode == "llm-only":
            return None

    heuristic_result = _infer_language_with_keywords(task_text)
    if heuristic_result:
        return heuristic_result

    logger.debug(f"无法从任务文本推断语言: {task_text[:100]}...")
    return None


def _infer_language_with_keywords(task_text: str) -> Optional[str]:
    """使用关键词规则推断语言（回退路径）"""
    task_lower = task_text.lower()

    for lang in LANGUAGE_PRIORITY_ORDER:
        keywords = LANGUAGE_KEYWORDS.get(lang, [])
        for keyword in keywords:
            # 仅对纯单词关键词使用 \b，避免误匹配
            # 含中文/空格/符号的关键词使用子串匹配，以兼容中文连续文本与符号语言
            use_plain_match = (
                re.search(r"[^\x00-\x7F]", keyword) is not None
                or re.search(r"[^\w]", keyword) is not None
            )

            if use_plain_match:
                if keyword in task_lower:
                    logger.debug(f"从任务文本推断语言(关键词): {lang} (关键词: {keyword})")
                    return lang
            else:
                # 使用单词边界匹配，避免误匹配
                # 例如 "python" 不应匹配 "pythonic" 中的子串
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, task_lower):
                    logger.debug(f"从任务文本推断语言(关键词): {lang} (关键词: {keyword})")
                    return lang

    return None


def _infer_language_with_llm(task_text: str) -> Optional[str]:
    """使用大模型推断语言

    通过 agent CLI 调用规划模式，要求只输出语言标签。
    """
    prompt = _build_language_infer_prompt(task_text)
    model = os.getenv("LANGUAGE_INFER_MODEL")
    cmd = ["agent", "-p", prompt, "--mode", "plan", "--output-format", "text"]
    if model:
        cmd.extend(["--model", model])

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.warning("未找到 agent CLI，无法使用大模型推断语言")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("大模型推断语言超时")
        return None
    except Exception as exc:
        logger.warning(f"大模型推断语言失败: {exc}")
        return None

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        logger.warning(f"大模型推断语言失败: {stderr[:200]}")
        return None

    output = (result.stdout or "").strip().lower()
    return _parse_language_from_output(output)


def _parse_language_from_output(output: str) -> Optional[str]:
    """从大模型输出中解析语言标签"""
    if not output:
        return None

    if "unknown" in output:
        return None

    for lang in LANGUAGE_PRIORITY_ORDER:
        if lang in output:
            return lang

    return None


def _build_language_infer_prompt(task_text: str) -> str:
    """构建语言推断提示词"""
    supported = ", ".join(get_supported_languages())
    return (
        "你是语言分类器。根据任务描述推断最可能的工程语言。\n"
        f"只返回以下标签之一：{supported}。\n"
        "如果无法判断，返回 unknown。\n"
        f"任务描述：{task_text}"
    )


def _analyze_task_with_agent(task_text: str) -> Optional[TaskAnalysis]:
    """使用 Agent 解析任务描述为结构化参数"""
    if not task_text:
        return None

    analyze_mode = os.getenv("TASK_ANALYSIS_MODE", "llm").lower()
    if analyze_mode == "heuristic":
        return None

    prompt = _build_task_analysis_prompt(task_text)
    model = os.getenv("TASK_ANALYSIS_MODEL")
    cmd = ["agent", "-p", prompt, "--mode", "plan", "--output-format", "text"]
    if model:
        cmd.extend(["--model", model])

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.warning("未找到 agent CLI，无法解析任务描述")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("任务解析超时")
        return None
    except Exception as exc:
        logger.warning(f"任务解析失败: {exc}")
        return None

    output = (result.stdout or "").strip()
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        logger.warning(f"任务解析失败: {stderr[:200]}")
        return None

    analysis = _parse_task_analysis_output(output)
    if analysis:
        return analysis

    return None


def _build_task_analysis_prompt(task_text: str) -> str:
    """构建任务解析提示词"""
    supported = ", ".join(get_supported_languages())
    return (
        "你是任务解析器。请将用户任务描述转换为结构化参数。\n"
        "只输出 JSON，不要添加额外文字。\n"
        "字段要求:\n"
        "- language: 语言标签之一（" + supported + "）\n"
        "- project_name: 项目名称（可选）\n"
        "- framework: 框架或技术栈（可选）\n"
        "- params: 其他参数（对象，可选）\n"
        "如果无法判断 language，填 null。\n"
        f"任务描述：{task_text}"
    )


def _parse_task_analysis_output(output: str) -> Optional[TaskAnalysis]:
    """解析任务分析输出为 TaskAnalysis"""
    if not output:
        return None

    payload = output
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    language = data.get("language")
    if isinstance(language, str):
        language = language.strip().lower()
        if language == "unknown":
            language = None
    else:
        language = None

    project_name = data.get("project_name")
    if isinstance(project_name, str):
        project_name = project_name.strip() or None
    else:
        project_name = None

    framework = data.get("framework")
    if isinstance(framework, str):
        framework = framework.strip() or None
    else:
        framework = None

    params = data.get("params")
    if not isinstance(params, dict):
        params = {}

    return TaskAnalysis(
        language=language,
        project_name=project_name,
        framework=framework,
        params={str(k): str(v) for k, v in params.items()},
        raw_output=output,
    )


def get_supported_languages() -> list[str]:
    """获取支持的语言列表

    Returns:
        支持的语言标识符列表
    """
    return list(LANGUAGE_KEYWORDS.keys())


def get_language_hint() -> str:
    """生成语言提示信息

    Returns:
        可用语言和示例的提示字符串
    """
    hints = [
        "支持的语言及示例写法:",
        "  - python: '用 Python 写一个 CLI 工具'",
        "  - node/javascript: '用 Node.js 创建 REST API'",
        "  - typescript: '用 TypeScript 实现服务端'",
        "  - go: '用 Go 写一个微服务'",
        "  - java: '用 Java 创建 Spring Boot 应用'",
        "  - rust: '用 Rust 实现命令行工具'",
        "  - cpp: '用 C++ 写一个数据处理程序'",
        "",
        "请在任务描述中包含语言关键词，或使用 --language 参数指定。",
    ]
    return "\n".join(hints)


# ============================================================
# 脚手架生成
# ============================================================

def scaffold(
    language: str,
    target_dir: Path,
    project_name: Optional[str] = None,
) -> ScaffoldResult:
    """生成最小工程骨架

    根据语言生成最小可识别/可编译的工程结构。

    Args:
        language: 语言标识符
        target_dir: 目标目录
        project_name: 项目名称（可选，默认从目录名推断）

    Returns:
        ScaffoldResult 包含生成结果
    """
    target_dir = Path(target_dir).resolve()

    # 确保目录存在
    target_dir.mkdir(parents=True, exist_ok=True)

    # 推断项目名
    if not project_name:
        project_name = target_dir.name
        # 规范化项目名（移除非法字符）
        project_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name)
        if not project_name or project_name[0].isdigit():
            project_name = "my_project"

    # 根据语言调用对应的脚手架函数
    scaffold_funcs = {
        "python": _scaffold_python,
        "node": _scaffold_node,
        "javascript": _scaffold_node,  # 与 node 共用
        "typescript": _scaffold_typescript,
        "go": _scaffold_go,
        "java": _scaffold_java,
        "rust": _scaffold_rust,
    }

    func = scaffold_funcs.get(language)
    if not func:
        return ScaffoldResult(
            success=False,
            language=language,
            error=f"不支持的语言: {language}",
        )

    try:
        created_files = func(target_dir, project_name)
        logger.info(f"脚手架生成成功: {language}, 创建 {len(created_files)} 个文件")
        return ScaffoldResult(
            success=True,
            language=language,
            created_files=created_files,
        )
    except Exception as e:
        logger.error(f"脚手架生成失败: {e}")
        return ScaffoldResult(
            success=False,
            language=language,
            error=str(e),
        )


def _scaffold_python(target_dir: Path, project_name: str) -> list[str]:
    """生成 Python 工程骨架

    结构:
    - pyproject.toml
    - src/<pkg>/__init__.py
    - tests/__init__.py
    """
    created = []
    pkg_name = project_name.replace("-", "_").lower()

    # pyproject.toml
    pyproject_content = f'''[project]
name = "{project_name}"
version = "0.1.0"
description = ""
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 120

[tool.pytest.ini_options]
testpaths = ["tests"]
'''
    (target_dir / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")
    created.append("pyproject.toml")

    # src/<pkg>/__init__.py
    src_dir = target_dir / "src" / pkg_name
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "__init__.py").write_text(f'"""Package {pkg_name}"""\n', encoding="utf-8")
    created.append(f"src/{pkg_name}/__init__.py")

    # tests/__init__.py
    tests_dir = target_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    created.append("tests/__init__.py")

    return created


def _scaffold_node(target_dir: Path, project_name: str) -> list[str]:
    """生成 Node.js 工程骨架

    结构:
    - package.json
    - src/index.js
    """
    created = []

    # package.json
    package_content = f'''{{"name": "{project_name}",
  "version": "0.1.0",
  "description": "",
  "main": "src/index.js",
  "scripts": {{
    "start": "node src/index.js",
    "test": "echo \\"Error: no test specified\\" && exit 1"
  }},
  "keywords": [],
  "author": "",
  "license": "MIT"
}}
'''
    (target_dir / "package.json").write_text(package_content, encoding="utf-8")
    created.append("package.json")

    # src/index.js
    src_dir = target_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "index.js").write_text('// Entry point\nconsole.log("Hello, world!");\n', encoding="utf-8")
    created.append("src/index.js")

    return created


def _scaffold_typescript(target_dir: Path, project_name: str) -> list[str]:
    """生成 TypeScript 工程骨架

    结构:
    - package.json
    - tsconfig.json
    - src/index.ts
    """
    created = []

    # package.json
    package_content = f'''{{"name": "{project_name}",
  "version": "0.1.0",
  "description": "",
  "main": "dist/index.js",
  "scripts": {{
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "ts-node src/index.ts",
    "test": "echo \\"Error: no test specified\\" && exit 1"
  }},
  "keywords": [],
  "author": "",
  "license": "MIT",
  "devDependencies": {{
    "typescript": "^5.0.0",
    "ts-node": "^10.9.0",
    "@types/node": "^20.0.0"
  }}
}}
'''
    (target_dir / "package.json").write_text(package_content, encoding="utf-8")
    created.append("package.json")

    # tsconfig.json
    tsconfig_content = '''{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
'''
    (target_dir / "tsconfig.json").write_text(tsconfig_content, encoding="utf-8")
    created.append("tsconfig.json")

    # src/index.ts
    src_dir = target_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "index.ts").write_text('// Entry point\nconsole.log("Hello, TypeScript!");\n', encoding="utf-8")
    created.append("src/index.ts")

    return created


def _scaffold_go(target_dir: Path, project_name: str) -> list[str]:
    """生成 Go 工程骨架

    结构:
    - go.mod
    - cmd/app/main.go
    """
    created = []

    # go.mod
    # 使用 example.com 作为默认 module 路径
    module_path = f"example.com/{project_name}"
    gomod_content = f'''module {module_path}

go 1.21
'''
    (target_dir / "go.mod").write_text(gomod_content, encoding="utf-8")
    created.append("go.mod")

    # cmd/app/main.go
    cmd_dir = target_dir / "cmd" / "app"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    main_content = '''package main

import "fmt"

func main() {
\tfmt.Println("Hello, Go!")
}
'''
    (cmd_dir / "main.go").write_text(main_content, encoding="utf-8")
    created.append("cmd/app/main.go")

    return created


def _scaffold_java(target_dir: Path, project_name: str) -> list[str]:
    """生成 Java 工程骨架 (Maven)

    结构:
    - pom.xml
    - src/main/java/com/example/App.java
    """
    created = []

    # pom.xml
    pom_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>{project_name}</artifactId>
    <version>0.1.0</version>
    <packaging>jar</packaging>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
</project>
'''
    (target_dir / "pom.xml").write_text(pom_content, encoding="utf-8")
    created.append("pom.xml")

    # src/main/java/com/example/App.java
    java_dir = target_dir / "src" / "main" / "java" / "com" / "example"
    java_dir.mkdir(parents=True, exist_ok=True)
    app_content = '''package com.example;

public class App {
    public static void main(String[] args) {
        System.out.println("Hello, Java!");
    }
}
'''
    (java_dir / "App.java").write_text(app_content, encoding="utf-8")
    created.append("src/main/java/com/example/App.java")

    return created


def _scaffold_rust(target_dir: Path, project_name: str) -> list[str]:
    """生成 Rust 工程骨架

    结构:
    - Cargo.toml
    - src/main.rs
    """
    created = []

    # Cargo.toml
    cargo_content = f'''[package]
name = "{project_name}"
version = "0.1.0"
edition = "2021"

[dependencies]
'''
    (target_dir / "Cargo.toml").write_text(cargo_content, encoding="utf-8")
    created.append("Cargo.toml")

    # src/main.rs
    src_dir = target_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    main_content = '''fn main() {
    println!("Hello, Rust!");
}
'''
    (src_dir / "main.rs").write_text(main_content, encoding="utf-8")
    created.append("src/main.rs")

    return created


# ============================================================
# 参考子工程发现
# ============================================================

def detect_reference_projects(
    root_dir: Path,
    max_depth: int = 2,
    max_results: int = 5,
) -> list[ReferenceProject]:
    """发现参考子工程

    在根目录下扫描有限深度，寻找工程标记文件聚集的子目录。

    Args:
        root_dir: 根目录
        max_depth: 最大扫描深度
        max_results: 最大返回结果数

    Returns:
        ReferenceProject 列表
    """
    root_dir = Path(root_dir).resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        return []

    candidates: list[ReferenceProject] = []

    def scan_dir(current_dir: Path, depth: int):
        if depth > max_depth:
            return
        if len(candidates) >= max_results * 2:  # 收集更多以便排序
            return

        try:
            for item in current_dir.iterdir():
                if not item.is_dir():
                    continue

                # 跳过忽略目录
                if item.name in IGNORE_DIRS:
                    continue

                # 跳过根目录本身
                if item == root_dir:
                    continue

                # 检测该子目录是否为工程
                marker_files: list[str] = []
                detected_language: Optional[str] = None

                for marker, lang in PROJECT_MARKERS.items():
                    if marker.startswith("*"):
                        pattern = marker[1:]
                        matches = list(item.glob(f"*{pattern}"))
                        if matches:
                            marker_files.extend([m.name for m in matches])
                            if not detected_language:
                                detected_language = lang
                    else:
                        marker_path = item / marker
                        if marker_path.exists():
                            marker_files.append(marker)
                            if not detected_language:
                                detected_language = lang

                if marker_files:
                    # 找到一个子工程
                    rel_path = item.relative_to(root_dir)
                    candidates.append(ReferenceProject(
                        path=item,
                        relative_path=str(rel_path),
                        detected_language=detected_language,
                        marker_files=marker_files,
                        description=f"{detected_language or 'unknown'} 工程，标记文件: {', '.join(marker_files[:3])}",
                    ))
                else:
                    # 继续递归
                    scan_dir(item, depth + 1)

        except PermissionError:
            logger.warning(f"无权限访问目录: {current_dir}")

    scan_dir(root_dir, 0)

    # 按目录深度排序（更浅的优先），然后按语言和路径排序
    candidates.sort(key=lambda x: (
        x.relative_path.count("/"),
        x.detected_language or "zzz",
        x.relative_path,
    ))

    # 去重（同一语言只保留最浅的）
    seen_languages: set[str] = set()
    unique_candidates: list[ReferenceProject] = []

    for candidate in candidates:
        lang = candidate.detected_language or "unknown"
        if lang not in seen_languages or len(unique_candidates) < max_results:
            unique_candidates.append(candidate)
            seen_languages.add(lang)

        if len(unique_candidates) >= max_results:
            break

    logger.debug(f"发现 {len(unique_candidates)} 个参考子工程")
    return unique_candidates


# ============================================================
# 高级接口
# ============================================================

@dataclass
class WorkspacePreparationResult:
    """工作目录准备结果"""
    project_info: ProjectInfo
    reference_projects: list[ReferenceProject] = field(default_factory=list)
    scaffold_result: Optional[ScaffoldResult] = None
    task_analysis: Optional[TaskAnalysis] = None
    error: Optional[str] = None
    hint: Optional[str] = None


def prepare_workspace(
    target_dir: Path,
    task_text: str,
    explicit_language: Optional[str] = None,
    force_scaffold: bool = False,
    dry_run: bool = False,
) -> WorkspacePreparationResult:
    """准备工作目录

    综合调用目录探测、语言推断、脚手架生成等功能，完成工作目录的初始化准备。

    逻辑：
    1. 探测目录状态
    2. 如果是空目录/仅文档目录：
       a. 推断语言（从 explicit_language 或 task_text）
       b. 如果推断成功，生成脚手架
       c. 如果推断失败，返回提示信息
    3. 无论是否生成脚手架，都发现参考子工程
    4. 返回完整结果

    Args:
        target_dir: 目标目录
        task_text: 任务描述文本（用于语言推断）
        explicit_language: 显式指定的语言（优先级最高）
        force_scaffold: 是否强制生成脚手架（即使目录已有工程）
        dry_run: 是否只读模式（为 True 时不写入脚手架）

    Returns:
        WorkspacePreparationResult 包含准备结果
    """
    target_dir = Path(target_dir).resolve()

    # 1. 探测目录状态
    project_info = inspect_project_state(target_dir)

    result = WorkspacePreparationResult(
        project_info=project_info,
    )

    # 任务解析（优先使用 Agent 推断任务形式和参数）
    task_analysis = _analyze_task_with_agent(task_text)
    if task_analysis:
        result.task_analysis = task_analysis

    # 2. 如果是空目录/仅文档目录，尝试生成脚手架
    needs_scaffold = (
        project_info.state == ProjectState.EMPTY_OR_DOCS_ONLY
        or force_scaffold
    )

    if needs_scaffold:
        # 推断语言
        language = (
            explicit_language
            or (task_analysis.language if task_analysis else None)
            or infer_language(task_text)
        )

        if not language:
            # 推断失败
            result.error = "无法推断项目语言"
            result.hint = get_language_hint()
            logger.warning(f"无法推断语言，目录: {target_dir}, 任务: {task_text[:100]}...")
        elif dry_run:
            # dry-run 模式下不创建脚手架，只给出提示
            result.project_info.detected_language = language
            result.error = "dry-run 模式不会创建脚手架"
            result.hint = (
                f"检测到语言: {language}。"
                "请取消 --dry-run 或先手动初始化工程后再重试。"
            )
            logger.info(f"dry-run 模式跳过脚手架创建: {language}, 目录: {target_dir}")
        else:
            # 生成脚手架
            scaffold_result = scaffold(
                language,
                target_dir,
                project_name=task_analysis.project_name if task_analysis else None,
            )
            result.scaffold_result = scaffold_result

            if scaffold_result.success:
                # 更新 project_info
                result.project_info.state = ProjectState.EXISTING_PROJECT
                result.project_info.detected_language = language
                result.project_info.is_newly_initialized = True
                result.project_info.marker_files = scaffold_result.created_files
                logger.info(f"脚手架生成成功: {language}, 目录: {target_dir}")
            else:
                result.error = scaffold_result.error

    # 3. 发现参考子工程
    result.reference_projects = detect_reference_projects(target_dir)

    return result
