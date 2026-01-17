"""代码分块器实现

提供基于语义的代码分块功能，支持多种编程语言
"""
import re
import ast
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import tiktoken
import aiofiles

from .base import CodeChunker, CodeChunk, ChunkType
from .config import ChunkConfig, ChunkStrategy


# 文件扩展名到语言的映射
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".m": "objectivec",
    ".mm": "objectivec",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
}


@dataclass
class ChunkContext:
    """分块上下文信息
    
    用于在分块过程中传递上下文信息
    """
    imports: list[str] = field(default_factory=list)      # 导入语句
    class_context: Optional[str] = None                    # 当前类名
    file_header: Optional[str] = None                      # 文件头部（模块文档、license 等）


class SemanticCodeChunker(CodeChunker):
    """语义代码分块器
    
    根据代码结构进行智能分块，支持：
    - AST 解析（Python）
    - 正则表达式匹配（JavaScript/TypeScript）
    - 启发式分块（其他语言）
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """初始化分块器
        
        Args:
            config: 分块配置，为 None 时使用默认配置
        """
        self.config = config or ChunkConfig()
        
        # 初始化 tiktoken 编码器（使用 cl100k_base，GPT-4/ChatGPT 使用的编码）
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 如果加载失败，使用简单的字符估算
            self._tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量
        
        Args:
            text: 要计算的文本
            
        Returns:
            token 数量
        """
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # 简单估算：平均每 4 个字符约 1 个 token
        return len(text) // 4
    
    def detect_language(self, file_path: str) -> str:
        """检测文件的编程语言
        
        Args:
            file_path: 文件路径
            
        Returns:
            编程语言标识符
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        return EXTENSION_LANGUAGE_MAP.get(suffix, "unknown")
    
    def get_supported_languages(self) -> list[str]:
        """获取支持的编程语言列表
        
        Returns:
            编程语言标识符列表
        """
        return list(set(EXTENSION_LANGUAGE_MAP.values()))
    
    async def chunk_file(self, file_path: str) -> list[CodeChunk]:
        """分块单个代码文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            代码分块列表
        """
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        
        language = self.detect_language(file_path)
        return await self.chunk_text(content, file_path, language)
    
    async def chunk_text(
        self,
        text: str,
        file_path: str = "<unknown>",
        language: Optional[str] = None
    ) -> list[CodeChunk]:
        """分块代码文本
        
        Args:
            text: 代码文本
            file_path: 文件路径（用于元数据）
            language: 编程语言（可选，会自动检测）
            
        Returns:
            代码分块列表
        """
        if not text.strip():
            return []
        
        # 检测语言
        if language is None:
            language = self.detect_language(file_path)
        
        # 根据策略选择分块方法
        if self.config.strategy == ChunkStrategy.FIXED_SIZE:
            chunks = self._chunk_by_sliding_window(text, file_path, language)
        elif self.config.strategy == ChunkStrategy.AST_BASED:
            chunks = await self._chunk_by_semantic(text, file_path, language)
        elif self.config.strategy == ChunkStrategy.SEMANTIC:
            chunks = await self._chunk_by_semantic(text, file_path, language)
        elif self.config.strategy == ChunkStrategy.HYBRID:
            # 混合策略：先语义分块，再对过大的块进行滑动窗口分块
            chunks = await self._chunk_by_semantic(text, file_path, language)
            chunks = self._split_large_chunks(chunks, file_path, language)
        else:
            chunks = self._chunk_by_sliding_window(text, file_path, language)
        
        return chunks
    
    async def _chunk_by_semantic(
        self,
        text: str,
        file_path: str,
        language: str
    ) -> list[CodeChunk]:
        """基于语义的分块
        
        根据代码结构（函数、类等）进行分块
        
        Args:
            text: 代码文本
            file_path: 文件路径
            language: 编程语言
            
        Returns:
            代码分块列表
        """
        if language == "python":
            return self._chunk_python(text, file_path)
        elif language in ("javascript", "typescript"):
            return self._chunk_javascript(text, file_path, language)
        else:
            # 其他语言使用启发式分块
            return self._chunk_heuristic(text, file_path, language)
    
    def _chunk_python(self, text: str, file_path: str) -> list[CodeChunk]:
        """Python 代码分块
        
        使用 AST 解析 Python 代码，按函数/类分块
        
        Args:
            text: Python 代码
            file_path: 文件路径
            
        Returns:
            代码分块列表
        """
        chunks: list[CodeChunk] = []
        lines = text.split("\n")
        
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # 语法错误时回退到启发式分块
            return self._chunk_heuristic(text, file_path, "python")
        
        # 提取导入语句
        imports: list[str] = []
        import_lines: list[int] = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = node.lineno - 1
                end_line = getattr(node, "end_lineno", node.lineno) - 1
                import_text = "\n".join(lines[start_line:end_line + 1])
                imports.append(import_text)
                import_lines.extend(range(start_line, end_line + 1))
        
        import_block = "\n".join(imports) if imports else ""
        
        # 处理顶层定义
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # 类定义
                chunk = self._create_python_class_chunk(
                    node, lines, file_path, import_block
                )
                if chunk:
                    chunks.append(chunk)
                    
                    # 处理类方法（如果配置要求按方法分块）
                    if self.config.split_functions:
                        method_chunks = self._extract_python_methods(
                            node, lines, file_path, import_block, node.name
                        )
                        chunks.extend(method_chunks)
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 顶层函数
                chunk = self._create_python_function_chunk(
                    node, lines, file_path, import_block
                )
                if chunk:
                    chunks.append(chunk)
        
        # 如果没有提取到任何块，将整个文件作为一个块
        if not chunks:
            chunks.append(CodeChunk(
                content=text,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                chunk_type=ChunkType.MODULE,
                language="python"
            ))
        
        return chunks
    
    def _create_python_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        file_path: str,
        import_block: str,
        parent_name: Optional[str] = None
    ) -> Optional[CodeChunk]:
        """创建 Python 函数分块
        
        Args:
            node: AST 函数节点
            lines: 代码行列表
            file_path: 文件路径
            import_block: 导入语句块
            parent_name: 父级名称（类名）
            
        Returns:
            代码分块，如果分块太小则返回 None
        """
        start_line = node.lineno - 1
        end_line = getattr(node, "end_lineno", node.lineno) - 1
        
        func_lines = lines[start_line:end_line + 1]
        func_content = "\n".join(func_lines)
        if not func_content.strip():
            return None
        
        # 组装完整内容（包含导入上下文）
        if self.config.include_imports and import_block:
            full_content = f"{import_block}\n\n{func_content}"
        else:
            full_content = func_content
        
        # 提取函数签名
        signature = self._get_python_function_signature(node)
        
        # 提取文档字符串
        docstring = ast.get_docstring(node)
        
        # 确定块类型
        chunk_type = ChunkType.METHOD if parent_name else ChunkType.FUNCTION
        
        return CodeChunk(
            content=full_content,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            chunk_type=chunk_type,
            language="python",
            name=node.name,
            parent_name=parent_name,
            signature=signature,
            docstring=docstring
        )
    
    def _create_python_class_chunk(
        self,
        node: ast.ClassDef,
        lines: list[str],
        file_path: str,
        import_block: str
    ) -> Optional[CodeChunk]:
        """创建 Python 类分块
        
        Args:
            node: AST 类节点
            lines: 代码行列表
            file_path: 文件路径
            import_block: 导入语句块
            
        Returns:
            代码分块
        """
        start_line = node.lineno - 1
        end_line = getattr(node, "end_lineno", node.lineno) - 1
        
        class_lines = lines[start_line:end_line + 1]
        class_content = "\n".join(class_lines)
        
        # 组装完整内容
        if self.config.include_imports and import_block:
            full_content = f"{import_block}\n\n{class_content}"
        else:
            full_content = class_content
        
        # 提取类签名（类定义行）
        signature = lines[start_line].strip()
        
        # 提取文档字符串
        docstring = ast.get_docstring(node)
        
        return CodeChunk(
            content=full_content,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            chunk_type=ChunkType.CLASS,
            language="python",
            name=node.name,
            signature=signature,
            docstring=docstring
        )
    
    def _extract_python_methods(
        self,
        class_node: ast.ClassDef,
        lines: list[str],
        file_path: str,
        import_block: str,
        class_name: str
    ) -> list[CodeChunk]:
        """提取 Python 类中的方法
        
        Args:
            class_node: AST 类节点
            lines: 代码行列表
            file_path: 文件路径
            import_block: 导入语句块
            class_name: 类名
            
        Returns:
            方法分块列表
        """
        chunks: list[CodeChunk] = []
        
        # 构建类头部上下文（类定义 + 类变量）
        class_start = class_node.lineno - 1
        first_method_line = None
        
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_method_line = item.lineno - 1
                break
        
        if first_method_line:
            class_header = "\n".join(lines[class_start:first_method_line])
        else:
            class_header = lines[class_start]
        
        # 提取每个方法
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._create_python_function_chunk(
                    item, lines, file_path, import_block, class_name
                )
                if chunk:
                    # 在方法内容前添加类头部上下文
                    if self.config.include_imports:
                        prefix_parts: list[str] = []
                        if import_block:
                            prefix_parts.append(import_block)
                        if class_header:
                            prefix_parts.append(class_header)
                        prefix = "\n\n".join(prefix_parts).strip()
                        if prefix:
                            chunk.content = f"{prefix}\n\n    # ... (方法实现) ...\n\n{chunk.content.strip()}"
                        else:
                            chunk.content = chunk.content.strip()
                    chunks.append(chunk)
        
        return chunks
    
    def _get_python_function_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str:
        """获取 Python 函数签名
        
        Args:
            node: AST 函数节点
            
        Returns:
            函数签名字符串
        """
        # 构建参数列表
        args = []
        
        # 位置参数
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        # 返回类型
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"
        
        # 异步标记
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        
        return f"{async_prefix}def {node.name}({', '.join(args)}){return_type}"
    
    def _chunk_javascript(
        self,
        text: str,
        file_path: str,
        language: str
    ) -> list[CodeChunk]:
        """JavaScript/TypeScript 代码分块
        
        使用正则表达式识别函数和类定义
        
        Args:
            text: 代码文本
            file_path: 文件路径
            language: 语言（javascript 或 typescript）
            
        Returns:
            代码分块列表
        """
        chunks: list[CodeChunk] = []
        lines = text.split("\n")
        
        # 提取 import/require 语句
        import_pattern = r'^(?:import\s+.*?(?:from\s+[\'"].*?[\'"])?;?|const\s+.*?=\s*require\([\'"].*?[\'"]\);?|export\s+.*?from\s+[\'"].*?[\'"];?)$'
        imports: list[str] = []
        
        for i, line in enumerate(lines):
            if re.match(import_pattern, line.strip()):
                imports.append(line)
        
        import_block = "\n".join(imports) if imports else ""
        
        # 定义模式
        patterns = [
            # 普通函数: function name(...) { 或 async function name(...)
            (r'^(\s*)((?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*\{)',
             ChunkType.FUNCTION),
            # 类定义: class Name { 或 export class Name
            (r'^(\s*)((?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{)',
             ChunkType.CLASS),
            # 箭头函数变量: const name = (...) => 或 export const name = async () =>
            (r'^(\s*)((?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\w+)?\s*=>)',
             ChunkType.FUNCTION),
            # 对象方法简写: name(...) { 或 async name(...)
            (r'^(\s+)((?:async\s+)?(\w+)\s*\([^)]*\)\s*\{)',
             ChunkType.METHOD),
        ]
        
        # 查找所有定义的起始位置
        definitions: list[tuple[int, str, ChunkType, str]] = []  # (行号, 签名, 类型, 名称)
        
        for i, line in enumerate(lines):
            for pattern, chunk_type in patterns:
                match = re.match(pattern, line)
                if match:
                    indent = match.group(1)
                    signature = match.group(2)
                    name = match.group(3)
                    definitions.append((i, signature, chunk_type, name))
                    break
        
        # 为每个定义找到结束位置
        for idx, (start_line, signature, chunk_type, name) in enumerate(definitions):
            # 找到匹配的闭合括号
            end_line = self._find_js_block_end(lines, start_line)
            
            chunk_lines = lines[start_line:end_line + 1]
            chunk_content = "\n".join(chunk_lines)
            if not chunk_content.strip():
                continue
            
            # 组装完整内容
            if self.config.include_imports and import_block:
                full_content = f"{import_block}\n\n{chunk_content}"
            else:
                full_content = chunk_content
            
            chunks.append(CodeChunk(
                content=full_content,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                chunk_type=chunk_type,
                language=language,
                name=name,
                signature=signature.strip()
            ))
        
        # 如果没有找到任何定义，使用启发式分块
        if not chunks:
            return self._chunk_heuristic(text, file_path, language)
        
        return chunks
    
    def _find_js_block_end(self, lines: list[str], start_line: int) -> int:
        """查找 JavaScript 代码块的结束行
        
        通过匹配花括号找到代码块结束位置
        
        Args:
            lines: 代码行列表
            start_line: 起始行号
            
        Returns:
            结束行号
        """
        brace_count = 0
        in_string = False
        string_char = None
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            j = 0
            while j < len(line):
                char = line[j]
                
                # 处理字符串
                if char in ('"', "'", "`") and (j == 0 or line[j-1] != "\\"):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                
                # 处理花括号
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            return i
                
                j += 1
        
        # 如果没找到匹配的闭合括号，返回文件末尾
        return len(lines) - 1
    
    def _chunk_heuristic(
        self,
        text: str,
        file_path: str,
        language: str
    ) -> list[CodeChunk]:
        """启发式分块
        
        基于缩进和空行的通用分块方法，适用于未专门支持的语言
        
        Args:
            text: 代码文本
            file_path: 文件路径
            language: 编程语言
            
        Returns:
            代码分块列表
        """
        chunks: list[CodeChunk] = []
        lines = text.split("\n")
        
        current_chunk_lines: list[str] = []
        current_start_line = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 判断是否是分块边界
            is_boundary = False
            
            # 空行可能是分块边界（如果已经积累了足够的内容）
            if not stripped and current_chunk_lines:
                current_content = "\n".join(current_chunk_lines)
                if len(current_content) >= self.config.min_chunk_size:
                    # 检查下一行是否是顶层定义
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if next_line and not next_line[0].isspace():
                            is_boundary = True
            
            # 顶层定义开始（无缩进的非空行）
            if stripped and i > 0 and (not line or not line[0].isspace()):
                # 常见的定义关键字
                definition_keywords = [
                    "def ", "class ", "async def ",  # Python
                    "function ", "const ", "let ", "var ", "export ",  # JS/TS
                    "func ", "type ", "struct ", "interface ",  # Go
                    "fn ", "impl ", "struct ", "enum ", "trait ",  # Rust
                    "public ", "private ", "protected ", "static ",  # Java/C#
                ]
                for keyword in definition_keywords:
                    if stripped.startswith(keyword):
                        is_boundary = True
                        break
            
            if is_boundary and current_chunk_lines:
                # 保存当前块
                chunk_content = "\n".join(current_chunk_lines)
                if len(chunk_content) >= self.config.min_chunk_size:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=current_start_line + 1,
                        end_line=i,
                        chunk_type=ChunkType.UNKNOWN,
                        language=language
                    ))
                current_chunk_lines = []
                current_start_line = i
            
            current_chunk_lines.append(line)
            
            # 检查当前块是否达到最大大小
            current_content = "\n".join(current_chunk_lines)
            if len(current_content) >= self.config.max_chunk_size:
                # 强制分割
                chunks.append(CodeChunk(
                    content=current_content,
                    file_path=file_path,
                    start_line=current_start_line + 1,
                    end_line=i + 1,
                    chunk_type=ChunkType.UNKNOWN,
                    language=language
                ))
                current_chunk_lines = []
                current_start_line = i + 1
        
        # 保存最后一个块
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            if len(chunk_content) >= self.config.min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_start_line + 1,
                    end_line=len(lines),
                    chunk_type=ChunkType.UNKNOWN,
                    language=language
                ))
        
        # 如果没有块或者只有很少内容，将整个文件作为一个块
        if not chunks:
            chunks.append(CodeChunk(
                content=text,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                chunk_type=ChunkType.MODULE,
                language=language
            ))
        
        return chunks
    
    def _chunk_by_sliding_window(
        self,
        text: str,
        file_path: str,
        language: str
    ) -> list[CodeChunk]:
        """滑动窗口分块
        
        按固定大小进行分块，支持重叠
        
        Args:
            text: 代码文本
            file_path: 文件路径
            language: 编程语言
            
        Returns:
            代码分块列表
        """
        chunks: list[CodeChunk] = []
        lines = text.split("\n")
        
        # 计算每块的行数（基于目标字符数）
        avg_line_length = len(text) / max(len(lines), 1)
        lines_per_chunk = max(1, int(self.config.chunk_size / avg_line_length))
        overlap_lines = max(0, int(self.config.chunk_overlap / avg_line_length))
        
        step = max(1, lines_per_chunk - overlap_lines)
        start = 0
        
        while start < len(lines):
            end = min(start + lines_per_chunk, len(lines))
            chunk_lines = lines[start:end]
            chunk_content = "\n".join(chunk_lines)
            
            # 检查 token 限制
            tokens = self.count_tokens(chunk_content)
            if tokens > self.config.max_chunk_size // 4:  # 假设平均 4 字符/token
                # 缩小块大小
                while tokens > self.config.max_chunk_size // 4 and len(chunk_lines) > 1:
                    chunk_lines = chunk_lines[:-1]
                    chunk_content = "\n".join(chunk_lines)
                    tokens = self.count_tokens(chunk_content)
            
            if len(chunk_content) >= self.config.min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=start + len(chunk_lines),
                    chunk_type=ChunkType.UNKNOWN,
                    language=language,
                    metadata={"is_sliding_window": True}
                ))
            
            start += step
        
        return chunks
    
    def _split_large_chunks(
        self,
        chunks: list[CodeChunk],
        file_path: str,
        language: str
    ) -> list[CodeChunk]:
        """拆分过大的分块
        
        对超过最大大小的块使用滑动窗口进行拆分
        
        Args:
            chunks: 原始分块列表
            file_path: 文件路径
            language: 编程语言
            
        Returns:
            处理后的分块列表
        """
        result: list[CodeChunk] = []
        
        for chunk in chunks:
            if len(chunk.content) > self.config.max_chunk_size:
                # 对过大的块进行滑动窗口分割
                sub_chunks = self._chunk_by_sliding_window(
                    chunk.content, file_path, language
                )
                # 保留原始块的元数据
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_type = chunk.chunk_type
                    sub_chunk.name = chunk.name
                    sub_chunk.parent_name = chunk.parent_name
                    # 调整行号
                    sub_chunk.start_line += chunk.start_line - 1
                    sub_chunk.end_line += chunk.start_line - 1
                result.extend(sub_chunks)
            else:
                result.append(chunk)
        
        return result


# 便捷函数
async def chunk_file(
    file_path: str,
    config: Optional[ChunkConfig] = None
) -> list[CodeChunk]:
    """便捷函数：分块单个文件
    
    Args:
        file_path: 文件路径
        config: 分块配置
        
    Returns:
        代码分块列表
    """
    chunker = SemanticCodeChunker(config)
    return await chunker.chunk_file(file_path)


async def chunk_text(
    text: str,
    file_path: str = "<unknown>",
    language: Optional[str] = None,
    config: Optional[ChunkConfig] = None
) -> list[CodeChunk]:
    """便捷函数：分块代码文本
    
    Args:
        text: 代码文本
        file_path: 文件路径
        language: 编程语言
        config: 分块配置
        
    Returns:
        代码分块列表
    """
    chunker = SemanticCodeChunker(config)
    return await chunker.chunk_text(text, file_path, language)
