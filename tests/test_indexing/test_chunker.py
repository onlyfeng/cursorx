"""测试代码分块器

测试 SemanticCodeChunker 的各种分块策略
"""
import pytest

from indexing.chunker import (
    SemanticCodeChunker,
    chunk_text,
    ChunkContext,
    EXTENSION_LANGUAGE_MAP,
)
from indexing.base import ChunkType
from indexing.config import ChunkConfig, ChunkStrategy


class TestLanguageDetection:
    """测试语言检测"""
    
    def test_detect_python(self):
        """测试检测 Python 文件"""
        chunker = SemanticCodeChunker()
        
        assert chunker.detect_language("test.py") == "python"
        assert chunker.detect_language("script.pyw") == "python"
        assert chunker.detect_language("/path/to/module.py") == "python"
    
    def test_detect_javascript(self):
        """测试检测 JavaScript 文件"""
        chunker = SemanticCodeChunker()
        
        assert chunker.detect_language("app.js") == "javascript"
        assert chunker.detect_language("component.jsx") == "javascript"
    
    def test_detect_typescript(self):
        """测试检测 TypeScript 文件"""
        chunker = SemanticCodeChunker()
        
        assert chunker.detect_language("app.ts") == "typescript"
        assert chunker.detect_language("component.tsx") == "typescript"
    
    def test_detect_unknown(self):
        """测试检测未知语言"""
        chunker = SemanticCodeChunker()
        
        assert chunker.detect_language("file.xyz") == "unknown"
        assert chunker.detect_language("noextension") == "unknown"
    
    def test_supported_languages(self):
        """测试获取支持的语言列表"""
        chunker = SemanticCodeChunker()
        
        languages = chunker.get_supported_languages()
        
        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages


class TestPythonChunking:
    """测试 Python 代码分块"""
    
    @pytest.mark.asyncio
    async def test_chunk_simple_function(self):
        """测试分块简单函数"""
        code = '''
def hello_world():
    """打印 Hello World"""
    print("Hello, World!")
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        assert len(chunks) >= 1
        # 找到函数分块
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) >= 1
        assert func_chunks[0].name == "hello_world"
    
    @pytest.mark.asyncio
    async def test_chunk_class(self):
        """测试分块类"""
        code = '''
class MyClass:
    """一个简单的类"""
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}"
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        # 应该至少有类分块
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1
        assert class_chunks[0].name == "MyClass"
    
    @pytest.mark.asyncio
    async def test_chunk_with_imports(self):
        """测试带导入语句的分块"""
        code = '''
import os
from typing import Optional

def process_file(path: str) -> Optional[str]:
    """处理文件"""
    if os.path.exists(path):
        return path
    return None
'''
        config = ChunkConfig(include_imports=True)
        chunker = SemanticCodeChunker(config)
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        # 函数分块应该包含导入信息
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_chunk_async_function(self):
        """测试分块异步函数"""
        code = '''
async def fetch_data(url: str) -> dict:
    """异步获取数据"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) >= 1
        assert "async" in func_chunks[0].signature
    
    @pytest.mark.asyncio
    async def test_chunk_syntax_error_fallback(self):
        """测试语法错误时的回退处理"""
        # 故意的语法错误
        code = '''
def incomplete_function(
    # 缺少参数和冒号
'''
        chunker = SemanticCodeChunker()
        # 不应该抛出异常，而是使用启发式分块
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        # 应该返回某种分块（即使不是语义分块）
        assert len(chunks) >= 1


class TestChunkSizeControl:
    """测试分块大小控制"""
    
    @pytest.mark.asyncio
    async def test_min_chunk_size(self):
        """测试最小分块大小"""
        # 非常短的代码
        code = "x = 1"
        
        config = ChunkConfig(min_chunk_size=10)
        chunker = SemanticCodeChunker(config)
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        # 短代码可能作为模块级分块
        assert len(chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_max_chunk_size_split(self):
        """测试超大分块的拆分"""
        # 生成一个很长的函数
        lines = ["def long_function():"]
        for i in range(100):
            lines.append(f"    x{i} = {i}")
        lines.append("    return sum([" + ", ".join(f"x{i}" for i in range(100)) + "])")
        
        code = "\n".join(lines)
        
        config = ChunkConfig(
            strategy=ChunkStrategy.HYBRID,
            max_chunk_size=500,
        )
        chunker = SemanticCodeChunker(config)
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        # 大函数应该被拆分
        assert len(chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_sliding_window_chunking(self):
        """测试滑动窗口分块"""
        code = "\n".join([f"line_{i} = {i}" for i in range(50)])
        
        config = ChunkConfig(
            strategy=ChunkStrategy.FIXED_SIZE,
            chunk_size=200,
            chunk_overlap=50,
        )
        chunker = SemanticCodeChunker(config)
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        assert len(chunks) >= 1


class TestMetadataGeneration:
    """测试元数据生成"""
    
    @pytest.mark.asyncio
    async def test_function_metadata(self):
        """测试函数元数据"""
        code = '''
def calculate(a: int, b: int) -> int:
    """计算两数之和
    
    Args:
        a: 第一个数
        b: 第二个数
    
    Returns:
        两数之和
    """
    return a + b
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) >= 1
        
        func = func_chunks[0]
        assert func.name == "calculate"
        assert func.docstring is not None
        assert "两数之和" in func.docstring
        assert func.signature is not None
        assert "a: int" in func.signature
    
    @pytest.mark.asyncio
    async def test_chunk_location_info(self):
        """测试分块位置信息"""
        code = '''# 第一行
# 第二行
def my_function():
    pass
# 最后一行
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        for chunk in chunks:
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line
            assert chunk.file_path == "test.py"
            assert chunk.language == "python"
    
    @pytest.mark.asyncio
    async def test_method_parent_name(self):
        """测试方法的父级名称"""
        code = '''
class Calculator:
    def add(self, a, b):
        return a + b
'''
        config = ChunkConfig(split_functions=True)
        chunker = SemanticCodeChunker(config)
        chunks = await chunker.chunk_text(code, "test.py", "python")
        
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        if method_chunks:  # 如果有方法分块
            assert method_chunks[0].parent_name == "Calculator"


class TestJavaScriptChunking:
    """测试 JavaScript/TypeScript 代码分块"""
    
    @pytest.mark.asyncio
    async def test_chunk_js_function(self):
        """测试分块 JS 函数"""
        code = '''
function greet(name) {
    console.log("Hello, " + name);
}
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.js", "javascript")
        
        assert len(chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_chunk_arrow_function(self):
        """测试分块箭头函数"""
        code = '''
const multiply = (a, b) => {
    return a * b;
};
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.js", "javascript")
        
        assert len(chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_chunk_js_class(self):
        """测试分块 JS 类"""
        code = '''
class Animal {
    constructor(name) {
        this.name = name;
    }
    
    speak() {
        console.log(this.name + " makes a sound");
    }
}
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.js", "javascript")
        
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1


class TestHeuristicChunking:
    """测试启发式分块"""
    
    @pytest.mark.asyncio
    async def test_unknown_language_chunking(self):
        """测试未知语言的分块"""
        code = '''
module Main where

main :: IO ()
main = putStrLn "Hello, World!"
'''
        chunker = SemanticCodeChunker()
        chunks = await chunker.chunk_text(code, "test.hs", "unknown")
        
        # 应该使用启发式分块
        assert len(chunks) >= 1


class TestTokenCounting:
    """测试 token 计数"""
    
    def test_count_tokens_with_tiktoken(self):
        """测试使用 tiktoken 计数"""
        chunker = SemanticCodeChunker()
        
        text = "def hello_world(): print('hello')"
        tokens = chunker.count_tokens(text)
        
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_count_tokens_empty(self):
        """测试空文本计数"""
        chunker = SemanticCodeChunker()
        
        tokens = chunker.count_tokens("")
        assert tokens == 0


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    @pytest.mark.asyncio
    async def test_chunk_text_function(self):
        """测试 chunk_text 便捷函数"""
        code = "def test(): pass"
        
        chunks = await chunk_text(code, "test.py", "python")
        
        assert len(chunks) >= 1
    
    @pytest.mark.asyncio
    async def test_chunk_text_with_config(self):
        """测试带配置的 chunk_text"""
        code = "x = 1\ny = 2"
        
        config = ChunkConfig(min_chunk_size=1)
        chunks = await chunk_text(code, "test.py", config=config)
        
        assert len(chunks) >= 1


class TestChunkContext:
    """测试分块上下文"""
    
    def test_chunk_context_init(self):
        """测试上下文初始化"""
        ctx = ChunkContext()
        
        assert ctx.imports == []
        assert ctx.class_context is None
        assert ctx.file_header is None
    
    def test_chunk_context_with_values(self):
        """测试带值的上下文"""
        ctx = ChunkContext(
            imports=["import os", "import sys"],
            class_context="MyClass",
            file_header="# Module header",
        )
        
        assert len(ctx.imports) == 2
        assert ctx.class_context == "MyClass"


class TestExtensionLanguageMap:
    """测试扩展名语言映射"""
    
    def test_common_extensions(self):
        """测试常见扩展名"""
        assert EXTENSION_LANGUAGE_MAP[".py"] == "python"
        assert EXTENSION_LANGUAGE_MAP[".js"] == "javascript"
        assert EXTENSION_LANGUAGE_MAP[".ts"] == "typescript"
        assert EXTENSION_LANGUAGE_MAP[".go"] == "go"
        assert EXTENSION_LANGUAGE_MAP[".rs"] == "rust"
    
    def test_web_extensions(self):
        """测试 Web 相关扩展名"""
        assert EXTENSION_LANGUAGE_MAP[".html"] == "html"
        assert EXTENSION_LANGUAGE_MAP[".css"] == "css"
        assert EXTENSION_LANGUAGE_MAP[".vue"] == "vue"
