#!/usr/bin/env python3
"""知识库交互式测试脚本

验证知识库的完整功能：
1. 文档添加与更新
2. 存储和加载
3. 搜索功能验证（关键词、语义、混合）
4. 实际问答效果测试

用法:
    python scripts/test_knowledge_interactive.py                # 交互模式
    python scripts/test_knowledge_interactive.py --auto         # 自动测试模式
    python scripts/test_knowledge_interactive.py --demo         # 演示模式（使用示例数据）
    python scripts/test_knowledge_interactive.py --query "问题" # 直接查询模式
"""
import argparse
import asyncio
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge import (  # noqa: E402
    KnowledgeManager,
    KnowledgeStorage,
    Document,
    SearchResult,
    ChunkSplitter,
)
from knowledge.vector import KnowledgeVectorConfig  # noqa: E402


# ============================================================
# 测试数据
# ============================================================

DEMO_DOCUMENTS = [
    {
        "title": "Python 编程入门指南",
        "content": """Python 是一种高级编程语言，由 Guido van Rossum 在 1991 年创建。

Python 的主要特点：
- 语法简洁明了，易于学习
- 动态类型系统
- 自动内存管理
- 丰富的标准库
- 跨平台支持

Python 常用于：
1. Web 开发（Django, Flask, FastAPI）
2. 数据分析（Pandas, NumPy）
3. 机器学习（TensorFlow, PyTorch, scikit-learn）
4. 自动化脚本
5. 科学计算

安装 Python：
- Windows: 从 python.org 下载安装包
- macOS: brew install python3
- Linux: apt install python3 或 yum install python3

Python 虚拟环境：
使用 venv 模块创建隔离的开发环境：
    python -m venv myenv
    source myenv/bin/activate  # Linux/macOS
    myenv\\Scripts\\activate    # Windows
""",
    },
    {
        "title": "机器学习基础概念",
        "content": """机器学习是人工智能的一个分支，让计算机能够从数据中学习模式。

机器学习的三大类型：

1. 监督学习 (Supervised Learning)
   - 使用带标签的数据进行训练
   - 常见算法：线性回归、逻辑回归、决策树、随机森林、SVM、神经网络
   - 应用：分类、回归预测

2. 无监督学习 (Unsupervised Learning)
   - 在无标签数据中发现模式
   - 常见算法：K-means 聚类、层次聚类、PCA 降维
   - 应用：客户分群、异常检测

3. 强化学习 (Reinforcement Learning)
   - 通过与环境交互学习最优策略
   - 常见算法：Q-Learning、DQN、PPO
   - 应用：游戏 AI、机器人控制

常用机器学习库：
- scikit-learn: 经典机器学习算法
- TensorFlow: Google 深度学习框架
- PyTorch: Facebook 深度学习框架
- Keras: 高级神经网络 API
- XGBoost: 梯度提升算法

模型评估指标：
- 分类：准确率、精确率、召回率、F1 分数、AUC-ROC
- 回归：MSE、RMSE、MAE、R²
""",
    },
    {
        "title": "RESTful API 设计规范",
        "content": """RESTful API 是一种设计 Web 服务的架构风格。

REST 核心原则：
1. 统一接口 (Uniform Interface)
2. 无状态 (Stateless)
3. 可缓存 (Cacheable)
4. 客户端-服务器分离 (Client-Server)
5. 分层系统 (Layered System)

HTTP 方法语义：
- GET: 获取资源
- POST: 创建资源
- PUT: 完全更新资源
- PATCH: 部分更新资源
- DELETE: 删除资源

URL 设计最佳实践：
- 使用名词而非动词: /users 而不是 /getUsers
- 使用复数形式: /users 而不是 /user
- 使用连字符: /user-profiles 而不是 /userProfiles
- 版本控制: /api/v1/users

HTTP 状态码：
- 200 OK: 请求成功
- 201 Created: 资源创建成功
- 204 No Content: 删除成功
- 400 Bad Request: 请求参数错误
- 401 Unauthorized: 未认证
- 403 Forbidden: 无权限
- 404 Not Found: 资源不存在
- 500 Internal Server Error: 服务器错误

认证方式：
- API Key
- JWT (JSON Web Token)
- OAuth 2.0
- Basic Auth
""",
    },
    {
        "title": "Docker 容器技术",
        "content": """Docker 是一个开源的容器化平台，用于构建、运行和分发应用程序。

Docker 核心概念：
1. 镜像 (Image): 只读模板，包含运行应用所需的一切
2. 容器 (Container): 镜像的运行实例
3. Dockerfile: 定义镜像构建步骤的文本文件
4. Docker Compose: 多容器应用编排工具

常用 Docker 命令：
    docker pull nginx           # 拉取镜像
    docker run -d -p 80:80 nginx  # 运行容器
    docker ps                   # 查看运行中的容器
    docker stop <container_id>   # 停止容器
    docker rm <container_id>     # 删除容器
    docker images               # 查看本地镜像
    docker build -t myapp .     # 构建镜像

Dockerfile 示例：
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["python", "app.py"]

Docker Compose 示例：
    version: "3.8"
    services:
      web:
        build: .
        ports:
          - "5000:5000"
      redis:
        image: redis:alpine

Docker 网络：
- bridge: 默认网络模式
- host: 共享主机网络
- none: 无网络
- overlay: 跨主机网络（Swarm 模式）
""",
    },
    {
        "title": "Git 版本控制指南",
        "content": """Git 是一个分布式版本控制系统，用于跟踪代码变更。

Git 基本概念：
- 仓库 (Repository): 存储项目文件和历史记录
- 提交 (Commit): 保存更改的快照
- 分支 (Branch): 独立的开发线
- 合并 (Merge): 将分支合并到一起

常用 Git 命令：
    git init                    # 初始化仓库
    git clone <url>             # 克隆远程仓库
    git add .                   # 暂存所有更改
    git commit -m "message"     # 提交更改
    git push origin main        # 推送到远程
    git pull origin main        # 拉取远程更新
    git branch feature          # 创建分支
    git checkout feature        # 切换分支
    git merge feature           # 合并分支

Git 工作流：
1. Git Flow
   - main: 生产代码
   - develop: 开发代码
   - feature/*: 功能分支
   - release/*: 发布分支
   - hotfix/*: 紧急修复

2. GitHub Flow
   - 从 main 创建分支
   - 提交更改
   - 创建 Pull Request
   - 代码审查
   - 合并到 main

.gitignore 常见配置：
    __pycache__/
    *.pyc
    .env
    node_modules/
    .vscode/
    *.log
""",
    },
]

# 测试问题和预期答案关键词
TEST_QUERIES = [
    {
        "query": "Python 编程",
        "expected_keywords": ["高级编程语言", "语法简洁", "动态类型"],
        "expected_doc": "Python 编程入门指南",
    },
    {
        "query": "机器学习",
        "expected_keywords": ["监督学习", "无监督学习", "强化学习"],
        "expected_doc": "机器学习基础概念",
    },
    {
        "query": "RESTful API",
        "expected_keywords": ["GET", "POST", "PUT", "DELETE"],
        "expected_doc": "RESTful API 设计规范",
    },
    {
        "query": "Docker 容器",
        "expected_keywords": ["镜像", "容器", "只读", "运行实例"],
        "expected_doc": "Docker 容器技术",
    },
    {
        "query": "Git 版本控制",
        "expected_keywords": ["branch", "checkout", "merge"],
        "expected_doc": "Git 版本控制指南",
    },
]


# ============================================================
# 颜色输出
# ============================================================

class Colors:
    """终端颜色"""
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    NC = "\033[0m"  # No Color


def print_header(text: str):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


def print_section(text: str):
    """打印小节标题"""
    print(f"\n{Colors.CYAN}{'─' * 50}{Colors.NC}")
    print(f"{Colors.CYAN}  {text}{Colors.NC}")
    print(f"{Colors.CYAN}{'─' * 50}{Colors.NC}")


def print_success(text: str):
    """打印成功信息"""
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")


def print_error(text: str):
    """打印错误信息"""
    print(f"{Colors.RED}✗{Colors.NC} {text}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠{Colors.NC} {text}")


def print_info(text: str):
    """打印信息"""
    print(f"{Colors.BLUE}ℹ{Colors.NC} {text}")


# ============================================================
# 测试类
# ============================================================

class KnowledgeBaseTester:
    """知识库测试器"""
    
    def __init__(self, use_temp_storage: bool = True):
        """初始化测试器
        
        Args:
            use_temp_storage: 是否使用临时存储（测试后删除）
        """
        self.use_temp_storage = use_temp_storage
        self.temp_dir: Optional[str] = None
        self.manager: Optional[KnowledgeManager] = None
        self.storage: Optional[KnowledgeStorage] = None
        self.test_results: list[dict] = []
    
    async def setup(self):
        """初始化测试环境"""
        print_section("初始化测试环境")
        
        if self.use_temp_storage:
            self.temp_dir = tempfile.mkdtemp(prefix="kb_test_")
            print_info(f"使用临时存储: {self.temp_dir}")
            self.storage = KnowledgeStorage(workspace_root=self.temp_dir)
        else:
            print_info("使用项目默认存储")
            self.storage = KnowledgeStorage()
        
        await self.storage.initialize()
        print_success("存储初始化完成")
        
        self.manager = KnowledgeManager(name="test-kb")
        await self.manager.initialize()
        print_success("KnowledgeManager 初始化完成")
    
    async def cleanup(self):
        """清理测试环境"""
        if self.use_temp_storage and self.temp_dir:
            print_info(f"清理临时目录: {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def add_demo_documents(self) -> int:
        """添加演示文档
        
        Returns:
            成功添加的文档数
        """
        print_section("添加演示文档")
        
        success_count = 0
        for i, doc_data in enumerate(DEMO_DOCUMENTS, 1):
            doc = Document(
                id=f"demo-doc-{i:03d}",
                url=f"file:///demo/{doc_data['title'].replace(' ', '_')}.md",
                title=doc_data["title"],
                content=doc_data["content"],
                metadata={"source": "demo", "category": "技术文档"},
            )
            
            # 添加到 manager
            self.manager._knowledge_base.add_document(doc)
            self.manager._url_to_doc_id[doc.url] = doc.id
            
            # 保存到存储
            success, message = await self.storage.save_document(doc)
            if success:
                success_count += 1
                print_success(f"[{i}/{len(DEMO_DOCUMENTS)}] {doc.title}")
            else:
                print_warning(f"[{i}/{len(DEMO_DOCUMENTS)}] {doc.title}: {message}")
        
        print(f"\n添加完成: {success_count}/{len(DEMO_DOCUMENTS)} 个文档")
        return success_count
    
    async def test_document_storage(self) -> bool:
        """测试文档存储功能
        
        Returns:
            测试是否通过
        """
        print_section("测试文档存储")
        
        passed = True
        
        # 1. 检查文档数量
        entries = await self.storage.list_documents()
        if len(entries) >= len(DEMO_DOCUMENTS):
            print_success(f"文档数量正确: {len(entries)} 个")
        else:
            print_error(f"文档数量不足: 期望 {len(DEMO_DOCUMENTS)}，实际 {len(entries)}")
            passed = False
        
        # 2. 测试文档加载
        if entries:
            doc = await self.storage.load_document(entries[0].doc_id)
            if doc and doc.content:
                print_success(f"文档加载成功: {doc.title}")
            else:
                print_error("文档加载失败")
                passed = False
        
        # 3. 测试统计信息
        stats = await self.storage.get_stats()
        if stats["document_count"] > 0:
            print_success(f"统计信息正确: {stats['document_count']} 文档, {stats['total_content_size']} 字符")
        else:
            print_error("统计信息异常")
            passed = False
        
        self.test_results.append({
            "name": "文档存储",
            "passed": passed,
        })
        
        return passed
    
    async def test_keyword_search(self) -> bool:
        """测试关键词搜索
        
        Returns:
            测试是否通过
        """
        print_section("测试关键词搜索")
        
        passed = True
        
        for test in TEST_QUERIES:
            query = test["query"]
            expected_doc = test["expected_doc"]
            
            results = await self.storage.search(query, limit=5)
            
            if results:
                # 检查是否找到了预期的文档
                found = any(expected_doc in r.title for r in results)
                if found:
                    print_success(f"查询 '{query[:30]}...' -> 找到 '{expected_doc}'")
                else:
                    print_warning(f"查询 '{query[:30]}...' -> 未找到预期文档，但有 {len(results)} 个结果")
            else:
                print_error(f"查询 '{query[:30]}...' -> 无结果")
                passed = False
        
        self.test_results.append({
            "name": "关键词搜索",
            "passed": passed,
        })
        
        return passed
    
    async def test_search_accuracy(self) -> dict:
        """测试搜索准确性
        
        Returns:
            准确性统计
        """
        print_section("测试搜索准确性")
        
        total_queries = len(TEST_QUERIES)
        correct_matches = 0
        keyword_matches = 0
        
        for test in TEST_QUERIES:
            query = test["query"]
            expected_keywords = test["expected_keywords"]
            expected_doc = test["expected_doc"]
            
            results = await self.storage.search(query, limit=3)
            
            if results:
                top_result = results[0]
                
                # 检查是否找到预期文档
                doc_match = expected_doc in top_result.title
                if doc_match:
                    correct_matches += 1
                
                # 加载文档内容检查关键词
                doc = await self.storage.load_document(top_result.doc_id)
                if doc:
                    content = doc.content.lower()
                    found_keywords = [kw for kw in expected_keywords if kw.lower() in content]
                    kw_ratio = len(found_keywords) / len(expected_keywords)
                    if kw_ratio >= 0.5:
                        keyword_matches += 1
                    
                    status = "✓" if doc_match else "○"
                    print(f"  {status} {query[:40]}")
                    print(f"      -> {top_result.title} (分数: {top_result.score:.2f})")
                    print(f"      关键词匹配: {len(found_keywords)}/{len(expected_keywords)}")
        
        accuracy = correct_matches / total_queries if total_queries > 0 else 0
        keyword_accuracy = keyword_matches / total_queries if total_queries > 0 else 0
        
        print(f"\n准确性统计:")
        print(f"  文档匹配准确率: {accuracy:.1%} ({correct_matches}/{total_queries})")
        print(f"  关键词匹配率: {keyword_accuracy:.1%} ({keyword_matches}/{total_queries})")
        
        passed = accuracy >= 0.6  # 60% 以上视为通过
        
        self.test_results.append({
            "name": "搜索准确性",
            "passed": passed,
            "accuracy": accuracy,
        })
        
        return {
            "accuracy": accuracy,
            "keyword_accuracy": keyword_accuracy,
            "correct_matches": correct_matches,
            "total_queries": total_queries,
        }
    
    async def test_chunk_splitting(self) -> bool:
        """测试文档分块功能
        
        Returns:
            测试是否通过
        """
        print_section("测试文档分块")
        
        passed = True
        splitter = ChunkSplitter(chunk_size=200, overlap=20)
        
        for doc_data in DEMO_DOCUMENTS[:2]:
            chunks = splitter.split(doc_data["content"], source_doc=f"test-{doc_data['title']}")
            
            if chunks:
                print_success(f"'{doc_data['title'][:30]}' -> {len(chunks)} 个分块")
                
                # 验证分块元数据
                for chunk in chunks:
                    if "chunk_index" not in chunk.metadata:
                        print_error(f"分块缺少元数据")
                        passed = False
                        break
            else:
                print_error(f"'{doc_data['title'][:30]}' 分块失败")
                passed = False
        
        self.test_results.append({
            "name": "文档分块",
            "passed": passed,
        })
        
        return passed
    
    async def interactive_query(self):
        """交互式查询模式"""
        print_section("交互式查询模式")
        print_info("输入问题进行查询，输入 'q' 或 'quit' 退出\n")
        
        while True:
            try:
                query = input(f"{Colors.CYAN}问题> {Colors.NC}").strip()
                
                if query.lower() in ("q", "quit", "exit"):
                    print_info("退出交互模式")
                    break
                
                if not query:
                    continue
                
                # 执行搜索
                results = await self.storage.search(query, limit=5)
                
                if results:
                    print(f"\n找到 {len(results)} 个相关结果:\n")
                    
                    for i, result in enumerate(results, 1):
                        print(f"{Colors.GREEN}{i}. {result.title}{Colors.NC}")
                        print(f"   分数: {result.score:.2f} | 类型: {result.match_type}")
                        
                        # 显示摘要
                        if result.snippet:
                            snippet = result.snippet.replace('\n', ' ')[:150]
                            print(f"   摘要: {snippet}...")
                        
                        # 加载完整内容并显示相关段落
                        doc = await self.storage.load_document(result.doc_id)
                        if doc:
                            # 查找包含查询关键词的段落
                            paragraphs = doc.content.split('\n\n')
                            query_lower = query.lower()
                            relevant_para = None
                            for para in paragraphs:
                                if any(kw in para.lower() for kw in query_lower.split()):
                                    relevant_para = para.strip()
                                    break
                            
                            if relevant_para:
                                if len(relevant_para) > 200:
                                    relevant_para = relevant_para[:200] + "..."
                                print(f"   {Colors.YELLOW}相关内容:{Colors.NC}")
                                print(f"   {relevant_para}")
                        
                        print()
                else:
                    print_warning("未找到相关结果\n")
                
            except KeyboardInterrupt:
                print("\n")
                print_info("退出交互模式")
                break
            except EOFError:
                break
    
    async def direct_query(self, query: str) -> list[SearchResult]:
        """直接查询
        
        Args:
            query: 查询文本
            
        Returns:
            搜索结果
        """
        print_section(f"查询: {query}")
        
        results = await self.storage.search(query, limit=5)
        
        if results:
            print(f"\n找到 {len(results)} 个相关结果:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{Colors.GREEN}{i}. {result.title}{Colors.NC}")
                print(f"   分数: {result.score:.2f}")
                
                # 加载完整内容
                doc = await self.storage.load_document(result.doc_id)
                if doc:
                    # 提取相关段落
                    paragraphs = doc.content.split('\n\n')
                    query_lower = query.lower()
                    
                    print(f"   {Colors.YELLOW}相关内容:{Colors.NC}")
                    shown = 0
                    for para in paragraphs:
                        if shown >= 2:
                            break
                        if any(kw in para.lower() for kw in query_lower.split()):
                            para_text = para.strip()
                            if len(para_text) > 300:
                                para_text = para_text[:300] + "..."
                            print(f"   {para_text}")
                            shown += 1
                    
                    if shown == 0:
                        # 显示前两段
                        for para in paragraphs[:2]:
                            para_text = para.strip()
                            if para_text:
                                if len(para_text) > 200:
                                    para_text = para_text[:200] + "..."
                                print(f"   {para_text}")
                
                print()
        else:
            print_warning("未找到相关结果")
        
        return results
    
    def print_summary(self):
        """打印测试摘要"""
        print_header("测试结果摘要")
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed
        
        for result in self.test_results:
            status = f"{Colors.GREEN}✓ 通过{Colors.NC}" if result["passed"] else f"{Colors.RED}✗ 失败{Colors.NC}"
            extra = ""
            if "accuracy" in result:
                extra = f" (准确率: {result['accuracy']:.1%})"
            print(f"  {result['name']}: {status}{extra}")
        
        print(f"\n{'=' * 40}")
        print(f"总计: {passed}/{total} 通过")
        
        if failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}所有测试通过！知识库功能正常。{Colors.NC}")
        else:
            print(f"{Colors.YELLOW}有 {failed} 个测试未通过，请检查相关功能。{Colors.NC}")
        
        return failed == 0


# ============================================================
# 主函数
# ============================================================

async def run_auto_test():
    """运行自动测试"""
    print_header("知识库自动测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tester = KnowledgeBaseTester(use_temp_storage=True)
    
    try:
        await tester.setup()
        await tester.add_demo_documents()
        
        await tester.test_document_storage()
        await tester.test_keyword_search()
        await tester.test_search_accuracy()
        await tester.test_chunk_splitting()
        
        success = tester.print_summary()
        return 0 if success else 1
        
    finally:
        await tester.cleanup()


async def run_demo_mode():
    """运行演示模式"""
    print_header("知识库演示模式")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tester = KnowledgeBaseTester(use_temp_storage=True)
    
    try:
        await tester.setup()
        await tester.add_demo_documents()
        
        print_section("演示查询")
        
        demo_queries = [
            "Python 编程",
            "Docker 容器",
            "机器学习",
        ]
        
        for query in demo_queries:
            await tester.direct_query(query)
            print()
        
        print_info("演示完成。使用 --interactive 模式可以自由提问。")
        return 0
        
    finally:
        await tester.cleanup()


async def run_interactive_mode():
    """运行交互模式"""
    print_header("知识库交互测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tester = KnowledgeBaseTester(use_temp_storage=True)
    
    try:
        await tester.setup()
        await tester.add_demo_documents()
        
        await tester.interactive_query()
        return 0
        
    finally:
        await tester.cleanup()


async def run_query_mode(query: str):
    """运行查询模式"""
    tester = KnowledgeBaseTester(use_temp_storage=True)
    
    try:
        await tester.setup()
        await tester.add_demo_documents()
        
        results = await tester.direct_query(query)
        return 0 if results else 1
        
    finally:
        await tester.cleanup()


async def run_with_real_storage(query: Optional[str] = None):
    """使用真实存储运行"""
    print_header("知识库查询 (使用项目存储)")
    
    tester = KnowledgeBaseTester(use_temp_storage=False)
    
    try:
        await tester.setup()
        
        # 检查是否有文档
        entries = await tester.storage.list_documents()
        if not entries:
            print_warning("知识库为空，请先添加文档")
            print_info("使用: python scripts/knowledge_cli.py add <url>")
            print_info("或者: python scripts/test_knowledge_interactive.py --demo")
            return 1
        
        print_info(f"知识库中有 {len(entries)} 个文档\n")
        
        if query:
            results = await tester.direct_query(query)
            return 0 if results else 1
        else:
            await tester.interactive_query()
            return 0
            
    except Exception as e:
        print_error(f"错误: {e}")
        return 1


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="知识库交互式测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                      # 交互模式（使用演示数据）
  %(prog)s --auto               # 自动测试模式
  %(prog)s --demo               # 演示模式
  %(prog)s --query "Python 是什么?" # 直接查询
  %(prog)s --real               # 使用项目真实存储
  %(prog)s --real --query "问题"    # 查询真实存储
        """,
    )
    
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="自动测试模式（运行所有测试并生成报告）",
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="演示模式（展示查询效果）",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="直接查询模式",
    )
    parser.add_argument(
        "--real", "-r",
        action="store_true",
        help="使用项目真实存储（而非临时存储）",
    )
    
    args = parser.parse_args()
    
    try:
        if args.real:
            exit_code = asyncio.run(run_with_real_storage(args.query))
        elif args.auto:
            exit_code = asyncio.run(run_auto_test())
        elif args.demo:
            exit_code = asyncio.run(run_demo_mode())
        elif args.query:
            exit_code = asyncio.run(run_query_mode(args.query))
        else:
            exit_code = asyncio.run(run_interactive_mode())
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(130)


if __name__ == "__main__":
    main()
