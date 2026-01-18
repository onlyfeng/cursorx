#!/usr/bin/env python3
"""CursorX 统一入口脚本

提供统一的命令行入口，支持多种运行模式和自然语言任务描述。

运行模式：
  - basic:     基本协程模式（规划-执行-审核）
  - mp:        多进程模式（并行执行）
  - knowledge: 知识库增强模式（自动搜索相关文档）
  - iterate:   自我迭代模式（检查更新、更新知识库、执行任务）
  - auto:      自动分析模式（使用 Agent 分析任务并选择最佳模式）

用法示例：
  # 显式指定模式
  python run.py --mode basic "实现 REST API"
  python run.py --mode mp "重构代码" --workers 5
  python run.py --mode iterate "更新 CLI 支持"
  
  # 自动模式（Agent 分析任务）
  python run.py "启动自我迭代，跳过在线更新"
  python run.py "使用多进程模式重构 src 目录"
  
  # 无限迭代
  python run.py "实现功能" --max-iterations MAX
"""
import argparse
import asyncio
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger


# ============================================================
# 运行模式定义
# ============================================================

class RunMode(str, Enum):
    """运行模式"""
    BASIC = "basic"           # 基本协程模式
    MP = "mp"                 # 多进程模式
    KNOWLEDGE = "knowledge"   # 知识库增强模式
    ITERATE = "iterate"       # 自我迭代模式
    AUTO = "auto"             # 自动分析模式


# 模式别名映射
MODE_ALIASES = {
    "default": RunMode.BASIC,
    "basic": RunMode.BASIC,
    "simple": RunMode.BASIC,
    "mp": RunMode.MP,
    "multiprocess": RunMode.MP,
    "parallel": RunMode.MP,
    "knowledge": RunMode.KNOWLEDGE,
    "kb": RunMode.KNOWLEDGE,
    "docs": RunMode.KNOWLEDGE,
    "iterate": RunMode.ITERATE,
    "self-iterate": RunMode.ITERATE,
    "self": RunMode.ITERATE,
    "update": RunMode.ITERATE,
    "auto": RunMode.AUTO,
    "smart": RunMode.AUTO,
}


@dataclass
class TaskAnalysis:
    """任务分析结果"""
    mode: RunMode = RunMode.BASIC
    goal: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


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


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.NC} {text}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.NC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.NC} {text}")


# ============================================================
# 参数解析
# ============================================================

def parse_max_iterations(value: str) -> int:
    """解析最大迭代次数参数"""
    value_upper = value.upper().strip()
    if value_upper in ("MAX", "UNLIMITED", "INF", "INFINITE"):
        return -1
    try:
        num = int(value)
        return -1 if num <= 0 else num
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"无效的迭代次数: {value}。使用正整数或 MAX/-1 表示无限迭代"
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CursorX 统一入口 - 多 Agent 协作系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式：
  basic      基本协程模式（默认）
  mp         多进程模式（并行执行）
  knowledge  知识库增强模式
  iterate    自我迭代模式
  auto       自动分析模式（Agent 分析任务选择最佳模式）

示例：
  # 显式指定模式
  python run.py --mode basic "实现 REST API"
  python run.py --mode mp "重构代码" --workers 5
  python run.py --mode iterate --skip-online "优化 CLI"
  
  # 自动模式（自然语言描述任务）
  python run.py "启动自我迭代，跳过在线更新，优化代码"
  python run.py "使用多进程并行重构 src 目录下的代码"
  
  # 无限迭代直到完成
  python run.py "实现功能" --max-iterations MAX
        """,
    )
    
    # 任务描述（可选，支持自然语言）
    parser.add_argument(
        "task",
        nargs="?",
        default="",
        help="任务描述（支持自然语言，会自动分析并路由到合适的模式）",
    )
    
    # 运行模式
    parser.add_argument(
        "--mode", "-M",
        type=str,
        default="auto",
        choices=list(MODE_ALIASES.keys()),
        help="运行模式 (默认: auto，自动分析任务选择模式)",
    )
    
    # 通用参数
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=".",
        help="工作目录 (默认: 当前目录)",
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=3,
        help="Worker 数量 (默认: 3)",
    )
    
    parser.add_argument(
        "-m", "--max-iterations",
        type=str,
        default="10",
        help="最大迭代次数 (默认: 10，MAX/-1 表示无限迭代)",
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="启用严格评审模式",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )
    
    # 自我迭代模式专用参数
    parser.add_argument(
        "--skip-online",
        action="store_true",
        help="[iterate] 跳过在线文档检查",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="[iterate] 仅分析不执行",
    )
    
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="[iterate] 强制更新知识库",
    )
    
    # 知识库模式专用参数
    parser.add_argument(
        "--use-knowledge",
        action="store_true",
        help="[knowledge] 使用知识库上下文",
    )
    
    parser.add_argument(
        "--search-knowledge",
        type=str,
        metavar="QUERY",
        help="[knowledge] 搜索知识库获取相关信息",
    )
    
    parser.add_argument(
        "--self-update",
        action="store_true",
        help="[knowledge] 自我更新模式",
    )
    
    # 多进程模式专用参数
    parser.add_argument(
        "--planner-model",
        type=str,
        default="gpt-5.2-high",
        help="[mp] 规划者模型",
    )
    
    parser.add_argument(
        "--worker-model",
        type=str,
        default="opus-4.5-thinking",
        help="[mp] 执行者模型",
    )
    
    # 流式日志
    parser.add_argument(
        "--stream-log",
        action="store_true",
        help="启用流式日志",
    )
    
    # 禁用自动模式分析（直接使用指定模式）
    parser.add_argument(
        "--no-auto-analyze",
        action="store_true",
        help="禁用自动分析，直接使用指定模式",
    )
    
    return parser.parse_args()


# ============================================================
# 自然语言任务分析
# ============================================================

class TaskAnalyzer:
    """任务分析器
    
    使用规则匹配和 Agent 分析自然语言任务，确定最佳运行模式。
    """
    
    # 模式关键词映射
    MODE_KEYWORDS = {
        RunMode.ITERATE: [
            "自我迭代", "self-iterate", "iterate", "迭代更新",
            "更新知识库", "检查更新", "changelog", "自我更新",
        ],
        RunMode.MP: [
            "多进程", "multiprocess", "并行", "parallel",
            "多worker", "多 worker", "并发",
        ],
        RunMode.KNOWLEDGE: [
            "知识库", "knowledge", "文档搜索", "搜索文档",
            "cursor 文档", "参考文档", "docs",
        ],
    }
    
    # 选项关键词映射
    OPTION_KEYWORDS = {
        "skip_online": ["跳过在线", "skip-online", "离线", "不检查更新", "跳过更新"],
        "dry_run": ["仅分析", "dry-run", "预览", "不执行"],
        "strict": ["严格", "strict", "严格审核", "严格评审"],
        "force_update": ["强制更新", "force-update", "强制刷新"],
        "self_update": ["自我更新", "self-update", "更新自身"],
        "use_knowledge": ["使用知识库", "use-knowledge", "启用知识库"],
    }
    
    # 无限迭代关键词
    UNLIMITED_KEYWORDS = ["无限", "持续", "一直", "不停", "循环", "max", "unlimited"]
    
    def __init__(self, use_agent: bool = True):
        """初始化分析器
        
        Args:
            use_agent: 是否使用 Agent 进行高级分析
        """
        self.use_agent = use_agent
    
    def analyze(self, task: str, args: argparse.Namespace) -> TaskAnalysis:
        """分析任务描述
        
        Args:
            task: 任务描述
            args: 命令行参数
            
        Returns:
            分析结果
        """
        if not task.strip():
            # 没有任务描述，使用默认模式
            return TaskAnalysis(
                mode=MODE_ALIASES.get(args.mode, RunMode.BASIC),
                goal="",
                reasoning="无任务描述，使用指定模式",
            )
        
        # 先用规则匹配
        analysis = self._rule_based_analysis(task, args)
        
        # 如果启用 Agent 分析且规则匹配不明确，使用 Agent
        if self.use_agent and analysis.mode == RunMode.BASIC:
            agent_analysis = self._agent_analysis(task)
            if agent_analysis:
                # 合并 Agent 分析结果
                analysis.mode = agent_analysis.mode
                analysis.options.update(agent_analysis.options)
                analysis.reasoning = agent_analysis.reasoning
        
        # 确保 goal 不为空
        if not analysis.goal:
            analysis.goal = task
        
        return analysis
    
    def _rule_based_analysis(self, task: str, args: argparse.Namespace) -> TaskAnalysis:
        """基于规则的分析"""
        task_lower = task.lower()
        analysis = TaskAnalysis(goal=task)
        reasoning_parts = []
        
        # 检测模式关键词
        for mode, keywords in self.MODE_KEYWORDS.items():
            if any(kw.lower() in task_lower for kw in keywords):
                analysis.mode = mode
                matched = [kw for kw in keywords if kw.lower() in task_lower]
                reasoning_parts.append(f"检测到 {mode.value} 模式关键词: {matched}")
                break
        
        # 检测选项关键词
        for option, keywords in self.OPTION_KEYWORDS.items():
            if any(kw.lower() in task_lower for kw in keywords):
                analysis.options[option] = True
                reasoning_parts.append(f"检测到选项 {option}")
        
        # 检测无限迭代
        if any(kw in task_lower for kw in self.UNLIMITED_KEYWORDS):
            analysis.options["max_iterations"] = -1
            reasoning_parts.append("检测到无限迭代关键词")
        
        # 提取 Worker 数量
        worker_match = re.search(r'(\d+)\s*(个)?\s*(worker|进程|并行)', task_lower)
        if worker_match:
            analysis.options["workers"] = int(worker_match.group(1))
            reasoning_parts.append(f"提取 Worker 数量: {worker_match.group(1)}")
        
        analysis.reasoning = "; ".join(reasoning_parts) if reasoning_parts else "默认模式"
        return analysis
    
    def _agent_analysis(self, task: str) -> Optional[TaskAnalysis]:
        """使用 Agent 分析任务"""
        try:
            # 构建分析提示
            prompt = f"""分析以下任务描述，确定最佳运行模式和选项。

任务描述: {task}

可用模式：
- basic: 基本协程模式，适合简单任务
- mp: 多进程模式，适合需要并行执行的大型任务
- knowledge: 知识库增强模式，适合需要参考文档的任务
- iterate: 自我迭代模式，适合需要检查更新、更新知识库的任务

可用选项：
- skip_online: 跳过在线文档检查
- dry_run: 仅分析不执行
- strict: 严格评审模式
- max_iterations: 最大迭代次数（-1 表示无限）
- workers: Worker 数量

请以 JSON 格式返回分析结果：
{{
  "mode": "模式名称",
  "options": {{"选项名": 值}},
  "reasoning": "分析理由",
  "refined_goal": "提炼后的任务目标"
}}

仅返回 JSON，不要其他内容。"""

            # 调用 agent CLI
            result = subprocess.run(
                ["agent", "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root),
            )
            
            if result.returncode != 0:
                return None
            
            # 解析 JSON 响应
            output = result.stdout.strip()
            
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', output)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            mode_str = data.get("mode", "basic").lower()
            mode = MODE_ALIASES.get(mode_str, RunMode.BASIC)
            
            return TaskAnalysis(
                mode=mode,
                goal=data.get("refined_goal", task),
                options=data.get("options", {}),
                reasoning=data.get("reasoning", "Agent 分析"),
            )
            
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.debug(f"Agent 分析失败: {e}")
            return None


# ============================================================
# 运行器
# ============================================================

class Runner:
    """运行器
    
    根据分析结果调用对应的入口脚本。
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.max_iterations = parse_max_iterations(args.max_iterations)
    
    async def run(self, analysis: TaskAnalysis) -> dict[str, Any]:
        """运行任务
        
        Args:
            analysis: 任务分析结果
            
        Returns:
            执行结果
        """
        mode = analysis.mode
        goal = analysis.goal or self.args.task
        options = analysis.options
        
        # 合并命令行参数和分析结果
        merged_options = self._merge_options(options)
        
        print_header(f"CursorX - {self._get_mode_name(mode)}")
        print_info(f"任务: {goal}")
        print_info(f"模式: {mode.value}")
        if analysis.reasoning:
            print_info(f"分析: {analysis.reasoning}")
        if merged_options:
            print_info(f"选项: {merged_options}")
        print()
        
        # 根据模式调用对应的运行函数
        if mode == RunMode.BASIC:
            return await self._run_basic(goal, merged_options)
        elif mode == RunMode.MP:
            return await self._run_mp(goal, merged_options)
        elif mode == RunMode.KNOWLEDGE:
            return await self._run_knowledge(goal, merged_options)
        elif mode == RunMode.ITERATE:
            return await self._run_iterate(goal, merged_options)
        else:
            return await self._run_basic(goal, merged_options)
    
    def _merge_options(self, analysis_options: dict) -> dict:
        """合并命令行参数和分析结果"""
        options = {
            "directory": self.args.directory,
            "workers": analysis_options.get("workers", self.args.workers),
            "max_iterations": analysis_options.get("max_iterations", self.max_iterations),
            "strict": analysis_options.get("strict", self.args.strict),
            "verbose": self.args.verbose,
        }
        
        # 自我迭代选项
        if analysis_options.get("skip_online") or self.args.skip_online:
            options["skip_online"] = True
        if analysis_options.get("dry_run") or self.args.dry_run:
            options["dry_run"] = True
        if analysis_options.get("force_update") or self.args.force_update:
            options["force_update"] = True
        
        # 知识库选项
        if analysis_options.get("use_knowledge") or self.args.use_knowledge:
            options["use_knowledge"] = True
        if analysis_options.get("self_update") or self.args.self_update:
            options["self_update"] = True
        if self.args.search_knowledge:
            options["search_knowledge"] = self.args.search_knowledge
        
        # 多进程选项
        options["planner_model"] = self.args.planner_model
        options["worker_model"] = self.args.worker_model
        
        # 流式日志
        options["stream_log"] = self.args.stream_log
        
        return options
    
    def _get_mode_name(self, mode: RunMode) -> str:
        """获取模式显示名称"""
        names = {
            RunMode.BASIC: "基本模式",
            RunMode.MP: "多进程模式",
            RunMode.KNOWLEDGE: "知识库增强模式",
            RunMode.ITERATE: "自我迭代模式",
            RunMode.AUTO: "自动模式",
        }
        return names.get(mode, mode.value)
    
    async def _run_basic(self, goal: str, options: dict) -> dict:
        """运行基本模式"""
        from coordinator import Orchestrator, OrchestratorConfig
        from cursor.client import CursorAgentConfig
        
        cursor_config = CursorAgentConfig(
            working_directory=options["directory"],
        )
        
        config = OrchestratorConfig(
            working_directory=options["directory"],
            max_iterations=options["max_iterations"],
            worker_pool_size=options["workers"],
            strict_review=options.get("strict", False),
            cursor_config=cursor_config,
            stream_events_enabled=options.get("stream_log", False),
        )
        
        if options["max_iterations"] == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")
        
        orchestrator = Orchestrator(config)
        return await orchestrator.run(goal)
    
    async def _run_mp(self, goal: str, options: dict) -> dict:
        """运行多进程模式"""
        from coordinator import MultiProcessOrchestrator, MultiProcessOrchestratorConfig
        
        config = MultiProcessOrchestratorConfig(
            working_directory=options["directory"],
            max_iterations=options["max_iterations"],
            worker_count=options["workers"],
            strict_review=options.get("strict", False),
            planner_model=options.get("planner_model", "gpt-5.2-high"),
            worker_model=options.get("worker_model", "opus-4.5-thinking"),
            reviewer_model=options.get("worker_model", "opus-4.5-thinking"),
            stream_events_enabled=options.get("stream_log", False),
        )
        
        if options["max_iterations"] == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")
        
        orchestrator = MultiProcessOrchestrator(config)
        return await orchestrator.run(goal)
    
    async def _run_knowledge(self, goal: str, options: dict) -> dict:
        """运行知识库增强模式"""
        from coordinator import Orchestrator, OrchestratorConfig
        from cursor.client import CursorAgentConfig
        from knowledge import KnowledgeManager, KnowledgeStorage
        
        # 初始化知识库
        manager = KnowledgeManager(name="cursor-docs")
        await manager.initialize()
        
        # 搜索相关文档
        knowledge_context = None
        if options.get("search_knowledge"):
            storage = KnowledgeStorage()
            await storage.initialize()
            results = await storage.search(options["search_knowledge"], limit=5)
            knowledge_context = []
            for result in results:
                doc = await storage.load_document(result.doc_id)
                if doc:
                    knowledge_context.append({
                        "title": doc.title,
                        "url": doc.url,
                        "content": doc.content[:2000],
                    })
        
        # 构建增强目标
        enhanced_goal = goal
        if knowledge_context:
            context_text = "\n\n## 参考文档（来自知识库）\n\n"
            for i, doc in enumerate(knowledge_context, 1):
                context_text += f"### {i}. {doc['title']}\n"
                context_text += f"```\n{doc['content'][:1000]}\n```\n\n"
            enhanced_goal = f"{goal}\n{context_text}"
        
        cursor_config = CursorAgentConfig(
            working_directory=options["directory"],
        )
        
        config = OrchestratorConfig(
            working_directory=options["directory"],
            max_iterations=options["max_iterations"],
            worker_pool_size=options["workers"],
            strict_review=options.get("strict", False),
            cursor_config=cursor_config,
        )
        
        if options["max_iterations"] == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")
        
        orchestrator = Orchestrator(config, knowledge_manager=manager)
        return await orchestrator.run(enhanced_goal)
    
    async def _run_iterate(self, goal: str, options: dict) -> dict:
        """运行自我迭代模式"""
        # 导入自我迭代模块
        from scripts.run_iterate import (
            SelfIterator,
            IterationContext,
            ChangelogAnalyzer,
            KnowledgeUpdater,
            IterationGoalBuilder,
        )
        
        # 创建模拟的 args 对象
        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                self.requirement = goal
                self.skip_online = opts.get("skip_online", False)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                self.max_iterations = str(opts.get("max_iterations", 5))
                self.workers = opts.get("workers", 3)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
        
        iterate_args = IterateArgs(goal, options)
        iterator = SelfIterator(iterate_args)
        
        return await iterator.run()


# ============================================================
# 结果输出
# ============================================================

def print_result(result: dict) -> None:
    """打印执行结果"""
    print("\n" + "=" * 60)
    print("执行结果")
    print("=" * 60)
    
    success = result.get("success", False)
    print(f"\n状态: {Colors.GREEN if success else Colors.YELLOW}{'成功' if success else '未完成'}{Colors.NC}")
    
    if result.get("goal"):
        goal_preview = result["goal"][:100]
        if len(result["goal"]) > 100:
            goal_preview += "..."
        print(f"目标: {goal_preview}")
    
    if "iterations_completed" in result:
        print(f"完成迭代: {result['iterations_completed']}")
    
    if "total_tasks_created" in result:
        print("\n任务统计:")
        print(f"  - 创建: {result.get('total_tasks_created', 0)}")
        print(f"  - 完成: {result.get('total_tasks_completed', 0)}")
        print(f"  - 失败: {result.get('total_tasks_failed', 0)}")
    
    if result.get("final_score"):
        print(f"\n最终评分: {result['final_score']:.1f}")
    
    if result.get("dry_run"):
        print(f"\n{Colors.YELLOW}(Dry-run 模式，未实际执行){Colors.NC}")
    
    print("=" * 60)


# ============================================================
# 日志配置
# ============================================================

def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
        filter=lambda record: record["level"].name != "DEBUG" or verbose,
    )
    
    # 确保日志目录存在
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        str(log_dir / "run_{time}.log"),
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


# ============================================================
# 主入口
# ============================================================

async def async_main() -> int:
    """异步主函数"""
    args = parse_args()
    setup_logging(args.verbose)
    
    # 解析模式
    explicit_mode = MODE_ALIASES.get(args.mode, RunMode.AUTO)
    
    # 任务分析
    if explicit_mode == RunMode.AUTO and args.task and not args.no_auto_analyze:
        # 自动模式：分析任务
        print_info("分析任务...")
        analyzer = TaskAnalyzer(use_agent=True)
        analysis = analyzer.analyze(args.task, args)
    else:
        # 显式模式或无任务
        if explicit_mode == RunMode.AUTO:
            explicit_mode = RunMode.BASIC
        analysis = TaskAnalysis(
            mode=explicit_mode,
            goal=args.task,
            reasoning="使用指定模式",
        )
    
    # 检查是否有任务
    if not analysis.goal:
        print_error("请提供任务描述")
        print_info("用法: python run.py [--mode MODE] \"任务描述\"")
        return 1
    
    # 运行任务
    runner = Runner(args)
    
    try:
        result = await runner.run(analysis)
        print_result(result)
        return 0 if result.get("success") else 1
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}用户中断{Colors.NC}")
        return 130


def main() -> None:
    """主函数"""
    try:
        exit_code = asyncio.run(async_main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(130)


if __name__ == "__main__":
    main()
