#!/bin/bash
# run_agent_system.sh - 运行多 Agent 系统
# 规划者-执行者-评审者 工作流
#
# 本脚本是 run.py --mode mp 的简化包装器。
# 仅做输入校验和参数透传，所有配置优先级处理由 run.py 和 core.config 统一负责。
#
# 配置优先级（由 run.py/core.config 处理）:
#   1. CLI 参数（最高）
#   2. 环境变量
#   3. config.yaml
#   4. 代码默认值（最低）
#
# 支持的环境变量（透传给 run.py）:
#   AGENT_MAX_ITERATIONS  - 最大迭代次数
#   AGENT_WORKERS         - Worker 数量
#   AGENT_PLANNER_MODEL   - 规划者模型
#   AGENT_WORKER_MODEL    - 执行者模型

set -e

# ============================================================
# 项目根目录定位
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ============================================================
# 入口文件自检
# ============================================================
ENTRY_FILE="$PROJECT_ROOT/run.py"
if [ ! -f "$ENTRY_FILE" ]; then
  echo "❌ 错误: 入口文件不存在: $ENTRY_FILE"
  echo "   请确保在项目根目录下运行，或检查 run.py 是否存在"
  exit 1
fi

# ============================================================
# 参数解析
# ============================================================
GOAL="$1"

if [ -z "$GOAL" ]; then
  echo "用法: $0 \"目标描述\" [最大迭代次数] [Worker数量]"
  echo "示例: $0 \"实现用户登录功能\" 5 3"
  echo ""
  echo "参数优先级由 run.py 统一处理: CLI 参数 > 环境变量 > config.yaml > 默认值"
  echo ""
  echo "环境变量:"
  echo "  AGENT_MAX_ITERATIONS  - 最大迭代次数"
  echo "  AGENT_WORKERS         - Worker 数量"
  echo "  AGENT_PLANNER_MODEL   - 规划者模型"
  echo "  AGENT_WORKER_MODEL    - 执行者模型"
  exit 1
fi

# ============================================================
# 构建 CLI 参数列表
# ============================================================
# 优先级: 脚本参数 > 环境变量（由 run.py 内部处理 config.yaml 和默认值）

CLI_ARGS=()

# 最大迭代次数: 脚本参数 $2 > 环境变量 AGENT_MAX_ITERATIONS
if [ -n "$2" ]; then
  CLI_ARGS+=(--max-iterations "$2")
  MAX_ITERATIONS_SOURCE="脚本参数"
  MAX_ITERATIONS_VALUE="$2"
elif [ -n "$AGENT_MAX_ITERATIONS" ]; then
  CLI_ARGS+=(--max-iterations "$AGENT_MAX_ITERATIONS")
  MAX_ITERATIONS_SOURCE="环境变量 AGENT_MAX_ITERATIONS"
  MAX_ITERATIONS_VALUE="$AGENT_MAX_ITERATIONS"
else
  MAX_ITERATIONS_SOURCE="config.yaml/默认值 (由 run.py 处理)"
  MAX_ITERATIONS_VALUE="(自动)"
fi

# Worker 数量: 脚本参数 $3 > 环境变量 AGENT_WORKERS
if [ -n "$3" ]; then
  CLI_ARGS+=(--workers "$3")
  WORKERS_SOURCE="脚本参数"
  WORKERS_VALUE="$3"
elif [ -n "$AGENT_WORKERS" ]; then
  CLI_ARGS+=(--workers "$AGENT_WORKERS")
  WORKERS_SOURCE="环境变量 AGENT_WORKERS"
  WORKERS_VALUE="$AGENT_WORKERS"
else
  WORKERS_SOURCE="config.yaml/默认值 (由 run.py 处理)"
  WORKERS_VALUE="(自动)"
fi

# 模型参数: 仅在环境变量显式指定时传递
if [ -n "$AGENT_PLANNER_MODEL" ]; then
  CLI_ARGS+=(--planner-model "$AGENT_PLANNER_MODEL")
  PLANNER_MODEL_SOURCE="环境变量 AGENT_PLANNER_MODEL"
  PLANNER_MODEL_VALUE="$AGENT_PLANNER_MODEL"
else
  PLANNER_MODEL_SOURCE="config.yaml/默认值 (由 run.py 处理)"
  PLANNER_MODEL_VALUE="(自动)"
fi

if [ -n "$AGENT_WORKER_MODEL" ]; then
  CLI_ARGS+=(--worker-model "$AGENT_WORKER_MODEL")
  WORKER_MODEL_SOURCE="环境变量 AGENT_WORKER_MODEL"
  WORKER_MODEL_VALUE="$AGENT_WORKER_MODEL"
else
  WORKER_MODEL_SOURCE="config.yaml/默认值 (由 run.py 处理)"
  WORKER_MODEL_VALUE="(自动)"
fi

# ============================================================
# 激活 conda 环境（如果需要）
# ============================================================
if command -v conda &> /dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cursorx 2>/dev/null || true
fi

# ============================================================
# 打印参数来源信息
# ============================================================
echo "🚀 启动多 Agent 系统"
echo "   目标: $GOAL"
echo ""
echo "📋 参数配置:"
echo "   最大迭代: $MAX_ITERATIONS_VALUE (来源: $MAX_ITERATIONS_SOURCE)"
echo "   Worker 数量: $WORKERS_VALUE (来源: $WORKERS_SOURCE)"
echo "   规划者模型: $PLANNER_MODEL_VALUE (来源: $PLANNER_MODEL_SOURCE)"
echo "   执行者模型: $WORKER_MODEL_VALUE (来源: $WORKER_MODEL_SOURCE)"
echo ""

# ============================================================
# 运行多进程版本
# ============================================================
cd "$PROJECT_ROOT"
python run.py --mode mp "$GOAL" "${CLI_ARGS[@]}"

echo ""
echo "🎉 完成"
