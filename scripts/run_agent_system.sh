#!/bin/bash
# run_agent_system.sh - 运行多 Agent 系统
# 规划者-执行者-评审者 工作流

set -e

GOAL="$1"
MAX_ITERATIONS="${2:-3}"
WORKERS="${3:-3}"

if [ -z "$GOAL" ]; then
  echo "用法: $0 \"目标描述\" [最大迭代次数] [Worker数量]"
  echo "示例: $0 \"实现用户登录功能\" 5 3"
  exit 1
fi

echo "🚀 启动多 Agent 系统"
echo "   目标: $GOAL"
echo "   最大迭代: $MAX_ITERATIONS"
echo "   Worker 数量: $WORKERS"
echo ""

# 激活 conda 环境（如果需要）
if command -v conda &> /dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cursorx 2>/dev/null || true
fi

# 运行多进程版本
python main_mp.py "$GOAL" \
  --max-iterations "$MAX_ITERATIONS" \
  --workers "$WORKERS" \
  --planner-model gpt-5.2-high \
  --worker-model opus-4.5-thinking

echo ""
echo "🎉 完成"
