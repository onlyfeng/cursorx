#!/bin/bash
# Pre-commit 检查 hook
# 在 Agent 结束时执行预提交验证
# 调用 scripts/pre_commit_check.py 进行全面检查
#
# 用法:
#   ./pre_commit.sh           # 完整检查
#   ./pre_commit.sh --quick   # 快速检查（仅语法和导入）

set -e

# 获取项目根目录（相对于 .cursor/hooks/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 解析参数
QUICK_MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE="--quick"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# 检查 Python 脚本是否存在
PRE_COMMIT_SCRIPT="$PROJECT_ROOT/scripts/pre_commit_check.py"

if [[ ! -f "$PRE_COMMIT_SCRIPT" ]]; then
    echo -e "${YELLOW}⚠ 预提交检查脚本不存在: $PRE_COMMIT_SCRIPT${NC}"
    exit 0  # 脚本不存在时不阻止操作
fi

if [[ -n "$QUICK_MODE" ]]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  执行快速预提交检查（语法和导入）...${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  执行预提交检查...${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
fi

# 切换到项目根目录执行检查
cd "$PROJECT_ROOT"

# 执行预提交检查
if python3 "$PRE_COMMIT_SCRIPT" $QUICK_MODE 2>&1; then
    echo -e "\n${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}${BOLD}  ✓ 预提交检查通过，可以安全提交${NC}"
    echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
else
    EXIT_CODE=$?
    echo -e "\n${RED}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}${BOLD}  ✗ 预提交检查失败！${NC}"
    echo -e "${RED}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}修复建议:${NC}"
    echo -e "  1. 查看上方的详细错误信息"
    echo -e "  2. 运行 ${BOLD}python scripts/pre_commit_check.py --verbose${NC} 获取更多细节"
    echo -e "  3. 运行 ${BOLD}python scripts/pre_commit_check.py --json${NC} 获取结构化输出"
    echo -e "  4. 修复问题后重新提交"
    echo ""
    echo -e "${YELLOW}常见问题:${NC}"
    echo -e "  • 依赖缺失: ${BOLD}pip install -r requirements.txt${NC}"
    echo -e "  • 语法错误: 检查报告中标记的文件"
    echo -e "  • 导入失败: 检查模块依赖关系"
    echo ""
    
    # 返回非零退出码阻止提交
    exit $EXIT_CODE
fi
