#!/bin/bash
# verify_no_key_iterate.sh - 验证无 API Key 环境下 Iterate 模式行为
#
# 验证内容:
# 1. run.py --mode iterate --print-config 输出 requested_mode/effective_mode/orchestrator 稳定
# 2. scripts/run_iterate.py --minimal --execution-mode auto 能正常结束（不触网/不写入）
# 3. 用 grep 断言关键行正确
#
# 用法:
#   bash scripts/verify_no_key_iterate.sh
#   bash scripts/verify_no_key_iterate.sh --quiet   # 静默模式（仅输出结果）
#   bash scripts/verify_no_key_iterate.sh --ci      # CI 模式（非交互式）

set -e

# ============================================================
# 参数解析
# ============================================================
QUIET_MODE=false
CI_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quiet|-q)
            QUIET_MODE=true
            shift
            ;;
        --ci)
            CI_MODE=true
            QUIET_MODE=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "验证无 API Key 环境下 Iterate 模式行为:"
            echo "  - run.py --mode iterate --print-config 输出稳定性"
            echo "  - scripts/run_iterate.py --minimal 模式正常结束"
            echo ""
            echo "选项:"
            echo "  --quiet, -q   静默模式（仅输出结果）"
            echo "  --ci          CI 模式（非交互式、静默）"
            echo "  --help, -h    显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# 颜色定义
# ============================================================
if [ "$CI_MODE" = true ] || [ -n "${NO_COLOR:-}" ]; then
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
fi

# ============================================================
# 辅助函数
# ============================================================
log_info() {
    if [ "$QUIET_MODE" != true ]; then
        echo -e "${BLUE}ℹ${NC} $1"
    fi
}

log_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ============================================================
# 主测试
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 明确清空 API Key 环境变量（双重保障）
unset CURSOR_API_KEY
unset CURSOR_CLOUD_API_KEY
export CURSOR_API_KEY=""
export CURSOR_CLOUD_API_KEY=""

log_info "工作目录: $PROJECT_ROOT"
log_info "清空 API Key 环境变量..."

ERRORS=0
TEMP_OUTPUT=$(mktemp)
trap "rm -f $TEMP_OUTPUT" EXIT

# ============================================================
# 测试 1: run.py --mode iterate --print-config
# ============================================================
log_info "测试 1: run.py --mode iterate --print-config"

python3 run.py --mode iterate --print-config > "$TEMP_OUTPUT" 2>&1
TEST1_EXIT=$?

if [ "$TEST1_EXIT" -ne 0 ]; then
    log_fail "run.py --print-config 执行失败 (exit=$TEST1_EXIT)"
    ((ERRORS++))
else
    # 检查 requested_mode: auto
    if grep -q "requested_mode: auto" "$TEMP_OUTPUT"; then
        log_pass "requested_mode: auto"
    else
        log_fail "requested_mode 不为 auto"
        grep "requested_mode" "$TEMP_OUTPUT" || echo "(未找到 requested_mode)"
        ((ERRORS++))
    fi

    # 检查 effective_mode: cli
    if grep -q "effective_mode: cli" "$TEMP_OUTPUT"; then
        log_pass "effective_mode: cli（无 API Key 时回退）"
    else
        log_fail "effective_mode 不为 cli（预期无 Key 时回退）"
        grep "effective_mode" "$TEMP_OUTPUT" || echo "(未找到 effective_mode)"
        ((ERRORS++))
    fi

    # 检查 orchestrator: basic
    if grep -q "orchestrator: basic" "$TEMP_OUTPUT"; then
        log_pass "orchestrator: basic（requested_mode=auto 强制 basic）"
    else
        log_fail "orchestrator 不为 basic（预期 auto 模式强制 basic）"
        grep "orchestrator:" "$TEMP_OUTPUT" || echo "(未找到 orchestrator)"
        ((ERRORS++))
    fi
fi

# ============================================================
# 测试 2: scripts/run_iterate.py --minimal --execution-mode auto
# ============================================================
log_info "测试 2: scripts/run_iterate.py --minimal --execution-mode auto"

python3 scripts/run_iterate.py --minimal --execution-mode auto "分析代码结构" > "$TEMP_OUTPUT" 2>&1
TEST2_EXIT=$?

if [ "$TEST2_EXIT" -eq 0 ]; then
    log_pass "minimal 模式正常结束 (exit=0)"
    
    # 验证 minimal 模式确实跳过了关键步骤
    if grep -q "minimal 模式" "$TEMP_OUTPUT"; then
        log_pass "输出包含 minimal 模式标识"
    else
        log_warn "输出未包含 minimal 模式标识（非致命）"
    fi
    
    # 验证跳过了在线检查
    if grep -q "跳过在线" "$TEMP_OUTPUT"; then
        log_pass "跳过了在线检查"
    else
        log_warn "未检测到跳过在线检查（非致命）"
    fi
else
    log_fail "minimal 模式执行失败 (exit=$TEST2_EXIT)"
    if [ "$QUIET_MODE" != true ]; then
        echo "--- 输出摘要 ---"
        tail -20 "$TEMP_OUTPUT"
        echo "----------------"
    fi
    ((ERRORS++))
fi

# ============================================================
# 测试 3: scripts/run_iterate.py --print-config --execution-mode auto
# ============================================================
log_info "测试 3: scripts/run_iterate.py --print-config --execution-mode auto"

python3 scripts/run_iterate.py --print-config --execution-mode auto > "$TEMP_OUTPUT" 2>&1
TEST3_EXIT=$?

if [ "$TEST3_EXIT" -ne 0 ]; then
    log_fail "scripts/run_iterate.py --print-config 执行失败 (exit=$TEST3_EXIT)"
    ((ERRORS++))
else
    # 验证输出一致性
    if grep -q "requested_mode: auto" "$TEMP_OUTPUT"; then
        log_pass "scripts/run_iterate.py 输出 requested_mode: auto"
    else
        log_fail "scripts/run_iterate.py 输出不包含 requested_mode: auto"
        ((ERRORS++))
    fi

    if grep -q "effective_mode: cli" "$TEMP_OUTPUT"; then
        log_pass "scripts/run_iterate.py 输出 effective_mode: cli"
    else
        log_fail "scripts/run_iterate.py 输出不包含 effective_mode: cli"
        ((ERRORS++))
    fi
fi

# ============================================================
# 结果汇总
# ============================================================
echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  所有验证通过! 无 API Key Iterate 行为正确${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  发现 $ERRORS 个验证失败${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
