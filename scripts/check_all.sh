#!/bin/bash
# check_all.sh - 一键检查脚本
# 执行全面的项目健康检查
#
# 支持选项:
#   --ci          CI 模式（非交互式、无颜色输出）
#   --json        JSON 输出格式（便于 CI 解析）
#   --fail-fast   遇到失败立即退出
#   --full, -f    执行完整检查
#   --mode MODE   运行特定检查模式（可多次指定）
#   --help, -h    显示帮助信息
#
# 运行模式（--mode）:
#   import        语法检查 + 核心模块导入（对齐 import-test.yml）
#   lint          flake8-critical + ruff + mypy（对齐 lint.yml）
#   test          单元测试 + E2E 测试（对齐 ci.yml test/e2e）
#   minimal       轻量检查（仅 Python 版本 + 基础验证，无额外依赖）
#   all           运行所有检查（默认，等同于不指定 --mode）
#
# 检查项包括:
#   - 依赖一致性检查（check_deps.py + sync-deps.sh check）
#     * 检测版本不匹配、缺失依赖、依赖冲突
#     * 检测未声明的第三方导入（定位到具体文件:行号）
#     * 检测 pyproject.toml 与 .in 文件同步状态
#     * 失败时给出层级（core/dev/test/ml）和修复命令
#
# 环境变量:
#   CI=true       自动启用 CI 模式
#   NO_COLOR=1    禁用颜色输出

set -e

# ============================================================
# 参数解析
# ============================================================

CI_MODE=false
JSON_OUTPUT=false
FAIL_FAST=false
FULL_CHECK=false
SPLIT_TESTS=false
# 检查模式：import/lint/test/all，支持多选
declare -a CHECK_MODES=()
CHECK_MODE_SET=false
TEST_CHUNK_SIZE=8
TEST_TIMEOUT=0
PYTEST_ARGS_STR=""
PYTEST_EXTRA_ARGS=()
PYTEST_PARALLEL=false
PYTEST_PARALLEL_WORKERS="auto"
LOG_DIR=""
LOG_DIR_SET=false
KEEP_LOGS=false
DIAGNOSE_HANG=false
DIAGNOSE_HANG_SET=false
CASE_TIMEOUT=0
SPLIT_COVERAGE=false
# 覆盖率/综合测试超时（秒）
# 默认值偏保守，避免本地环境下单元+覆盖率过早超时；可用 --coverage-timeout 覆盖
COVERAGE_TIMEOUT=1800
E2E_TIMEOUT=900
# 覆盖率阈值（百分比），用于 --cov-fail-under
# 默认 80，CI matrix 仅收集覆盖率时可传 0（覆盖率 gating 收敛到 pr-check full-check）
COV_FAIL_UNDER=80
COV_FAIL_UNDER_SET=false

# Marker 包含控制（默认排除 cloud/network，与 CI 一致）
INCLUDE_NETWORK=false
INCLUDE_CLOUD=false
ALL_MARKERS=false
RUN_NETWORK_ISOLATION=false

# 跟踪脚本是否正常完成（用于 trap 判断）
_CHECK_ALL_COMPLETED=false

# JSON 错误输出（用于参数解析阶段的错误）
# 注意：此函数在 json_escape 定义前调用，使用简单转义
_json_error_output() {
    local message="$1"
    local exit_code="${2:-1}"
    # 简单转义（此时 json_escape 可能未定义）
    message=$(echo "$message" | sed 's/\\/\\\\/g; s/"/\\"/g; s/	/\\t/g' | tr '\n' ' ' | tr '\r' ' ')
    cat << EOF
{
  "success": false,
  "exit_code": $exit_code,
  "error": "$message",
  "summary": {"passed": 0, "failed": 1, "warnings": 0, "skipped": 0, "total": 1},
  "ci_mode": $CI_MODE,
  "fail_fast": $FAIL_FAST,
  "full_check": $FULL_CHECK,
  "timestamp": "$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')",
  "checks": []
}
EOF
}

# trap 处理函数：确保异常退出时 JSON 模式输出有效 JSON
_trap_handler() {
    local exit_code=$?
    # 如果脚本已正常完成，不需要 trap 输出
    if [ "$_CHECK_ALL_COMPLETED" = true ]; then
        return
    fi
    # JSON 模式下输出最小有效 JSON
    if [ "$JSON_OUTPUT" = true ]; then
        _json_error_output "Script terminated unexpectedly (exit_code=$exit_code)" "$exit_code"
    fi
    exit "$exit_code"
}

# 设置 trap（EXIT 信号，确保任何退出都触发）
trap '_trap_handler' EXIT

# 参数错误处理（JSON 模式下输出 JSON，否则输出文本）
_arg_error() {
    local message="$1"
    if [ "$JSON_OUTPUT" = true ]; then
        _json_error_output "$message" 1
    else
        echo "$message" >&2
    fi
    _CHECK_ALL_COMPLETED=true  # 标记完成避免 trap 重复输出
    exit 1
}

# 环境变量检测：CI=true 自动启用 CI 模式
if [ "${CI:-}" = "true" ] || [ "${CI:-}" = "1" ] || [ -n "${GITHUB_ACTIONS:-}" ] || [ -n "${GITLAB_CI:-}" ] || [ -n "${JENKINS_URL:-}" ]; then
    CI_MODE=true
fi

# NO_COLOR 环境变量支持
if [ -n "${NO_COLOR:-}" ]; then
    CI_MODE=true
fi

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ci)
            CI_MODE=true
            shift
            ;;
        --json)
            JSON_OUTPUT=true
            CI_MODE=true  # JSON 输出时自动禁用颜色
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --full|-f)
            FULL_CHECK=true
            shift
            ;;
        --mode)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --mode 需要指定模式 (import/lint/test/all)"
            fi
            case "$2" in
                import|lint|test|all|minimal)
                    CHECK_MODES+=("$2")
                    CHECK_MODE_SET=true
                    ;;
                *)
                    _arg_error "未知模式: $2 (支持的模式: import, lint, test, minimal, all)"
                    ;;
            esac
            shift 2
            ;;
        --split-tests)
            SPLIT_TESTS=true
            shift
            ;;
        --test-chunk-size)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --test-chunk-size 需要数值"
            fi
            TEST_CHUNK_SIZE="$2"
            shift 2
            ;;
        --test-timeout)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --test-timeout 需要数值 (秒)"
            fi
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --case-timeout)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --case-timeout 需要数值 (秒)"
            fi
            CASE_TIMEOUT="$2"
            shift 2
            ;;
        --coverage-timeout)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --coverage-timeout 需要数值 (秒)"
            fi
            COVERAGE_TIMEOUT="$2"
            shift 2
            ;;
        --split-coverage)
            SPLIT_COVERAGE=true
            shift
            ;;
        --pytest-args)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --pytest-args 需要参数"
            fi
            PYTEST_ARGS_STR="$2"
            shift 2
            ;;
        --pytest-arg)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --pytest-arg 需要参数"
            fi
            PYTEST_EXTRA_ARGS+=("$2")
            shift 2
            ;;
        --pytest-parallel)
            PYTEST_PARALLEL=true
            if [ -n "${2:-}" ] && [[ "${2:-}" =~ ^[0-9]+$ ]]; then
                PYTEST_PARALLEL_WORKERS="$2"
                shift 2
            else
                PYTEST_PARALLEL_WORKERS="auto"
                shift
            fi
            ;;
        --log-dir)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --log-dir 需要路径"
            fi
            LOG_DIR="$2"
            LOG_DIR_SET=true
            shift 2
            ;;
        --keep-logs)
            KEEP_LOGS=true
            shift
            ;;
        --diagnose-hang)
            DIAGNOSE_HANG=true
            DIAGNOSE_HANG_SET=true
            shift
            ;;
        --no-diagnose-hang)
            DIAGNOSE_HANG=false
            DIAGNOSE_HANG_SET=true
            shift
            ;;
        --include-network)
            INCLUDE_NETWORK=true
            shift
            ;;
        --include-cloud)
            INCLUDE_CLOUD=true
            shift
            ;;
        --all-markers)
            ALL_MARKERS=true
            INCLUDE_NETWORK=true
            INCLUDE_CLOUD=true
            shift
            ;;
        --run-network-isolation)
            RUN_NETWORK_ISOLATION=true
            shift
            ;;
        --cov-fail-under)
            if [ -z "${2:-}" ]; then
                _arg_error "参数错误: --cov-fail-under 需要数值 (0-100)"
            fi
            COV_FAIL_UNDER="$2"
            COV_FAIL_UNDER_SET=true
            shift 2
            ;;
        --help|-h)
            # JSON 模式下返回帮助信息的 JSON 格式
            if [ "$JSON_OUTPUT" = true ]; then
                _CHECK_ALL_COMPLETED=true
                cat << 'HELPJSON'
{
  "success": true,
  "exit_code": 0,
  "type": "help",
  "message": "使用 --help 查看帮助信息（非 JSON 模式）",
  "summary": {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0, "total": 0},
  "checks": []
}
HELPJSON
                exit 0
            fi
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --ci          CI 模式（非交互式、无颜色输出）"
            echo "  --json        JSON 输出格式（便于 CI 解析）"
            echo "  --fail-fast   遇到失败立即退出"
            echo "  -f, --full    执行完整检查（包括类型、风格和测试）"
            echo "  --mode MODE   运行特定检查模式，可多次指定（见下方模式说明）"
            echo "  --split-tests 按组运行测试（便于定位卡住/失败用例）"
            echo "  --test-chunk-size N  每组包含的测试文件数（默认 8）"
            echo "  --test-timeout N     单组测试超时秒数（默认 0=不限制）"
            echo "  --case-timeout N     单用例超时秒数（需 pytest-timeout）"
            echo "  --coverage-timeout N 覆盖率测试超时秒数（默认 1800）"
            echo "  --split-coverage     分组测试下启用覆盖率合并"
            echo "  --pytest-args \"...\" 透传 pytest 参数（简单场景）"
            echo "  --pytest-arg ARG     追加单个 pytest 参数（可多次）"
            echo "  --pytest-parallel [N] 启用 pytest-xdist 并行（默认 auto）"
            echo "  --log-dir PATH       指定测试日志目录"
            echo "  --keep-logs          保留测试日志（默认成功可清理）"
            echo "  --diagnose-hang      启用卡住定位诊断输出"
            echo "  --no-diagnose-hang   禁用卡住定位诊断输出"
            echo "  --include-network    包含 network 标记的测试（默认排除）"
            echo "  --include-cloud      包含 cloud 标记的测试（默认排除）"
            echo "  --all-markers        包含所有 marker 的测试（等同于 --include-network --include-cloud）"
            echo "  --run-network-isolation 运行网络隔离测试（与 CI no-api-key-smoke-test 对齐）"
            echo "  --cov-fail-under N      覆盖率阈值 (默认 80；CI 模式自动为 0)，0 表示仅收集不检查"
            echo "  -h, --help    显示此帮助信息"
            echo ""
            echo "运行模式 (--mode):"
            echo "  import        语法检查 + 核心模块导入（对齐 import-test.yml）"
            echo "  lint          flake8-critical + ruff + mypy（对齐 lint.yml）"
            echo "  test          单元测试 + E2E 测试（对齐 ci.yml test/e2e）"
            echo "  minimal       轻量检查（仅 Python 版本 + 基础验证，无额外依赖）"
            echo "  all           运行所有检查（默认，等同于不指定 --mode）"
            echo ""
            echo "环境变量:"
            echo "  CI=true       自动启用 CI 模式"
            echo "  NO_COLOR=1    禁用颜色输出"
            echo ""
            echo "示例:"
            echo "  $0                 快速检查"
            echo "  $0 --full          完整检查"
            echo "  $0 --mode import   仅运行导入检查（对齐 import-test.yml）"
            echo "  $0 --mode lint     仅运行 lint 检查（对齐 lint.yml）"
            echo "  $0 --mode test     仅运行测试（对齐 ci.yml）"
            echo "  $0 --mode minimal  轻量检查（无额外依赖，快速验证）"
            echo "  $0 --mode import --mode lint  运行导入和 lint 检查"
            echo "  $0 --full --split-tests --test-chunk-size 6"
            echo "  $0 --full --split-tests --test-timeout 600"
            echo "  $0 --full --coverage-timeout 900"
            echo "  $0 --full --split-tests --split-coverage"
            echo "  $0 --full --diagnose-hang"
            echo "  $0 --full --pytest-parallel 4"
            echo "  $0 --ci --json     CI 环境 JSON 输出"
            echo "  $0 --fail-fast     遇到失败立即退出"
            echo "  CI=true $0         自动 CI 模式"
            echo ""
            echo "Marker 控制示例:"
            echo "  $0 --full --include-network   包含 network 标记测试"
            echo "  $0 --full --include-cloud     包含 cloud 标记测试"
            echo "  $0 --full --all-markers       包含所有 marker 测试"
            echo "  $0 --full --run-network-isolation  运行网络隔离测试"
            _CHECK_ALL_COMPLETED=true
            exit 0
            ;;
        *)
            _arg_error "未知选项: $1 (使用 --help 查看帮助)"
            ;;
    esac
done

# CI 模式默认不做覆盖率阈值 gating（除非显式指定）
if [ "$CI_MODE" = true ] && [ "$COV_FAIL_UNDER_SET" = false ]; then
    COV_FAIL_UNDER=0
fi

# 分组测试默认超时（避免卡死）
if [ "$SPLIT_TESTS" = true ] && [ "$TEST_TIMEOUT" -le 0 ]; then
    TEST_TIMEOUT=600
fi

# 完整检查默认启用卡住定位诊断（可用 --no-diagnose-hang 关闭）
if [ "$FULL_CHECK" = true ] && [ "$DIAGNOSE_HANG_SET" = false ]; then
    DIAGNOSE_HANG=true
fi

# 动态构建 UNIT_MARKER_EXPR（根据 --include-* 参数调整）
# 基础表达式：排除 e2e、slow、integration
build_unit_marker_expr() {
    local expr="not e2e and not slow and not integration"
    
    if [ "$ALL_MARKERS" = true ]; then
        # --all-markers: 仅排除 e2e、slow、integration
        echo "$expr"
        return
    fi
    
    if [ "$INCLUDE_CLOUD" != true ]; then
        expr="$expr and not cloud"
    fi
    
    if [ "$INCLUDE_NETWORK" != true ]; then
        expr="$expr and not network"
    fi
    
    echo "$expr"
}

# 设置 UNIT_MARKER_EXPR（在参数解析后调用）
UNIT_MARKER_EXPR=$(build_unit_marker_expr)

# ============================================================
# 颜色定义 (CI 模式下禁用)
# ============================================================

if [ "$CI_MODE" = true ]; then
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
    BOLD=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
    BOLD='\033[1m'
fi

# ============================================================
# 计数器和结果收集
# ============================================================

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
SKIP_COUNT=0

# JSON 结果收集
declare -a JSON_CHECKS=()
declare -a SECTION_DURATIONS=()
CURRENT_SECTION=""
LOG_DIR_CREATED=false
PYTEST_ENV_READY=false
TIMEOUT_BACKEND=""
RUN_ID=""
PYTEST_COMMON_ARGS=()
LOG_DIR_NOTICE=false

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_ID="$(date '+%Y%m%d_%H%M%S' 2>/dev/null || date '+%Y%m%d_%H%M%S')"
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="${TMPDIR:-/tmp}/check_all_${RUN_ID}"
fi
cd "$PROJECT_ROOT"

# ============================================================
# 辅助函数
# ============================================================

# JSON 字符串转义（使用 Python json.dumps 确保正确处理所有控制字符）
json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1])[1:-1])' "$1"
}

# 添加 JSON 检查结果
add_json_check() {
    local status="$1"
    local name="$2"
    local message="$3"
    local log_file="${4:-}"
    local duration_ms="${5:-}"
    local meta_json="${6:-}"
    local section="${CURRENT_SECTION:-unknown}"

    # 转义 JSON 特殊字符
    message=$(json_escape "$message")
    name=$(json_escape "$name")
    section=$(json_escape "$section")

    local json="{\"section\":\"$section\",\"status\":\"$status\",\"name\":\"$name\",\"message\":\"$message\""
    if [ -n "$log_file" ]; then
        log_file=$(json_escape "$log_file")
        json="$json,\"log_file\":\"$log_file\""
    fi
    if [ -n "$duration_ms" ]; then
        json="$json,\"duration_ms\":$duration_ms"
    fi
    if [ -n "$meta_json" ]; then
        json="$json,$meta_json"
    fi
    json="$json}"

    JSON_CHECKS+=("$json")
}

print_header() {
    if [ "$JSON_OUTPUT" = true ]; then
        return
    fi
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_section() {
    CURRENT_SECTION="$1"
    if [ "$JSON_OUTPUT" = true ]; then
        return
    fi
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
}

check_pass() {
    local name="$1"
    local message="${2:-}"
    local log_file="${3:-}"
    local duration_ms="${4:-}"
    local meta_json="${5:-}"

    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "pass" "$name" "$message" "$log_file" "$duration_ms" "$meta_json"
    else
        echo -e "  ${GREEN}✓${NC} $name"
    fi
    ((++PASS_COUNT)) || true
}

check_fail() {
    local name="$1"
    local message="${2:-}"
    local log_file="${3:-}"
    local duration_ms="${4:-}"
    local meta_json="${5:-}"

    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "fail" "$name" "$message" "$log_file" "$duration_ms" "$meta_json"
    else
        echo -e "  ${RED}✗${NC} $name"
    fi
    ((++FAIL_COUNT)) || true

    # --fail-fast 模式下立即退出
    if [ "$FAIL_FAST" = true ]; then
        if [ "$JSON_OUTPUT" = true ]; then
            output_json_result 1
        else
            echo ""
            echo -e "${RED}${BOLD}[fail-fast] 检查失败，立即退出${NC}"
        fi
        exit 1
    fi
}

check_warn() {
    local name="$1"
    local message="${2:-}"
    local log_file="${3:-}"
    local duration_ms="${4:-}"
    local meta_json="${5:-}"

    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "warn" "$name" "$message" "$log_file" "$duration_ms" "$meta_json"
    else
        echo -e "  ${YELLOW}⚠${NC} $name"
    fi
    ((++WARN_COUNT)) || true
}

check_warn_or_info() {
    local name="$1"
    local message="${2:-}"
    local log_file="${3:-}"
    local duration_ms="${4:-}"
    local meta_json="${5:-}"

    if [ "$CI_MODE" = true ]; then
        check_info "$name" "$message" "$log_file" "$duration_ms" "$meta_json"
    else
        check_warn "$name" "$message" "$log_file" "$duration_ms" "$meta_json"
    fi
}

check_skip() {
    local name="$1"
    local message="${2:-}"
    local log_file="${3:-}"
    local duration_ms="${4:-}"
    local meta_json="${5:-}"

    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "skip" "$name" "$message" "$log_file" "$duration_ms" "$meta_json"
    else
        echo -e "  ${BLUE}○${NC} $name ${YELLOW}(跳过)${NC}"
    fi
    ((++SKIP_COUNT)) || true
}

check_info() {
    local name="$1"
    local message="${2:-}"
    local log_file="${3:-}"
    local duration_ms="${4:-}"
    local meta_json="${5:-}"

    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "info" "$name" "$message" "$log_file" "$duration_ms" "$meta_json"
    else
        echo -e "  ${BLUE}ℹ${NC} $name"
    fi
}

# 当前时间 (毫秒)
now_ms() {
    python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
}

# 格式化耗时
format_duration() {
    local ms="$1"
    if [ -z "$ms" ] || [ "$ms" -lt 0 ]; then
        echo "unknown"
        return
    fi
    if [ "$ms" -lt 1000 ]; then
        echo "${ms}ms"
    else
        awk "BEGIN {printf \"%.2fs\", $ms/1000}"
    fi
}

# 记录检查耗时
record_section_duration() {
    local name="$1"
    local duration_ms="$2"
    local escaped_name
    escaped_name=$(json_escape "$name")
    SECTION_DURATIONS+=("{\"name\":\"$escaped_name\",\"duration_ms\":$duration_ms}")
}

# 运行检查并统计耗时
run_check() {
    local section_name="$1"
    shift
    local start_ms
    start_ms=$(now_ms)
    set +e
    "$@"
    local status=$?
    set -e
    local end_ms
    end_ms=$(now_ms)
    local duration_ms=$((end_ms - start_ms))
    record_section_duration "$section_name" "$duration_ms"
    if [ "$JSON_OUTPUT" != true ]; then
        echo -e "  ${BLUE}ℹ${NC} 耗时: $(format_duration "$duration_ms")"
    fi
    return $status
}

# 检测超时实现
detect_timeout_backend() {
    if command -v timeout &> /dev/null; then
        echo "timeout"
        return
    fi
    if command -v gtimeout &> /dev/null; then
        echo "gtimeout"
        return
    fi
    echo "python"
}

# 确保日志目录存在
ensure_log_dir() {
    if [ "$LOG_DIR_CREATED" = true ]; then
        return
    fi
    if [ -z "$LOG_DIR" ]; then
        LOG_DIR="${TMPDIR:-/tmp}/check_all_${RUN_ID}"
    fi
    if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
        LOG_DIR="${TMPDIR:-/tmp}/check_all_${RUN_ID}"
        mkdir -p "$LOG_DIR" 2>/dev/null || true
    fi
    LOG_DIR_CREATED=true
}

# 生成 pytest 复现命令字符串
format_pytest_command() {
    local cmd="pytest"
    for arg in "$@"; do
        cmd+=" $(printf '%q' "$arg")"
    done
    echo "$cmd"
}

# 从日志中提取最后一个测试用例
extract_last_test_case() {
    local log_file="$1"
    python3 - <<'PY' "$log_file" 2>/dev/null
import re
import sys

path = sys.argv[1]
try:
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
except FileNotFoundError:
    sys.exit(0)

pattern = re.compile(r"(tests/[^\\s:]+::[^\\s]+)")
for line in reversed(lines):
    m = pattern.search(line)
    if m:
        print(m.group(1))
        break
PY
}

# 生成 JSON 额外字段
build_meta_json() {
    local last_test="$1"
    local command="$2"
    local parts=()
    if [ -n "$last_test" ]; then
        parts+=("\"last_test\":\"$(json_escape "$last_test")\"")
    fi
    if [ -n "$command" ]; then
        parts+=("\"command\":\"$(json_escape "$command")\"")
    fi
    if [ ${#parts[@]} -eq 0 ]; then
        echo ""
        return
    fi
    local IFS=,
    echo "${parts[*]}"
}

# 准备 pytest 参数（按需启用诊断/并行/超时）
prepare_pytest_env() {
    if [ "$PYTEST_ENV_READY" = true ]; then
        return
    fi
    ensure_log_dir
    if [ "$LOG_DIR_NOTICE" != true ] && [ "$JSON_OUTPUT" != true ]; then
        check_info "测试日志目录: $LOG_DIR"
        LOG_DIR_NOTICE=true
    fi

    if [ -n "$PYTEST_ARGS_STR" ]; then
        read -r -a EXTRA_FROM_STR <<< "$PYTEST_ARGS_STR"
        for arg in "${EXTRA_FROM_STR[@]}"; do
            PYTEST_EXTRA_ARGS+=("$arg")
        done
    fi

    PYTEST_COMMON_ARGS=()
    if [ "$DIAGNOSE_HANG" = true ]; then
        PYTEST_COMMON_ARGS+=("-vv" "--durations=25" "--durations-min=1")
    else
        PYTEST_COMMON_ARGS+=("-v")
    fi
    PYTEST_COMMON_ARGS+=("--tb=short")

    if [ "$FAIL_FAST" = true ]; then
        PYTEST_COMMON_ARGS+=("--maxfail=1")
    fi

    if [ "$PYTEST_PARALLEL" = true ]; then
        if python3 -c "import xdist" 2>/dev/null; then
            PYTEST_COMMON_ARGS+=("-n" "$PYTEST_PARALLEL_WORKERS")
        else
            check_warn "pytest-xdist 未安装，--pytest-parallel 无效"
        fi
    fi

    if [ "$CASE_TIMEOUT" -gt 0 ]; then
        if python3 -c "import pytest_timeout" 2>/dev/null; then
            PYTEST_COMMON_ARGS+=("--timeout=$CASE_TIMEOUT" "--timeout-method=thread")
        else
            check_warn "pytest-timeout 未安装，--case-timeout 无效"
        fi
    fi

    if [ ${#PYTEST_EXTRA_ARGS[@]} -gt 0 ]; then
        PYTEST_COMMON_ARGS+=("${PYTEST_EXTRA_ARGS[@]}")
    fi

    PYTEST_ENV_READY=true
}

# 输出 JSON 结果
output_json_result() {
    local exit_code="${1:-0}"
    local success="true"
    [ "$FAIL_COUNT" -gt 0 ] && success="false"

    # 构建检查数组
    local checks_json=""
    for check in "${JSON_CHECKS[@]}"; do
        if [ -n "$checks_json" ]; then
            checks_json="$checks_json,$check"
        else
            checks_json="$check"
        fi
    done

    # 构建耗时数组
    local durations_json=""
    for duration in "${SECTION_DURATIONS[@]}"; do
        if [ -n "$durations_json" ]; then
            durations_json="$durations_json,$duration"
        else
            durations_json="$duration"
        fi
    done

    cat << EOF
{
  "success": $success,
  "exit_code": $exit_code,
  "summary": {
    "passed": $PASS_COUNT,
    "failed": $FAIL_COUNT,
    "warnings": $WARN_COUNT,
    "skipped": $SKIP_COUNT,
    "total": $((PASS_COUNT + FAIL_COUNT + WARN_COUNT + SKIP_COUNT))
  },
  "ci_mode": $CI_MODE,
  "fail_fast": $FAIL_FAST,
  "full_check": $FULL_CHECK,
  "diagnose_hang": $DIAGNOSE_HANG,
  "timeout_backend": "$(json_escape "${TIMEOUT_BACKEND:-}")",
  "log_dir": "$(json_escape "${LOG_DIR:-}")",
  "project_root": "$PROJECT_ROOT",
  "timestamp": "$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')",
  "durations": [$durations_json],
  "checks": [$checks_json]
}
EOF
}

# 解析 pytest 输出中的统计数量
parse_pytest_count() {
    local output_file="$1"
    local label="$2"

    python3 - <<'PY' "$output_file" "$label" 2>/dev/null
import re
import sys

path = sys.argv[1]
label = sys.argv[2]
try:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
except FileNotFoundError:
    sys.exit(0)

summary_line = ""
for line in reversed(text.splitlines()):
    if "==" in line:
        summary_line = line
        break

if not summary_line:
    sys.exit(0)

match = re.search(r"(\\d+)\\s+" + re.escape(label), summary_line)
if match:
    sys.stdout.write(match.group(1))
PY
}

# 从混合输出中提取 JSON（取最后一个可解析对象）
extract_json_from_output() {
    python3 -c 'import json,sys
text = sys.stdin.read()
decoder = json.JSONDecoder()
last_obj = None
idx = 0
text_len = len(text)
while idx < text_len:
    try:
        obj, end = decoder.raw_decode(text, idx)
    except json.JSONDecodeError:
        idx += 1
        continue
    if isinstance(obj, dict):
        last_obj = obj
    idx = end
if last_obj is None:
    sys.exit(1)
print(json.dumps(last_obj, ensure_ascii=False))'
}

# 运行命令并支持超时与输出捕获
# macOS 兼容：优先使用 gtimeout (brew install coreutils)，否则用 Python 实现
run_command_with_timeout() {
    local timeout_seconds="$1"
    local output_file="$2"
    local stream_output="$3"
    shift 3

    if [ -z "$timeout_seconds" ] || [ "$timeout_seconds" -le 0 ]; then
        if [ "$stream_output" = true ]; then
            "$@" 2>&1 | tee "$output_file"
            return ${PIPESTATUS[0]}
        fi
        "$@" > "$output_file" 2>&1
        return $?
    fi

    # 优先使用 GNU timeout（Linux 或 brew install coreutils）
    if command -v timeout &> /dev/null; then
        if [ "$stream_output" = true ]; then
            timeout "$timeout_seconds" "$@" 2>&1 | tee "$output_file"
            return ${PIPESTATUS[0]}
        fi
        timeout "$timeout_seconds" "$@" > "$output_file" 2>&1
        return $?
    fi

    # macOS: 尝试 gtimeout (brew install coreutils)
    if command -v gtimeout &> /dev/null; then
        if [ "$stream_output" = true ]; then
            gtimeout "$timeout_seconds" "$@" 2>&1 | tee "$output_file"
            return ${PIPESTATUS[0]}
        fi
        gtimeout "$timeout_seconds" "$@" > "$output_file" 2>&1
        return $?
    fi

    # 回退: 使用 Python 实现（非阻塞，使用 select/poll）
    python3 - "$timeout_seconds" "$output_file" "$stream_output" "$@" <<'PY'
import os
import select
import subprocess
import sys
import time

timeout_seconds = int(sys.argv[1])
output_file = sys.argv[2]
stream_output = sys.argv[3].lower() == "true"
cmd = sys.argv[4:]

with open(output_file, "w", encoding="utf-8", errors="ignore") as handle:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    start = time.time()
    timed_out = False
    fd = proc.stdout.fileno()

    # 设置非阻塞读取
    import fcntl
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    while True:
        remaining = timeout_seconds - (time.time() - start)
        if remaining <= 0:
            timed_out = True
            break

        # 使用 select 等待数据，最多等 1 秒
        ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
        if ready:
            try:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break  # EOF
                text = chunk.decode("utf-8", errors="ignore")
                handle.write(text)
                handle.flush()
                if stream_output:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            except BlockingIOError:
                pass
        else:
            # 检查进程是否已结束
            if proc.poll() is not None:
                # 读取剩余输出
                try:
                    remaining_data = proc.stdout.read()
                    if remaining_data:
                        text = remaining_data.decode("utf-8", errors="ignore")
                        handle.write(text)
                        if stream_output:
                            sys.stdout.write(text)
                except Exception:
                    pass
                break

    if timed_out:
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass
        sys.exit(124)

    exit_code = proc.wait()
    sys.exit(exit_code)
PY
}

# 获取 Python 文件列表（优先使用 git）
collect_python_files() {
    if command -v git &> /dev/null && [ -d ".git" ]; then
        # 仅返回当前存在的文件（避免 git index 中已删除文件导致误报）
        git ls-files "*.py" | while IFS= read -r file; do
            [ -f "$file" ] && echo "$file"
        done
        return 0
    fi

    python3 - <<'PY'
import os

exclude_dirs = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "node_modules",
    ".cursor",
}

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for name in files:
        if name.endswith(".py"):
            path = os.path.join(root, name)
            if path.startswith("./"):
                path = path[2:]
            print(path)
PY
}

# 构建测试分组（每行一组文件）
build_test_groups() {
    local exclude_csv="$1"
    EXCLUDE_TESTS="$exclude_csv" TEST_CHUNK_SIZE="$TEST_CHUNK_SIZE" python3 - <<'PY'
import glob
import os

exclude = set(filter(None, os.environ.get("EXCLUDE_TESTS", "").split(",")))
try:
    chunk_size = int(os.environ.get("TEST_CHUNK_SIZE", "8"))
except ValueError:
    chunk_size = 8
chunk_size = max(chunk_size, 1)

files = [f for f in sorted(glob.glob("tests/**/test_*.py", recursive=True)) if f not in exclude]

for i in range(0, len(files), chunk_size):
    group = files[i:i + chunk_size]
    if group:
        print(" ".join(group))
PY
}

# 运行 pytest 并保存输出
run_pytest_group() {
    local output_file="$1"
    local timeout_seconds="$2"
    shift 2
    local exit_code=0
    local stream_output=true
    local errexit_on=0

    if [ "$JSON_OUTPUT" = true ]; then
        stream_output=false
    fi

    case $- in
        *e*) errexit_on=1 ;;
    esac

    set +e
    if [ -z "$timeout_seconds" ] || [ "$timeout_seconds" -lt 0 ]; then
        timeout_seconds="$TEST_TIMEOUT"
    fi
    run_command_with_timeout "$timeout_seconds" "$output_file" "$stream_output" pytest "$@"
    exit_code=$?
    if [ "$errexit_on" -eq 1 ]; then
        set -e
    else
        set +e
    fi

    return $exit_code
}

# ============================================================
# 检查函数
# ============================================================

# 核心测试文件列表（用于拆分测试时排除）
CORE_TEST_FILES=(
    "tests/test_run.py"
    "tests/test_self_iterate.py"
    "tests/test_e2e_execution_modes.py"
    "tests/test_orchestrator_mp_commit.py"
    "tests/test_project_workspace.py"
)

# 测试分层（与 CI 的 markers 定义保持一致）
# UNIT_MARKER_EXPR 动态构建（见 build_unit_marker_expr 函数，第 266 行）
# 此处不重新赋值，保持 build_unit_marker_expr 的结果
# E2E 测试：仅运行 e2e 标记（不排除 cloud/network，由 CI 的 e2e-test job 决定）
E2E_MARKER_EXPR="e2e and not slow"

# 网络隔离测试文件（与 CI no-api-key-smoke-test job 对齐）
NETWORK_ISOLATION_TEST_FILES=(
    "tests/test_network_blocking.py"
    "tests/test_no_api_key_network_isolation.py"
)

check_python_version() {
    print_section "Python 环境检查"

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            check_pass "Python 版本: $PYTHON_VERSION (>= 3.9)"
        else
            check_warn "Python 版本: $PYTHON_VERSION (推荐 >= 3.9)"
        fi
    else
        check_fail "Python3 未安装"
        return 1
    fi

    # 检查 pip
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version 2>/dev/null | cut -d' ' -f2)
        check_pass "pip 版本: $PIP_VERSION"
    else
        check_warn "pip3 命令不可用"
    fi
}

check_timeout_backend() {
    print_section "超时机制检查"

    TIMEOUT_BACKEND=$(detect_timeout_backend)
    if [ "$TIMEOUT_BACKEND" = "timeout" ]; then
        check_pass "timeout 可用 (Linux/GNU coreutils)"
    elif [ "$TIMEOUT_BACKEND" = "gtimeout" ]; then
        check_pass "gtimeout 可用 (macOS coreutils)"
        check_info "可选安装: brew install coreutils"
    else
        check_info "未检测到 timeout/gtimeout，使用 Python 回退实现"
    fi
}

check_dependencies() {
    print_section "依赖检查"

    # 检查分层依赖文件
    # - requirements.txt: 核心依赖（CI 默认）
    # - requirements-dev.txt: 开发依赖（本地开发）
    # - requirements-test.txt: 测试依赖（CI 测试阶段）
    REQ_FILES_FOUND=0
    
    if [ -f "requirements.txt" ]; then
        check_pass "requirements.txt 存在（核心依赖）"
        ((REQ_FILES_FOUND++))
    else
        check_fail "requirements.txt 不存在"
        return 1
    fi
    
    if [ -f "requirements-dev.txt" ]; then
        check_pass "requirements-dev.txt 存在（开发依赖）"
        ((REQ_FILES_FOUND++))
    else
        check_warn "requirements-dev.txt 不存在"
    fi
    
    if [ -f "requirements-test.txt" ]; then
        check_pass "requirements-test.txt 存在（测试依赖）"
        ((REQ_FILES_FOUND++))
    else
        check_warn "requirements-test.txt 不存在"
    fi
    
    check_info "找到 ${REQ_FILES_FOUND}/3 个依赖文件"

    # 使用 Python 直接导入检查（比 pip list 更快）
    DEPS_MISSING=0

    # 核心依赖 (包名:导入名)
    # 必须与 requirements.in 保持同步
    CORE_DEPS=(
        "pydantic:pydantic"
        "loguru:loguru"
        "pyyaml:yaml"
        "aiofiles:aiofiles"
        "httpx:httpx"
        "websockets:websockets"
    )

    for dep in "${CORE_DEPS[@]}"; do
        pkg="${dep%%:*}"
        import_name="${dep##*:}"
        if python3 -c "import $import_name" 2>/dev/null; then
            check_pass "已安装: $pkg"
        else
            check_fail "未安装: $pkg"
            ((DEPS_MISSING++))
        fi
    done

    # 检查开发依赖（来自 requirements-dev.txt）
    DEV_DEPS=(
        "mypy:mypy"
        "ruff:ruff"
        "flake8:flake8"
    )

    for dep in "${DEV_DEPS[@]}"; do
        pkg="${dep%%:*}"
        import_name="${dep##*:}"
        if python3 -c "import $import_name" 2>/dev/null; then
            check_pass "已安装 (开发): $pkg"
        else
            check_warn "未安装 (开发): $pkg"
        fi
    done

    # 检查测试依赖（来自 requirements-test.txt）
    TEST_DEPS=(
        "pytest:pytest"
        "pytest-asyncio:pytest_asyncio"
        "pytest-cov:pytest_cov"
    )

    for dep in "${TEST_DEPS[@]}"; do
        pkg="${dep%%:*}"
        import_name="${dep##*:}"
        if python3 -c "import $import_name" 2>/dev/null; then
            check_pass "已安装 (测试): $pkg"
        else
            check_warn "未安装 (测试): $pkg"
        fi
    done

    # 检查可选依赖 - Web 处理（来自 pyproject.toml [web]）
    WEB_DEPS=(
        "beautifulsoup4:bs4"
        "html2text:html2text"
        "lxml:lxml"
    )

    for dep in "${WEB_DEPS[@]}"; do
        pkg="${dep%%:*}"
        import_name="${dep##*:}"
        if python3 -c "import $import_name" 2>/dev/null; then
            check_pass "已安装 (可选/Web): $pkg"
        else
            check_info "未安装 (可选/Web): $pkg"
        fi
    done

    # 检查可选依赖 - ML/向量功能（来自 pyproject.toml [ml]）
    ML_DEPS=(
        "sentence-transformers:sentence_transformers"
        "chromadb:chromadb"
    )

    for dep in "${ML_DEPS[@]}"; do
        pkg="${dep%%:*}"
        import_name="${dep##*:}"
        if python3 -c "import $import_name" 2>/dev/null; then
            check_pass "已安装 (可选/ML): $pkg"
        else
            check_info "未安装 (可选/ML): $pkg"
        fi
    done

    if [ $DEPS_MISSING -gt 0 ]; then
        check_info "运行 'pip-sync requirements.txt requirements-dev.txt requirements-test.txt' 安装依赖"
        check_info "或: pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt"
    fi
}

check_syntax() {
    print_section "Python 语法检查"

    SYNTAX_ERRORS=0
    PY_FILES=$(collect_python_files)

    if [ -z "$PY_FILES" ]; then
        check_warn "未找到 Python 文件"
        return 0
    fi

    local file_count=0
    local old_ifs="$IFS"
    IFS=$'\n'
    for file in $PY_FILES; do
        [ -z "$file" ] && continue
        file_count=$((file_count + 1))
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            check_fail "语法错误: $file"
            ((SYNTAX_ERRORS++))
        fi
    done
    IFS="$old_ifs"

    if [ $SYNTAX_ERRORS -eq 0 ]; then
        check_pass "语法检查通过 ($file_count 个文件)"
    else
        check_fail "发现 $SYNTAX_ERRORS 个语法错误"
    fi
}

check_imports() {
    print_section "模块结构检查"

    MODULES=("core" "agents" "coordinator" "tasks" "cursor" "indexing" "knowledge" "process")
    IMPORT_ERRORS=0

    for module in "${MODULES[@]}"; do
        if [ -d "$module" ]; then
            # 检查是否有 __init__.py
            if [ -f "$module/__init__.py" ]; then
                check_pass "模块结构正确: $module/"
            else
                check_warn "模块缺少 __init__.py: $module/"
                ((IMPORT_ERRORS++))
            fi
        else
            check_skip "模块不存在: $module"
        fi
    done

    return $IMPORT_ERRORS
}

check_type_hints() {
    print_section "类型检查 (mypy)"

    if command -v mypy &> /dev/null; then
        # 只检查核心模块
        # JSON 模式下将输出重定向到日志文件，避免污染 JSON 输出
        ensure_log_dir
        local mypy_log="$LOG_DIR/mypy.log"
        if [ "$JSON_OUTPUT" = true ]; then
            if mypy core/ agents/ --ignore-missing-imports --no-error-summary > "$mypy_log" 2>&1; then
                check_pass "类型检查通过" "" "$mypy_log"
            else
                check_warn_or_info "类型检查有警告 (非致命)" "" "$mypy_log"
            fi
        else
            if mypy core/ agents/ --ignore-missing-imports --no-error-summary > "$mypy_log" 2>&1; then
                check_pass "类型检查通过"
            else
                check_warn_or_info "类型检查有警告 (非致命)" "" "$mypy_log"
                if [ -s "$mypy_log" ]; then
                    tail -n 20 "$mypy_log" | sed 's/^/    /'
                    check_info "日志: $mypy_log"
                fi
            fi
        fi
    else
        check_skip "mypy 未安装 (pip install mypy)"
    fi
}

check_code_style() {
    print_section "代码风格检查"

    # flake8 检查
    if command -v flake8 &> /dev/null; then
        FLAKE8_ERRORS=$(flake8 --count --select=E9,F63,F7,F82 --show-source --statistics . 2>&1 | tail -1)
        if [ "$FLAKE8_ERRORS" = "0" ] || [ -z "$FLAKE8_ERRORS" ]; then
            check_pass "flake8 严重错误检查通过"
        else
            check_fail "flake8 发现 $FLAKE8_ERRORS 个严重错误"
        fi

        # 代码风格警告（不计入失败）
        STYLE_WARNS=$(flake8 --count --exit-zero --max-complexity=10 --max-line-length=120 . 2>&1 | tail -1)
        if [ "$STYLE_WARNS" = "0" ] || [ -z "$STYLE_WARNS" ]; then
            check_pass "flake8 代码风格检查通过"
        else
            check_warn "flake8 代码风格警告: $STYLE_WARNS 个"
        fi
    else
        check_skip "flake8 未安装 (pip install flake8)"
    fi

    # ruff 检查（更快的替代方案）
    if command -v ruff &> /dev/null; then
        RUFF_ERRORS=$(ruff check . --quiet 2>&1 | wc -l)
        if [ "$RUFF_ERRORS" -eq 0 ]; then
            check_pass "ruff 检查通过"
        else
            check_warn_or_info "ruff 发现 $RUFF_ERRORS 个问题"
        fi
    else
        check_info "ruff 未安装 (pip install ruff) - 推荐使用"
    fi
}

check_tests() {
    print_section "测试检查"

    if [ -d "tests" ]; then
        TEST_COUNT=$(find tests -name "test_*.py" | wc -l)
        check_pass "测试目录存在 ($TEST_COUNT 个测试文件)"

        if command -v pytest &> /dev/null; then
            prepare_pytest_env
            if [ "$JSON_OUTPUT" != true ]; then
                echo -e "  ${BLUE}ℹ${NC} 运行测试..."
            fi

            if [ "$SPLIT_TESTS" = true ]; then
                check_info "按组运行测试 (每组 $TEST_CHUNK_SIZE 个文件)"

                EXCLUDE_TESTS=$(IFS=,; echo "${CORE_TEST_FILES[*]}")
                GROUP_INDEX=0
                TOTAL_GROUPS=0
                while IFS= read -r group; do
                    [ -z "$group" ] && continue
                    TOTAL_GROUPS=$((TOTAL_GROUPS + 1))
                done < <(build_test_groups "$EXCLUDE_TESTS")

                while IFS= read -r group; do
                    [ -z "$group" ] && continue
                    GROUP_INDEX=$((GROUP_INDEX + 1))
                    OUTPUT_FILE="$LOG_DIR/pytest_group_${GROUP_INDEX}.log"

                    if [ "$JSON_OUTPUT" != true ]; then
                        echo -e "  ${BLUE}ℹ${NC} 测试组 ${GROUP_INDEX}/${TOTAL_GROUPS}: $group"
                    fi

                    local start_ms
                    start_ms=$(now_ms)
                    if run_pytest_group "$OUTPUT_FILE" "$TEST_TIMEOUT" "${PYTEST_COMMON_ARGS[@]}" $group; then
                        local duration_ms=$(( $(now_ms) - start_ms ))
                        PASSED=$(parse_pytest_count "$OUTPUT_FILE" "passed")
                        check_pass "测试组 ${GROUP_INDEX} 通过: ${PASSED:-0} 个" "" "$OUTPUT_FILE" "$duration_ms"
                    else
                        group_exit_code=$?
                        local duration_ms=$(( $(now_ms) - start_ms ))
                        last_test=$(extract_last_test_case "$OUTPUT_FILE")
                        cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" $group)
                        meta_json=$(build_meta_json "$last_test" "$cmd_str")

                        if [ "$group_exit_code" -eq 124 ]; then
                            if [ "$JSON_OUTPUT" != true ]; then
                                echo -e "  ${YELLOW}⚠${NC} 测试组 ${GROUP_INDEX} 超时，输出摘要:"
                                tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                                [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                                check_info "日志: $OUTPUT_FILE"
                                check_info "复现: $cmd_str"
                                check_info "建议: 使用 --test-chunk-size 1 缩小范围"
                            fi
                            check_fail "测试组 ${GROUP_INDEX} 超时" "timeout" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                        else
                            FAILED=$(parse_pytest_count "$OUTPUT_FILE" "failed")
                            if [ "$JSON_OUTPUT" != true ]; then
                                echo -e "  ${YELLOW}⚠${NC} 测试组 ${GROUP_INDEX} 失败，输出摘要:"
                                tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                                [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                                check_info "日志: $OUTPUT_FILE"
                                check_info "复现: $cmd_str"
                            fi
                            check_fail "测试组 ${GROUP_INDEX} 失败: ${FAILED:-?} 个" "failed" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                        fi
                    fi
                done < <(build_test_groups "$EXCLUDE_TESTS")
            else
                OUTPUT_FILE="$LOG_DIR/pytest_all.log"
                local start_ms
                start_ms=$(now_ms)
                if run_pytest_group "$OUTPUT_FILE" "$TEST_TIMEOUT" "${PYTEST_COMMON_ARGS[@]}" tests/; then
                    local duration_ms=$(( $(now_ms) - start_ms ))
                    PASSED=$(parse_pytest_count "$OUTPUT_FILE" "passed")
                    check_pass "测试通过: ${PASSED:-0} 个" "" "$OUTPUT_FILE" "$duration_ms"
                else
                    group_exit_code=$?
                    local duration_ms=$(( $(now_ms) - start_ms ))
                    last_test=$(extract_last_test_case "$OUTPUT_FILE")
                    cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" tests/)
                    meta_json=$(build_meta_json "$last_test" "$cmd_str")

                    if [ "$group_exit_code" -eq 124 ]; then
                        if [ "$JSON_OUTPUT" != true ]; then
                            echo -e "  ${YELLOW}⚠${NC} 超时前输出:"
                            tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                            check_info "日志: $OUTPUT_FILE"
                            check_info "复现: $cmd_str"
                        fi
                        check_fail "测试超时" "timeout" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                    else
                        FAILED=$(parse_pytest_count "$OUTPUT_FILE" "failed")
                        if [ "$JSON_OUTPUT" != true ]; then
                            echo -e "  ${YELLOW}⚠${NC} 失败输出摘要:"
                            tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                            check_info "日志: $OUTPUT_FILE"
                            check_info "复现: $cmd_str"
                        fi
                        check_fail "测试失败: ${FAILED:-?} 个" "failed" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                    fi
                fi
            fi
        else
            check_skip "pytest 未安装 (pip install pytest)"
        fi
    else
        check_warn "tests 目录不存在"
    fi
}

check_core_tests() {
    # 核心测试集合：run.py、self_iterate、execution_modes、orchestrator_mp_commit、project_workspace
    # 这些测试确保 CI 稳定性
    print_section "核心测试集合"

    if ! command -v pytest &> /dev/null; then
        check_skip "pytest 未安装 (pip install pytest)"
        return 0
    fi
    prepare_pytest_env

    MISSING_FILES=0
    for test_file in "${CORE_TEST_FILES[@]}"; do
        if [ ! -f "$test_file" ]; then
            check_fail "核心测试文件缺失: $test_file"
            ((MISSING_FILES++))
        fi
    done

    if [ $MISSING_FILES -gt 0 ]; then
        check_fail "有 $MISSING_FILES 个核心测试文件缺失"
        return 1
    fi

    check_pass "所有核心测试文件存在 (${#CORE_TEST_FILES[@]} 个)"

    if [ "$JSON_OUTPUT" != true ]; then
        echo -e "  ${BLUE}ℹ${NC} 运行核心测试集合 (使用 unit marker，跳过慢速测试)..."
    fi

    # 运行核心测试（跳过 slow 和 e2e 标记的测试以保持快速）
    # 使用超时保护，默认 300 秒
    CORE_TEST_TIMEOUT="$TEST_TIMEOUT"
    if [ "$CORE_TEST_TIMEOUT" -le 0 ]; then
        CORE_TEST_TIMEOUT=300
    fi
    CORE_OUTPUT_FILE="$LOG_DIR/pytest_core.log"

    local start_ms
    start_ms=$(now_ms)
    set +e
    run_pytest_group "$CORE_OUTPUT_FILE" "$CORE_TEST_TIMEOUT" "${PYTEST_COMMON_ARGS[@]}" \
        "${CORE_TEST_FILES[@]}" -m 'not slow and not e2e'
    CORE_EXIT=$?
    set -e
    local duration_ms=$(( $(now_ms) - start_ms ))

    if [ "$CORE_EXIT" -eq 0 ]; then
        PASSED=$(parse_pytest_count "$CORE_OUTPUT_FILE" "passed")
        check_pass "核心测试通过: ${PASSED:-0} 个" "" "$CORE_OUTPUT_FILE" "$duration_ms"
    elif [ "$CORE_EXIT" -eq 124 ]; then
        last_test=$(extract_last_test_case "$CORE_OUTPUT_FILE")
        cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" "${CORE_TEST_FILES[@]}" -m 'not slow and not e2e')
        meta_json=$(build_meta_json "$last_test" "$cmd_str")
        check_fail "核心测试超时 (${CORE_TEST_TIMEOUT}s)" "timeout" "$CORE_OUTPUT_FILE" "$duration_ms" "$meta_json"
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${YELLOW}⚠${NC} 超时前输出:"
            tail -n 30 "$CORE_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
            check_info "日志: $CORE_OUTPUT_FILE"
            check_info "复现: $cmd_str"
        fi
    else
        FAILED=$(parse_pytest_count "$CORE_OUTPUT_FILE" "failed")
        last_test=$(extract_last_test_case "$CORE_OUTPUT_FILE")
        cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" "${CORE_TEST_FILES[@]}" -m 'not slow and not e2e')
        meta_json=$(build_meta_json "$last_test" "$cmd_str")
        check_fail "核心测试失败: ${FAILED:-?} 个" "failed" "$CORE_OUTPUT_FILE" "$duration_ms" "$meta_json"
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${YELLOW}⚠${NC} 失败输出摘要:"
            tail -n 30 "$CORE_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
            check_info "日志: $CORE_OUTPUT_FILE"
            check_info "复现: $cmd_str"
        fi
    fi
}

check_unit_tests() {
    print_section "单元测试"

    if [ -d "tests" ]; then
        TEST_COUNT=$(find tests -name "test_*.py" | wc -l)
        check_pass "测试目录存在 ($TEST_COUNT 个测试文件)"

        if ! command -v pytest &> /dev/null; then
            check_skip "pytest 未安装 (pip install pytest)"
            return 0
        fi

        prepare_pytest_env

        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 运行单元测试 (-m \"$UNIT_MARKER_EXPR\")..."
        fi

        if [ "$SPLIT_TESTS" = true ]; then
            check_info "按组运行单元测试 (每组 $TEST_CHUNK_SIZE 个文件)"

            GROUP_INDEX=0
            TOTAL_GROUPS=0
            while IFS= read -r group; do
                [ -z "$group" ] && continue
                TOTAL_GROUPS=$((TOTAL_GROUPS + 1))
            done < <(build_test_groups "")

            while IFS= read -r group; do
                [ -z "$group" ] && continue
                GROUP_INDEX=$((GROUP_INDEX + 1))
                OUTPUT_FILE="$LOG_DIR/pytest_unit_group_${GROUP_INDEX}.log"

                if [ "$JSON_OUTPUT" != true ]; then
                    echo -e "  ${BLUE}ℹ${NC} 单元测试组 ${GROUP_INDEX}/${TOTAL_GROUPS}: $group"
                fi

                local start_ms
                start_ms=$(now_ms)
                if run_pytest_group "$OUTPUT_FILE" "$TEST_TIMEOUT" "${PYTEST_COMMON_ARGS[@]}" -m "$UNIT_MARKER_EXPR" $group; then
                    local duration_ms=$(( $(now_ms) - start_ms ))
                    PASSED=$(parse_pytest_count "$OUTPUT_FILE" "passed")
                    check_pass "单元测试组 ${GROUP_INDEX} 通过: ${PASSED:-0} 个" "" "$OUTPUT_FILE" "$duration_ms"
                else
                    group_exit_code=$?
                    local duration_ms=$(( $(now_ms) - start_ms ))
                    last_test=$(extract_last_test_case "$OUTPUT_FILE")
                    cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" -m "$UNIT_MARKER_EXPR" $group)
                    meta_json=$(build_meta_json "$last_test" "$cmd_str")

                    if [ "$group_exit_code" -eq 124 ]; then
                        if [ "$JSON_OUTPUT" != true ]; then
                            echo -e "  ${YELLOW}⚠${NC} 单元测试组 ${GROUP_INDEX} 超时，输出摘要:"
                            tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                            check_info "日志: $OUTPUT_FILE"
                            check_info "复现: $cmd_str"
                            check_info "建议: 使用 --test-chunk-size 1 缩小范围"
                        fi
                        check_fail "单元测试组 ${GROUP_INDEX} 超时" "timeout" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                    else
                        FAILED=$(parse_pytest_count "$OUTPUT_FILE" "failed")
                        if [ "$JSON_OUTPUT" != true ]; then
                            echo -e "  ${YELLOW}⚠${NC} 单元测试组 ${GROUP_INDEX} 失败，输出摘要:"
                            tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                            check_info "日志: $OUTPUT_FILE"
                            check_info "复现: $cmd_str"
                        fi
                        check_fail "单元测试组 ${GROUP_INDEX} 失败: ${FAILED:-?} 个" "failed" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                    fi
                fi
            done < <(build_test_groups "")
        else
            OUTPUT_FILE="$LOG_DIR/pytest_unit_all.log"
            local start_ms
            start_ms=$(now_ms)
            if run_pytest_group "$OUTPUT_FILE" "$TEST_TIMEOUT" "${PYTEST_COMMON_ARGS[@]}" -m "$UNIT_MARKER_EXPR" tests/; then
                local duration_ms=$(( $(now_ms) - start_ms ))
                PASSED=$(parse_pytest_count "$OUTPUT_FILE" "passed")
                check_pass "单元测试通过: ${PASSED:-0} 个" "" "$OUTPUT_FILE" "$duration_ms"
            else
                group_exit_code=$?
                local duration_ms=$(( $(now_ms) - start_ms ))
                last_test=$(extract_last_test_case "$OUTPUT_FILE")
                cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" -m "$UNIT_MARKER_EXPR" tests/)
                meta_json=$(build_meta_json "$last_test" "$cmd_str")

                if [ "$group_exit_code" -eq 124 ]; then
                    if [ "$JSON_OUTPUT" != true ]; then
                        echo -e "  ${YELLOW}⚠${NC} 单元测试超时，输出摘要:"
                        tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                        [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                        check_info "日志: $OUTPUT_FILE"
                        check_info "复现: $cmd_str"
                    fi
                    check_fail "单元测试超时" "timeout" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                else
                    FAILED=$(parse_pytest_count "$OUTPUT_FILE" "failed")
                    if [ "$JSON_OUTPUT" != true ]; then
                        echo -e "  ${YELLOW}⚠${NC} 单元测试失败，输出摘要:"
                        tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                        [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                        check_info "日志: $OUTPUT_FILE"
                        check_info "复现: $cmd_str"
                    fi
                    check_fail "单元测试失败: ${FAILED:-?} 个" "failed" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                fi
            fi
        fi
    else
        check_warn "tests 目录不存在"
    fi
}

check_unit_tests_with_coverage() {
    print_section "单元测试 + 覆盖率"

    if ! command -v pytest &> /dev/null; then
        check_skip "pytest 未安装 (pip install pytest)"
        return 0
    fi

    # 检查 pytest-cov 是否安装
    if ! python3 -c "import pytest_cov" 2>/dev/null; then
        check_skip "pytest-cov 未安装 (pip install pytest-cov)"
        return 0
    fi
    prepare_pytest_env

    if [ "$SPLIT_TESTS" = true ] && [ "$SPLIT_COVERAGE" != true ]; then
        check_warn "--split-tests 模式下未启用覆盖率合并（使用 --split-coverage 开启）"
        return 0
    fi

    if [ "$JSON_OUTPUT" != true ]; then
        echo -e "  ${BLUE}ℹ${NC} 运行单元测试（含覆盖率）..."
    fi
    COV_OUTPUT_FILE=""
    duration_ms=""
    COV_META_JSON=""
    COV_CMD_STR=""
    COV_LAST_TEST=""

    if [ "$SPLIT_TESTS" = true ] && [ "$SPLIT_COVERAGE" = true ]; then
        # 分组运行覆盖率（覆盖率合并）
        check_info "按组运行单元测试 + 覆盖率 (每组 $TEST_CHUNK_SIZE 个文件)"
        rm -f .coverage 2>/dev/null || true

        GROUP_INDEX=0
        TOTAL_GROUPS=0
        COV_GROUP_FAILED=false
        while IFS= read -r group; do
            [ -z "$group" ] && continue
            TOTAL_GROUPS=$((TOTAL_GROUPS + 1))
        done < <(build_test_groups "")

        local group_timeout="$TEST_TIMEOUT"
        if [ "$group_timeout" -le 0 ]; then
            group_timeout=600
        fi

        while IFS= read -r group; do
            [ -z "$group" ] && continue
            GROUP_INDEX=$((GROUP_INDEX + 1))
            OUTPUT_FILE="$LOG_DIR/pytest_cov_group_${GROUP_INDEX}.log"

            if [ "$JSON_OUTPUT" != true ]; then
                echo -e "  ${BLUE}ℹ${NC} 覆盖率组 ${GROUP_INDEX}/${TOTAL_GROUPS}: $group"
            fi

            local cov_args=(
                --cov=run --cov=core --cov=agents --cov=coordinator --cov=cursor --cov=tasks
                --cov=knowledge --cov=indexing --cov=process --cov-report=term-missing
            )
            if [ "$GROUP_INDEX" -gt 1 ]; then
                cov_args+=(--cov-append)
            fi

            local start_ms
            start_ms=$(now_ms)
            if run_pytest_group "$OUTPUT_FILE" "$group_timeout" "${PYTEST_COMMON_ARGS[@]}" -m "$UNIT_MARKER_EXPR" "${cov_args[@]}" $group; then
                local duration_ms=$(( $(now_ms) - start_ms ))
                PASSED=$(parse_pytest_count "$OUTPUT_FILE" "passed")
                check_pass "覆盖率组 ${GROUP_INDEX} 通过: ${PASSED:-0} 个" "" "$OUTPUT_FILE" "$duration_ms"
            else
                group_exit_code=$?
                local duration_ms=$(( $(now_ms) - start_ms ))
                last_test=$(extract_last_test_case "$OUTPUT_FILE")
                cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" -m "$UNIT_MARKER_EXPR" "${cov_args[@]}" $group)
                meta_json=$(build_meta_json "$last_test" "$cmd_str")

                if [ "$group_exit_code" -eq 124 ]; then
                    if [ "$JSON_OUTPUT" != true ]; then
                        echo -e "  ${YELLOW}⚠${NC} 覆盖率组 ${GROUP_INDEX} 超时，输出摘要:"
                        tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                        [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                        check_info "日志: $OUTPUT_FILE"
                        check_info "复现: $cmd_str"
                        check_info "建议: 使用 --test-chunk-size 1 缩小范围"
                    fi
                    check_fail "覆盖率组 ${GROUP_INDEX} 超时" "timeout" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                    COV_GROUP_FAILED=true
                else
                    FAILED=$(parse_pytest_count "$OUTPUT_FILE" "failed")
                    if [ "$JSON_OUTPUT" != true ]; then
                        echo -e "  ${YELLOW}⚠${NC} 覆盖率组 ${GROUP_INDEX} 失败，输出摘要:"
                        tail -n 20 "$OUTPUT_FILE" | sed 's/^/    /'
                        [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
                        check_info "日志: $OUTPUT_FILE"
                        check_info "复现: $cmd_str"
                    fi
                    check_fail "覆盖率组 ${GROUP_INDEX} 失败: ${FAILED:-?} 个" "failed" "$OUTPUT_FILE" "$duration_ms" "$meta_json"
                    COV_GROUP_FAILED=true
                fi
            fi
        done < <(build_test_groups "")

        # 汇总覆盖率（基于 .coverage）
        COV_REPORT_FILE="$LOG_DIR/coverage_report.log"
        if python3 -c "import coverage" 2>/dev/null; then
            set +e
            python3 -m coverage report --fail-under="$COV_FAIL_UNDER" > "$COV_REPORT_FILE" 2>&1
            COV_EXIT=$?
            set -e
            COV_OUTPUT=$(cat "$COV_REPORT_FILE" 2>/dev/null || echo "")
        else
            COV_EXIT=1
            COV_OUTPUT=""
        fi
        COV_OUTPUT_FILE="$COV_REPORT_FILE"
        if [ "$COV_GROUP_FAILED" = true ] && [ "$COV_EXIT" -eq 0 ]; then
            COV_EXIT=1
        fi
    else
        # 运行带覆盖率的单元测试（一次完成，避免重复跑 pytest）
        COV_OUTPUT_FILE="$LOG_DIR/pytest_unit_cov.log"
        local start_ms
        start_ms=$(now_ms)
        set +e
        run_command_with_timeout "$COVERAGE_TIMEOUT" "$COV_OUTPUT_FILE" false pytest \
            "${PYTEST_COMMON_ARGS[@]}" \
            -m "$UNIT_MARKER_EXPR" \
            --cov=run --cov=core --cov=agents --cov=coordinator --cov=cursor --cov=tasks \
            --cov=knowledge --cov=indexing --cov=process --cov-report=term-missing --cov-fail-under="$COV_FAIL_UNDER" \
            tests/
        COV_EXIT=$?
        set -e
        local duration_ms=$(( $(now_ms) - start_ms ))
        COV_OUTPUT=$(cat "$COV_OUTPUT_FILE" 2>/dev/null || echo "")
        COV_CMD_STR=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" -m "$UNIT_MARKER_EXPR" \
            --cov=run --cov=core --cov=agents --cov=coordinator --cov=cursor --cov=tasks \
            --cov=knowledge --cov=indexing --cov=process --cov-report=term-missing --cov-fail-under="$COV_FAIL_UNDER" tests/)
        if [ "$COV_EXIT" -ne 0 ]; then
            COV_LAST_TEST=$(extract_last_test_case "$COV_OUTPUT_FILE")
            COV_META_JSON=$(build_meta_json "$COV_LAST_TEST" "$COV_CMD_STR")
        fi
    fi

    # 提取覆盖率百分比
    TOTAL_COV=$(echo "$COV_OUTPUT" | grep -E "^TOTAL" | awk '{print $NF}' | tr -d '%')

    if [ -n "$TOTAL_COV" ]; then
        if [ "$COV_EXIT" -eq 0 ]; then
            check_pass "单元测试通过，覆盖率: ${TOTAL_COV}% (阈值: ${COV_FAIL_UNDER}%)" "" "$COV_OUTPUT_FILE" "$duration_ms"
        else
            if [ "$COV_EXIT" -eq 124 ]; then
                check_fail "单元测试/覆盖率检查超时" "timeout" "$COV_OUTPUT_FILE" "$duration_ms" "$COV_META_JSON"
                if [ "$JSON_OUTPUT" != true ] && [ -n "$COV_OUTPUT_FILE" ]; then
                    echo -e "  ${YELLOW}⚠${NC} 超时前输出:"
                    tail -n 30 "$COV_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
                    [ -n "$COV_LAST_TEST" ] && check_info "最后开始的测试: $COV_LAST_TEST"
                    check_info "日志: $COV_OUTPUT_FILE"
                    [ -n "$COV_CMD_STR" ] && check_info "复现: $COV_CMD_STR"
                fi
            else
                # 检查是否是覆盖率不足导致的失败
                if echo "$COV_OUTPUT" | grep -q "FAIL Required test coverage"; then
                    check_fail "代码覆盖率不足: ${TOTAL_COV}% (阈值: ${COV_FAIL_UNDER}%)" "coverage" "$COV_OUTPUT_FILE" "$duration_ms" "$COV_META_JSON"
                    if [ "$JSON_OUTPUT" != true ] && [ -n "$COV_OUTPUT_FILE" ]; then
                        echo -e "  ${YELLOW}⚠${NC} 覆盖率不足输出摘要:"
                        tail -n 20 "$COV_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
                        [ -n "$COV_LAST_TEST" ] && check_info "最后开始的测试: $COV_LAST_TEST"
                        check_info "日志: $COV_OUTPUT_FILE"
                        [ -n "$COV_CMD_STR" ] && check_info "复现: $COV_CMD_STR"
                    fi
                else
                    # 测试失败但覆盖率可能已计算
                    check_fail "单元测试失败，覆盖率: ${TOTAL_COV}%" "failed" "$COV_OUTPUT_FILE" "$duration_ms" "$COV_META_JSON"
                    if [ "$JSON_OUTPUT" != true ] && [ -n "$COV_OUTPUT_FILE" ]; then
                        echo -e "  ${YELLOW}⚠${NC} 失败输出摘要:"
                        tail -n 20 "$COV_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
                        [ -n "$COV_LAST_TEST" ] && check_info "最后开始的测试: $COV_LAST_TEST"
                        check_info "日志: $COV_OUTPUT_FILE"
                        [ -n "$COV_CMD_STR" ] && check_info "复现: $COV_CMD_STR"
                    fi
                fi
            fi
        fi

        # 显示详细信息 (非 JSON 模式)
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 覆盖率详情:"
            echo "$COV_OUTPUT" | grep -E "^(Name|TOTAL|run|core|agents|coordinator|cursor|tasks|knowledge|indexing|process)" | head -20 | while read line; do
                echo "      $line"
            done
        fi
    else
        if [ "$COV_EXIT" -eq 124 ]; then
            check_fail "单元测试/覆盖率检查超时" "timeout" "$COV_OUTPUT_FILE" "$duration_ms" "$COV_META_JSON"
        else
            check_warn "无法获取覆盖率数据"
        fi
        if [ "$JSON_OUTPUT" != true ]; then
            echo "$COV_OUTPUT" | tail -10
        fi
    fi

    # 生成 HTML 报告 (如果需要) - 跳过，因为已经在上面运行过覆盖率测试
    # 如果上面的测试成功，coverage 数据已存在，只需生成报告
    if [ "$FULL_CHECK" = true ] && [ "$COV_EXIT" -eq 0 ]; then
        # 使用已有的 .coverage 数据生成报告（不重新运行测试）
        if python3 -c "import coverage" 2>/dev/null; then
            python3 -m coverage html -d htmlcov 2>/dev/null || true
            python3 -m coverage xml -o coverage.xml 2>/dev/null || true
        fi
        if [ -d "htmlcov" ]; then
            check_info "HTML 覆盖率报告: htmlcov/index.html"
        fi
        if [ -f "coverage.xml" ]; then
            check_info "XML 覆盖率报告: coverage.xml"
        fi
    fi
}

check_e2e_tests() {
    print_section "E2E 测试"

    if ! command -v pytest &> /dev/null; then
        check_skip "pytest 未安装 (pip install pytest)"
        return 0
    fi
    prepare_pytest_env

    if [ "$JSON_OUTPUT" != true ]; then
        echo -e "  ${BLUE}ℹ${NC} 运行 E2E 测试..."
    fi

    E2E_OUTPUT_FILE="$LOG_DIR/pytest_e2e.log"
    local start_ms
    start_ms=$(now_ms)
    set +e
    run_pytest_group "$E2E_OUTPUT_FILE" "$E2E_TIMEOUT" "${PYTEST_COMMON_ARGS[@]}" \
        -m "$E2E_MARKER_EXPR" tests/
    E2E_EXIT=$?
    set -e
    local duration_ms=$(( $(now_ms) - start_ms ))

    if [ "$E2E_EXIT" -eq 0 ]; then
        PASSED=$(parse_pytest_count "$E2E_OUTPUT_FILE" "passed")
        check_pass "E2E 测试通过: ${PASSED:-0} 个" "" "$E2E_OUTPUT_FILE" "$duration_ms"
    elif [ "$E2E_EXIT" -eq 5 ]; then
        # pytest 退出码 5: no tests collected
        check_info "未找到 E2E 测试用例（不计为失败）"
    elif [ "$E2E_EXIT" -eq 124 ]; then
        last_test=$(extract_last_test_case "$E2E_OUTPUT_FILE")
        cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" -m "$E2E_MARKER_EXPR" tests/)
        meta_json=$(build_meta_json "$last_test" "$cmd_str")
        check_fail "E2E 测试超时 (${E2E_TIMEOUT}s)" "timeout" "$E2E_OUTPUT_FILE" "$duration_ms" "$meta_json"
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${YELLOW}⚠${NC} 超时前输出:"
            tail -n 30 "$E2E_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
            check_info "日志: $E2E_OUTPUT_FILE"
            check_info "复现: $cmd_str"
        fi
    else
        FAILED=$(parse_pytest_count "$E2E_OUTPUT_FILE" "failed")
        last_test=$(extract_last_test_case "$E2E_OUTPUT_FILE")
        cmd_str=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" -m "$E2E_MARKER_EXPR" tests/)
        meta_json=$(build_meta_json "$last_test" "$cmd_str")
        check_fail "E2E 测试失败: ${FAILED:-?} 个" "failed" "$E2E_OUTPUT_FILE" "$duration_ms" "$meta_json"
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${YELLOW}⚠${NC} 失败输出摘要:"
            tail -n 30 "$E2E_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
            check_info "日志: $E2E_OUTPUT_FILE"
            check_info "复现: $cmd_str"
        fi
    fi
}

check_network_isolation_tests() {
    # 网络隔离测试（与 CI no-api-key-smoke-test job 对齐）
    # 包含 test_network_blocking.py 和 test_no_api_key_network_isolation.py
    print_section "网络隔离测试"

    if ! command -v pytest &> /dev/null; then
        check_skip "pytest 未安装 (pip install pytest)"
        return 0
    fi
    prepare_pytest_env

    # 验证测试文件存在
    MISSING_FILES=0
    for test_file in "${NETWORK_ISOLATION_TEST_FILES[@]}"; do
        if [ ! -f "$test_file" ]; then
            check_fail "网络隔离测试文件缺失: $test_file"
            ((MISSING_FILES++))
        fi
    done

    if [ $MISSING_FILES -gt 0 ]; then
        check_fail "有 $MISSING_FILES 个网络隔离测试文件缺失"
        return 1
    fi

    check_pass "所有网络隔离测试文件存在 (${#NETWORK_ISOLATION_TEST_FILES[@]} 个)"

    if [ "$JSON_OUTPUT" != true ]; then
        echo -e "  ${BLUE}ℹ${NC} 运行网络隔离测试..."
        echo -e "  ${BLUE}ℹ${NC} 这些测试验证无 API Key 时的网络阻断机制"
        echo ""
        echo -e "  ${CYAN}复现命令:${NC}"
        echo -e "    # 同步网络阻断用例"
        echo -e "    CURSOR_API_KEY='' CURSOR_CLOUD_API_KEY='' pytest tests/test_network_blocking.py -v --timeout=60"
        echo ""
        echo -e "    # 异步网络隔离用例"
        echo -e "    CURSOR_API_KEY='' CURSOR_CLOUD_API_KEY='' pytest tests/test_no_api_key_network_isolation.py -v --timeout=60"
        echo ""
    fi

    # 使用子 shell 隔离环境变量
    NETWORK_TEST_OUTPUT_FILE="$LOG_DIR/pytest_network_isolation.log"
    ensure_log_dir

    local start_ms
    start_ms=$(now_ms)
    set +e
    (
        # 清除 API Key 环境变量
        unset CURSOR_API_KEY
        unset CURSOR_CLOUD_API_KEY
        export CURSOR_API_KEY=""
        export CURSOR_CLOUD_API_KEY=""

        run_pytest_group "$NETWORK_TEST_OUTPUT_FILE" 180 "${PYTEST_COMMON_ARGS[@]}" \
            "${NETWORK_ISOLATION_TEST_FILES[@]}"
    )
    NETWORK_EXIT=$?
    set -e
    local duration_ms=$(( $(now_ms) - start_ms ))

    if [ "$NETWORK_EXIT" -eq 0 ]; then
        PASSED=$(parse_pytest_count "$NETWORK_TEST_OUTPUT_FILE" "passed")
        check_pass "网络隔离测试通过: ${PASSED:-0} 个" "" "$NETWORK_TEST_OUTPUT_FILE" "$duration_ms"
    elif [ "$NETWORK_EXIT" -eq 5 ]; then
        check_warn "未找到网络隔离测试用例（不计为失败）"
    elif [ "$NETWORK_EXIT" -eq 124 ]; then
        last_test=$(extract_last_test_case "$NETWORK_TEST_OUTPUT_FILE")
        cmd_str="CURSOR_API_KEY='' CURSOR_CLOUD_API_KEY='' pytest ${NETWORK_ISOLATION_TEST_FILES[*]} -v --timeout=60"
        meta_json=$(build_meta_json "$last_test" "$cmd_str")
        check_fail "网络隔离测试超时" "timeout" "$NETWORK_TEST_OUTPUT_FILE" "$duration_ms" "$meta_json"
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${YELLOW}⚠${NC} 超时前输出:"
            tail -n 30 "$NETWORK_TEST_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
            check_info "日志: $NETWORK_TEST_OUTPUT_FILE"
        fi
    else
        FAILED=$(parse_pytest_count "$NETWORK_TEST_OUTPUT_FILE" "failed")
        last_test=$(extract_last_test_case "$NETWORK_TEST_OUTPUT_FILE")
        cmd_str="CURSOR_API_KEY='' CURSOR_CLOUD_API_KEY='' pytest ${NETWORK_ISOLATION_TEST_FILES[*]} -v --timeout=60"
        meta_json=$(build_meta_json "$last_test" "$cmd_str")
        check_fail "网络隔离测试失败: ${FAILED:-?} 个" "failed" "$NETWORK_TEST_OUTPUT_FILE" "$duration_ms" "$meta_json"
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${YELLOW}⚠${NC} 失败输出摘要:"
            tail -n 30 "$NETWORK_TEST_OUTPUT_FILE" 2>/dev/null | sed 's/^/    /'
            [ -n "$last_test" ] && check_info "最后开始的测试: $last_test"
            check_info "日志: $NETWORK_TEST_OUTPUT_FILE"
        fi
    fi
}

check_config() {
    print_section "配置文件检查"

    # config.yaml
    if [ -f "config.yaml" ]; then
        if python3 -c "import yaml; yaml.safe_load(open('config.yaml'))" 2>/dev/null; then
            check_pass "config.yaml 格式正确"
        else
            check_fail "config.yaml 格式错误"
        fi
    else
        check_fail "config.yaml 不存在"
    fi

    # mcp.json
    if [ -f "mcp.json" ]; then
        if python3 -c "import json; json.load(open('mcp.json'))" 2>/dev/null; then
            check_pass "mcp.json 格式正确"
        else
            check_fail "mcp.json 格式错误"
        fi
    else
        check_warn "mcp.json 不存在"
    fi

    # .cursor 目录
    if [ -d ".cursor" ]; then
        check_pass ".cursor 目录存在"

        # 检查子配置
        [ -f ".cursor/cli.json" ] && check_pass ".cursor/cli.json 存在" || check_warn ".cursor/cli.json 不存在"
        [ -f ".cursor/hooks.json" ] && check_pass ".cursor/hooks.json 存在" || check_info ".cursor/hooks.json 不存在"
        [ -d ".cursor/agents" ] && check_pass ".cursor/agents/ 目录存在" || check_info ".cursor/agents/ 不存在"
        [ -d ".cursor/rules" ] && check_pass ".cursor/rules/ 目录存在" || check_info ".cursor/rules/ 不存在"
    else
        check_warn ".cursor 目录不存在"
    fi
}

check_git() {
    print_section "Git 状态检查"

    if [ -d ".git" ]; then
        check_pass "Git 仓库已初始化"

        # 当前分支
        BRANCH=$(git branch --show-current 2>/dev/null)
        check_info "当前分支: $BRANCH"

        # 未提交更改
        CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
        if [ "$CHANGES" -eq 0 ]; then
            check_pass "工作区干净"
        else
            check_warn_or_info "有 $CHANGES 个未提交的更改"
        fi

        # 未跟踪文件
        UNTRACKED=$(git status --porcelain 2>/dev/null | grep "^??" | wc -l)
        if [ "$UNTRACKED" -gt 0 ]; then
            check_info "有 $UNTRACKED 个未跟踪的文件"
        fi

        # 检查与远程的差异（不做 fetch，使用缓存信息）
        BEHIND=$(git rev-list --count HEAD..@{upstream} 2>/dev/null || echo "")
        AHEAD=$(git rev-list --count @{upstream}..HEAD 2>/dev/null || echo "")

        if [ -n "$BEHIND" ] && [ "$BEHIND" -gt 0 ]; then
            check_info "落后远程 $BEHIND 个提交 (使用缓存)"
        fi
        if [ -n "$AHEAD" ] && [ "$AHEAD" -gt 0 ]; then
            check_info "领先远程 $AHEAD 个提交"
        fi
    else
        check_warn "非 Git 仓库"
    fi
}

check_agent_cli() {
    print_section "Agent CLI 检查"

    if command -v agent &> /dev/null; then
        # 只获取版本，不做网络调用
        AGENT_OUTPUT="/tmp/agent_version.txt"
        if run_command_with_timeout 5 "$AGENT_OUTPUT" false agent --version; then
            AGENT_VERSION=$(head -n 1 "$AGENT_OUTPUT")
            [ -z "$AGENT_VERSION" ] && AGENT_VERSION="未知"
            check_pass "agent CLI 已安装: $AGENT_VERSION"
        else
            AGENT_VERSION=$(head -n 1 "$AGENT_OUTPUT")
            [ -z "$AGENT_VERSION" ] && AGENT_VERSION="未知"
            check_warn "agent CLI 可执行，但获取版本失败: $AGENT_VERSION"
        fi
        rm -f "$AGENT_OUTPUT"
        check_info "运行 'agent status' 检查认证状态"
    else
        check_warn "agent CLI 未安装"
        check_info "安装命令: curl https://cursor.com/install -fsS | bash"
    fi

    # 检查 API 密钥
    if [ -n "$CURSOR_API_KEY" ]; then
        check_pass "CURSOR_API_KEY 环境变量已设置"
    else
        check_info "CURSOR_API_KEY 未设置 (可选)"
    fi
}

check_knowledge_base() {
    print_section "知识库验证"

    # 检查知识库模块是否可导入
    if PYTHONPATH=. python3 -c "from knowledge import KnowledgeManager" 2>/dev/null; then
        check_pass "knowledge 模块可导入"
    else
        check_fail "knowledge 模块导入失败"
        return 1
    fi

    # 检查验证测试文件
    if [ -f "tests/test_knowledge_validation.py" ]; then
        check_pass "知识库验证测试文件存在"

        # 运行快速验证
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 运行知识库快速验证..."
        fi
        if PYTHONPATH=. python3 tests/test_knowledge_validation.py 2>/dev/null | grep -q "所有验证通过"; then
            check_pass "知识库快速验证通过"
        else
            check_warn "知识库快速验证有问题"
        fi
    else
        check_warn "知识库验证测试文件不存在"
        check_info "运行 'bash scripts/validate_knowledge.sh' 进行验证"
    fi

    # 检查验证脚本
    if [ -f "scripts/validate_knowledge.sh" ]; then
        check_pass "知识库验证脚本存在"
    else
        check_info "知识库验证脚本不存在"
    fi
}

check_pre_commit() {
    print_section "预提交检查 (Python)"

    # 检查预提交检查脚本是否存在
    if [ ! -f "scripts/pre_commit_check.py" ]; then
        check_skip "预提交检查脚本不存在" "scripts/pre_commit_check.py 缺失"
        check_info "运行 'python scripts/pre_commit_check.py' 进行检查"
        return 0
    fi

    # 检查 Python3 是否可用
    if ! command -v python3 &> /dev/null; then
        check_skip "Python3 不可用，跳过预提交检查"
        return 0
    fi

    check_pass "预提交检查脚本存在"

    # 根据检查模式选择参数
    # 默认（非 --full）：快速 core-only 导入验证，仅检查核心依赖
    # --full：更严格的检查（仍使用 core-only 以确保稳定性，额外依赖由 check_deps 覆盖）
    local PRE_COMMIT_ARGS=""
    if [ "$FULL_CHECK" = true ]; then
        # --full 模式：完整检查，但仍使用 core-only 保持稳定
        # 依赖一致性由 check_dependencies 覆盖
        PRE_COMMIT_ARGS="--json --core-only --req-files requirements.txt requirements-test.txt"
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 运行完整预提交检查 (core-only + test deps)..."
        fi
    else
        # 默认模式：快速 core-only 检查，仅检查核心依赖
        PRE_COMMIT_ARGS="--json --quick --core-only --req-files requirements.txt"
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 运行快速预提交检查 (core-only)..."
        fi
    fi

    # 捕获输出并检查结果
    set +e
    PRE_COMMIT_OUTPUT=$(PYTHONPATH=. python3 scripts/pre_commit_check.py $PRE_COMMIT_ARGS 2>&1)
    PRE_COMMIT_EXIT=$?
    set -e

    PRE_COMMIT_JSON=$(printf "%s" "$PRE_COMMIT_OUTPUT" | extract_json_from_output 2>/dev/null || echo "")

    # 处理脚本执行错误（如依赖缺失导致的 ImportError）
    if [ -z "$PRE_COMMIT_JSON" ]; then
        # 无法解析 JSON，可能是依赖缺失或脚本错误
        if echo "$PRE_COMMIT_OUTPUT" | grep -qiE "(ModuleNotFoundError|ImportError|No module named)"; then
            check_warn "预提交检查跳过：依赖未安装"
            if [ "$JSON_OUTPUT" != true ]; then
                echo -e "  ${YELLOW}⚠${NC} 原因: 缺少运行检查脚本所需的依赖"
                echo -e "  ${BLUE}ℹ${NC} 建议: pip install -r requirements.txt"
            fi
            return 0
        else
            check_warn "预提交检查输出无法解析"
            if [ "$JSON_OUTPUT" != true ]; then
                echo "$PRE_COMMIT_OUTPUT" | tail -5 | sed 's/^/    /'
            fi
            return 0
        fi
    fi

    if [ $PRE_COMMIT_EXIT -eq 0 ]; then
        # 解析 JSON 输出获取通过数量
        PASSED_COUNT=$(echo "$PRE_COMMIT_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('passed_count', 0))" 2>/dev/null || echo "?")
        check_pass "预提交检查通过 ($PASSED_COUNT 项)"
    else
        # 解析失败数量
        FAILED_COUNT=$(echo "$PRE_COMMIT_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('failed_count', 0))" 2>/dev/null || echo "?")
        check_fail "预提交检查失败 ($FAILED_COUNT 项)"

        # 在非 JSON 模式下显示失败详情
        if [ "$JSON_OUTPUT" != true ] && [ -n "$PRE_COMMIT_JSON" ]; then
            echo "$PRE_COMMIT_JSON" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for c in d.get('checks', []):
        if not c.get('passed'):
            print(f\"    - {c.get('name')}: {c.get('message')}\")
            if c.get('fix_suggestion'):
                print(f\"      修复: {c.get('fix_suggestion')}\")
except Exception:
    pass
" 2>/dev/null
        fi
    fi
}

check_directories() {
    print_section "目录结构检查"

    REQUIRED_DIRS=("core" "agents" "coordinator" "tasks" "cursor" "scripts" "logs")

    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            FILE_COUNT=$(find "$dir" -name "*.py" 2>/dev/null | wc -l)
            check_pass "目录存在: $dir/ ($FILE_COUNT 个 .py 文件)"
        else
            check_fail "目录缺失: $dir/"
        fi
    done
}

check_main_entries() {
    print_section "入口文件检查"

    # 实际使用的入口文件列表（run.py 及 scripts 下的模式脚本）
    MAIN_FILES=(
        "run.py"
        "scripts/run_basic.py"
        "scripts/run_mp.py"
        "scripts/run_knowledge.py"
        "scripts/run_iterate.py"
    )

    for file in "${MAIN_FILES[@]}"; do
        if [ -f "$file" ]; then
            if python3 -m py_compile "$file" 2>/dev/null; then
                check_pass "入口文件可用: $file"
            else
                check_fail "入口文件语法错误: $file"
            fi
        else
            check_fail "入口文件不存在: $file"
        fi
    done
}

check_deprecated_patterns() {
    print_section "过时模式检测"

    # 检测过时的入口调用模式（避免回归）
    # 注意：这里检测的是代码/文档中的过时引用，不包括注释中的合法引用
    DEPRECATED_PATTERNS=(
        "python main\.py"
        "python main_mp\.py"
        "python main_with_knowledge\.py"
    )

    # 要检查的文件类型（排除 check_all.sh 本身）
    SEARCH_DIRS=("scripts" "docs" "." )
    SEARCH_EXTENSIONS=("*.py" "*.md" "*.sh" "*.txt")
    DEPRECATED_FOUND=0

    for pattern in "${DEPRECATED_PATTERNS[@]}"; do
        # 搜索文件中的过时模式（排除 check_all.sh 自身、.git、venv 等）
        MATCHES=$(grep -rn --include="*.py" --include="*.md" --include="*.sh" --include="*.txt" \
            -E "$pattern" . \
            --exclude-dir=.git --exclude-dir=venv --exclude-dir=__pycache__ \
            --exclude="check_all.sh" \
            2>/dev/null | head -10)

        if [ -n "$MATCHES" ]; then
            check_warn "发现过时模式: $pattern"
            ((DEPRECATED_FOUND++))
            # 非 JSON 模式下显示具体位置
            if [ "$JSON_OUTPUT" != true ]; then
                echo "$MATCHES" | while read -r line; do
                    echo -e "    ${YELLOW}→${NC} $line"
                done | head -5
                MATCH_COUNT=$(echo "$MATCHES" | wc -l)
                if [ "$MATCH_COUNT" -gt 5 ]; then
                    echo -e "    ${YELLOW}... 还有更多匹配 (共 $MATCH_COUNT 处)${NC}"
                fi
            fi
        fi
    done

    if [ $DEPRECATED_FOUND -eq 0 ]; then
        check_pass "未发现过时的入口调用模式"
    else
        check_info "建议: 使用 'python run.py --mode <mode>' 或 'python scripts/run_<mode>.py' 替代"
    fi
}

check_run_modes() {
    print_section "运行模式检查"

    # 检查 run.py 是否存在
    if [ ! -f "run.py" ]; then
        check_fail "run.py 不存在"
        return 1
    fi

    # 验证 run.py --help
    if python3 run.py --help &>/dev/null; then
        check_pass "python run.py --help 可用"
    else
        check_fail "python run.py --help 失败"
        return 1
    fi

    # 定义运行模式
    RUN_MODES=("basic" "iterate" "plan" "ask")
    MODE_ERRORS=0

    # 循环验证所有模式
    for mode in "${RUN_MODES[@]}"; do
        if python3 run.py --mode "$mode" --help &>/dev/null; then
            check_pass "运行模式可用: --mode $mode"
        else
            check_fail "运行模式失败: --mode $mode"
            ((MODE_ERRORS++))
        fi
    done

    # 汇总结果
    if [ $MODE_ERRORS -eq 0 ]; then
        check_info "所有运行模式验证通过 (${#RUN_MODES[@]} 个模式)"
    else
        check_warn "有 $MODE_ERRORS 个运行模式验证失败"
    fi
}

check_no_api_key_smoke() {
    # 无 API Key 环境 Smoke 测试
    # 验证在没有 CURSOR_API_KEY 和 CURSOR_CLOUD_API_KEY 时系统行为正确
    print_section "无 API Key Smoke 测试"

    SMOKE_ERRORS=0

    # 使用子 shell 隔离环境变量
    (
        # 明确清空 API Key 环境变量
        unset CURSOR_API_KEY
        unset CURSOR_CLOUD_API_KEY
        export CURSOR_API_KEY=""
        export CURSOR_CLOUD_API_KEY=""

        # 测试 1: run.py --help
        if python3 run.py --help &>/dev/null; then
            echo "SMOKE_TEST_1=pass"
        else
            echo "SMOKE_TEST_1=fail"
        fi

        # 测试 2: --print-config 能正确显示配置
        if python3 run.py --print-config 2>&1 | grep -qE "(requested_mode|effective_mode|config_path)"; then
            echo "SMOKE_TEST_2=pass"
        else
            echo "SMOKE_TEST_2=fail"
        fi

        # 测试 3: scripts/run_iterate.py --help
        if python3 -m scripts.run_iterate --help &>/dev/null; then
            echo "SMOKE_TEST_3=pass"
        else
            echo "SMOKE_TEST_3=fail"
        fi
    ) > /tmp/smoke_test_results.txt 2>/dev/null

    # 解析结果
    if grep -q "SMOKE_TEST_1=pass" /tmp/smoke_test_results.txt 2>/dev/null; then
        check_pass "无 API Key: run.py --help"
    else
        check_fail "无 API Key: run.py --help 失败"
        ((SMOKE_ERRORS++))
    fi

    if grep -q "SMOKE_TEST_2=pass" /tmp/smoke_test_results.txt 2>/dev/null; then
        check_pass "无 API Key: --print-config"
    else
        check_fail "无 API Key: --print-config 失败"
        ((SMOKE_ERRORS++))
    fi

    if grep -q "SMOKE_TEST_3=pass" /tmp/smoke_test_results.txt 2>/dev/null; then
        check_pass "无 API Key: scripts/run_iterate.py --help"
    else
        check_fail "无 API Key: scripts/run_iterate.py --help 失败"
        ((SMOKE_ERRORS++))
    fi

    rm -f /tmp/smoke_test_results.txt

    if [ $SMOKE_ERRORS -eq 0 ]; then
        check_info "无 API Key Smoke 测试全部通过"
    else
        check_warn "有 $SMOKE_ERRORS 个 Smoke 测试失败"
    fi
}

check_dependency_consistency() {
    # 依赖一致性检查
    # 检查已安装依赖与声明依赖的一致性，检测未声明的 import
    print_section "依赖一致性检查"

    # 检查 check_deps.py 是否存在
    if [[ ! -f "scripts/check_deps.py" ]]; then
        check_skip "依赖检查脚本不存在" "scripts/check_deps.py 缺失"
        return 0
    fi

    # 检查 sync-deps.sh 是否存在
    if [[ ! -f "scripts/sync-deps.sh" ]]; then
        check_warn "依赖同步脚本不存在: scripts/sync-deps.sh"
    fi

    check_pass "依赖检查脚本存在"

    ensure_log_dir
    local DEPS_OUTPUT_FILE="$LOG_DIR/check_deps.json"
    local SYNC_OUTPUT_FILE="$LOG_DIR/sync_deps_check.log"
    local DEPS_ERRORS=0

    # ======================================
    # 步骤 1: 运行 check_deps.py --ci
    # ======================================
    if [[ "$JSON_OUTPUT" != true ]]; then
        echo -e "  ${BLUE}ℹ${NC} 运行依赖版本与导入一致性检查..."
    fi

    local start_ms
    start_ms=$(now_ms)
    set +e
    PYTHONPATH=. python3 scripts/check_deps.py --ci --format json > "$DEPS_OUTPUT_FILE" 2>&1
    local DEPS_EXIT=$?
    set -e
    local duration_ms=$(( $(now_ms) - start_ms ))

    if [[ ! -s "$DEPS_OUTPUT_FILE" ]]; then
        check_fail "依赖检查脚本执行失败（无输出）" "execution_error" "$DEPS_OUTPUT_FILE" "$duration_ms"
        return 1
    fi

    # 解析 JSON 输出
    local PARSE_RESULT
    PARSE_RESULT=$(python3 - "$DEPS_OUTPUT_FILE" <<'PYEOF'
import json
import sys

try:
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"PARSE_ERROR:{e}")
    sys.exit(1)

summary = data.get("summary", {})
has_errors = summary.get("has_errors", False)
missing = data.get("missing_packages", [])
mismatches = data.get("version_mismatches", [])
conflicts = data.get("dependency_conflicts", [])
undeclared = data.get("undeclared_imports", {})
import_summary = data.get("import_consistency_summary", {})

# 输出汇总信息
print(f"HAS_ERRORS:{has_errors}")
print(f"MISSING_COUNT:{len(missing)}")
print(f"MISMATCH_COUNT:{len(mismatches)}")
print(f"CONFLICT_COUNT:{len(conflicts)}")
print(f"UNDECLARED_COUNT:{len(undeclared)}")
print(f"TOTAL_IMPORTS:{import_summary.get('total_imports', 0)}")
print(f"TOTAL_PY_FILES:{import_summary.get('total_py_files', 0)}")

# 输出层级详情（用于定位问题）
if missing:
    print("MISSING:" + ",".join(missing))

# 输出版本不匹配详情（按层级分组）
for m in mismatches:
    source = m.get("source", "unknown")
    pkg = m.get("package", "?")
    spec = m.get("declared_spec", "?")
    installed = m.get("installed_version", "?")
    print(f"MISMATCH:{source}|{pkg}|{spec}|{installed}")

# 输出未声明导入详情（包含文件位置）
for pkg, info in undeclared.items():
    locations = info.get("locations", [])[:3]  # 最多 3 个位置
    loc_str = ";".join(locations) if locations else "unknown"
    print(f"UNDECLARED:{pkg}|{loc_str}")
PYEOF
    )

    if echo "$PARSE_RESULT" | grep -q "^PARSE_ERROR:"; then
        check_fail "依赖检查结果解析失败" "parse_error" "$DEPS_OUTPUT_FILE" "$duration_ms"
        if [[ "$JSON_OUTPUT" != true ]]; then
            echo -e "  ${YELLOW}⚠${NC} 日志: $DEPS_OUTPUT_FILE"
        fi
        return 1
    fi

    # 解析汇总数据
    local HAS_ERRORS MISSING_COUNT MISMATCH_COUNT CONFLICT_COUNT UNDECLARED_COUNT
    HAS_ERRORS=$(echo "$PARSE_RESULT" | grep "^HAS_ERRORS:" | cut -d: -f2)
    MISSING_COUNT=$(echo "$PARSE_RESULT" | grep "^MISSING_COUNT:" | cut -d: -f2)
    MISMATCH_COUNT=$(echo "$PARSE_RESULT" | grep "^MISMATCH_COUNT:" | cut -d: -f2)
    CONFLICT_COUNT=$(echo "$PARSE_RESULT" | grep "^CONFLICT_COUNT:" | cut -d: -f2)
    UNDECLARED_COUNT=$(echo "$PARSE_RESULT" | grep "^UNDECLARED_COUNT:" | cut -d: -f2)

    # 输出检查结果
    if [[ "$HAS_ERRORS" == "True" ]]; then
        ((DEPS_ERRORS++)) || true

        # 输出缺失的包（按层级）
        if [[ "$MISSING_COUNT" -gt 0 ]]; then
            check_fail "缺失 $MISSING_COUNT 个依赖包" "missing" "$DEPS_OUTPUT_FILE" "$duration_ms"
            if [[ "$JSON_OUTPUT" != true ]]; then
                local MISSING_PKGS
                MISSING_PKGS=$(echo "$PARSE_RESULT" | grep "^MISSING:" | cut -d: -f2)
                if [[ -n "$MISSING_PKGS" ]]; then
                    echo -e "  ${YELLOW}⚠${NC} 缺失: $MISSING_PKGS"
                    echo -e "  ${BLUE}ℹ${NC} 修复: pip install $MISSING_PKGS"
                fi
            fi
        fi

        # 输出版本不匹配（按层级分组）
        if [[ "$MISMATCH_COUNT" -gt 0 ]]; then
            check_fail "发现 $MISMATCH_COUNT 个版本不匹配" "mismatch" "$DEPS_OUTPUT_FILE" "$duration_ms"
            if [[ "$JSON_OUTPUT" != true ]]; then
                echo "$PARSE_RESULT" | grep "^MISMATCH:" | while read -r line; do
                    # 格式: MISMATCH:source|pkg|spec|installed
                    local mismatch_data="${line#MISMATCH:}"
                    local mismatch_source="${mismatch_data%%|*}"
                    local mismatch_rest="${mismatch_data#*|}"
                    local mismatch_pkg="${mismatch_rest%%|*}"
                    mismatch_rest="${mismatch_rest#*|}"
                    local mismatch_spec="${mismatch_rest%%|*}"
                    local mismatch_installed="${mismatch_rest#*|}"
                    echo -e "  ${YELLOW}⚠${NC} [$mismatch_source] $mismatch_pkg: 要求 $mismatch_spec, 已安装 $mismatch_installed"
                done | head -5
                if [[ "$MISMATCH_COUNT" -gt 5 ]]; then
                    echo -e "  ${YELLOW}...${NC} 还有 $((MISMATCH_COUNT - 5)) 个不匹配"
                fi
                echo -e "  ${BLUE}ℹ${NC} 修复: pip-sync requirements.txt requirements-dev.txt requirements-test.txt"
            fi
        fi

        # 输出依赖冲突
        if [[ "$CONFLICT_COUNT" -gt 0 ]]; then
            check_fail "发现 $CONFLICT_COUNT 个依赖冲突" "conflict" "$DEPS_OUTPUT_FILE" "$duration_ms"
            if [[ "$JSON_OUTPUT" != true ]]; then
                echo -e "  ${BLUE}ℹ${NC} 修复: bash scripts/sync-deps.sh compile && pip-sync requirements.txt requirements-dev.txt requirements-test.txt"
            fi
        fi

        # 输出未声明的导入（包含文件位置）
        if [[ "$UNDECLARED_COUNT" -gt 0 ]]; then
            check_fail "发现 $UNDECLARED_COUNT 个未声明的第三方导入" "undeclared" "$DEPS_OUTPUT_FILE" "$duration_ms"
            if [[ "$JSON_OUTPUT" != true ]]; then
                echo "$PARSE_RESULT" | grep "^UNDECLARED:" | while read -r line; do
                    # 格式: UNDECLARED:pkg|loc1;loc2;loc3
                    local undeclared_data="${line#UNDECLARED:}"
                    local undeclared_pkg="${undeclared_data%%|*}"
                    local undeclared_locs="${undeclared_data#*|}"
                    # 将分号分隔的位置转换为逗号+空格分隔
                    local loc_display=$(echo "$undeclared_locs" | sed 's/;/, /g')
                    echo -e "  ${YELLOW}⚠${NC} $undeclared_pkg (引用: $loc_display)"
                done | head -5
                if [[ "$UNDECLARED_COUNT" -gt 5 ]]; then
                    echo -e "  ${YELLOW}...${NC} 还有 $((UNDECLARED_COUNT - 5)) 个未声明导入"
                fi
                echo -e "  ${BLUE}ℹ${NC} 修复: 将包添加到对应层级的 requirements 文件"
                echo -e "  ${BLUE}ℹ${NC}   核心依赖 → requirements.in"
                echo -e "  ${BLUE}ℹ${NC}   开发依赖 → requirements-dev.in"
                echo -e "  ${BLUE}ℹ${NC}   测试依赖 → requirements-test.in"
                echo -e "  ${BLUE}ℹ${NC}   ML测试依赖 → requirements-test-ml.in"
                echo -e "  ${BLUE}ℹ${NC}   然后运行: bash scripts/sync-deps.sh compile"
            fi
        fi
    else
        check_pass "依赖版本与导入一致性检查通过" "" "$DEPS_OUTPUT_FILE" "$duration_ms"
    fi

    # ======================================
    # 步骤 2: 运行 sync-deps.sh check（可选）
    # ======================================
    if [[ -f "scripts/sync-deps.sh" ]]; then
        if [[ "$JSON_OUTPUT" != true ]]; then
            echo -e "  ${BLUE}ℹ${NC} 运行 pyproject.toml 与 .in 文件同步检查..."
        fi

        local sync_start_ms
        sync_start_ms=$(now_ms)
        set +e
        bash scripts/sync-deps.sh check > "$SYNC_OUTPUT_FILE" 2>&1
        local SYNC_EXIT=$?
        set -e
        local sync_duration_ms=$(( $(now_ms) - sync_start_ms ))

        if [[ $SYNC_EXIT -eq 0 ]]; then
            check_pass "pyproject.toml 与 .in 文件同步检查通过" "" "$SYNC_OUTPUT_FILE" "$sync_duration_ms"
        else
            ((DEPS_ERRORS++)) || true
            check_fail "pyproject.toml 与 .in 文件不同步" "sync_mismatch" "$SYNC_OUTPUT_FILE" "$sync_duration_ms"

            if [[ "$JSON_OUTPUT" != true ]]; then
                # 提取警告信息
                grep -E "^\[警告\]" "$SYNC_OUTPUT_FILE" 2>/dev/null | head -5 | while read -r line; do
                    echo -e "  ${YELLOW}⚠${NC} ${line#\[警告\] }"
                done
                # 检查锁文件过期
                if grep -q "可能已过期" "$SYNC_OUTPUT_FILE" 2>/dev/null; then
                    echo -e "  ${BLUE}ℹ${NC} 修复: bash scripts/sync-deps.sh compile"
                fi
                # 检查依赖缺失
                if grep -q "中缺少" "$SYNC_OUTPUT_FILE" 2>/dev/null; then
                    echo -e "  ${BLUE}ℹ${NC} 修复: 同步 pyproject.toml 与 requirements.in 文件"
                fi
                echo -e "  ${BLUE}ℹ${NC} 详细日志: $SYNC_OUTPUT_FILE"
            fi
        fi
    fi

    # 汇总
    if [[ $DEPS_ERRORS -eq 0 ]]; then
        check_info "依赖一致性检查全部通过"
    else
        check_warn "依赖一致性检查发现 $DEPS_ERRORS 类问题"
        if [[ "$JSON_OUTPUT" != true ]]; then
            echo -e "  ${BLUE}ℹ${NC} 查看完整报告: python3 scripts/check_deps.py --format text"
        fi
    fi
}

check_format() {
    # 格式检查（与 pr-check.yml format-check job 对齐）
    # 包括：ruff format --check 和 isort --check-only --diff
    # 失败 -> warn（不阻止 CI，只是警告）
    print_section "格式检查"

    ensure_log_dir
    local FORMAT_ERRORS=0

    # ======================================
    # 检查 1: ruff format --check
    # ======================================
    if command -v ruff &> /dev/null; then
        local RUFF_FORMAT_OUTPUT_FILE="$LOG_DIR/ruff_format_check.log"
        local ruff_start_ms
        ruff_start_ms=$(now_ms)
        set +e
        ruff format --check . > "$RUFF_FORMAT_OUTPUT_FILE" 2>&1
        local RUFF_FORMAT_EXIT=$?
        set -e
        local ruff_duration_ms=$(( $(now_ms) - ruff_start_ms ))

        if [ "$RUFF_FORMAT_EXIT" -eq 0 ]; then
            check_pass "ruff format 检查通过" "" "$RUFF_FORMAT_OUTPUT_FILE" "$ruff_duration_ms"
        else
            ((FORMAT_ERRORS++)) || true
            local ruff_meta_json
            ruff_meta_json=$(build_meta_json "" "ruff format .")
            check_warn_or_info "发现未格式化的代码" "ruff format --check 失败" "$RUFF_FORMAT_OUTPUT_FILE" "$ruff_duration_ms" "$ruff_meta_json"
            if [ "$JSON_OUTPUT" != true ]; then
                echo -e "  ${BLUE}ℹ${NC} 修复: ruff format ."
                echo -e "  ${BLUE}ℹ${NC} 日志: $RUFF_FORMAT_OUTPUT_FILE"
            fi
        fi
    else
        check_skip "ruff 未安装 (pip install ruff)"
    fi

    # ======================================
    # 检查 2: ruff import 排序（I001）
    # ======================================
    if command -v ruff &> /dev/null; then
        local RUFF_IMPORT_OUTPUT_FILE="$LOG_DIR/ruff_import_check.log"
        local ruff_import_start_ms
        ruff_import_start_ms=$(now_ms)
        set +e
        ruff check --select I001 . > "$RUFF_IMPORT_OUTPUT_FILE" 2>&1
        local RUFF_IMPORT_EXIT=$?
        set -e
        local ruff_import_duration_ms=$(( $(now_ms) - ruff_import_start_ms ))

        if [ "$RUFF_IMPORT_EXIT" -eq 0 ]; then
            check_pass "Import 排序检查通过 (ruff)" "" "$RUFF_IMPORT_OUTPUT_FILE" "$ruff_import_duration_ms"
        else
            ((FORMAT_ERRORS++)) || true
            local ruff_import_meta_json
            ruff_import_meta_json=$(build_meta_json "" "ruff check --select I001 .")
            check_warn_or_info "Import 排序不正确" "ruff I001 检查失败" "$RUFF_IMPORT_OUTPUT_FILE" "$ruff_import_duration_ms" "$ruff_import_meta_json"
            if [ "$JSON_OUTPUT" != true ]; then
                echo -e "  ${BLUE}ℹ${NC} 修复: ruff check --select I001 --fix ."
                echo -e "  ${BLUE}ℹ${NC} 日志: $RUFF_IMPORT_OUTPUT_FILE"
            fi
        fi
    else
        check_skip "ruff 未安装 (pip install ruff)"
    fi

    # 汇总
    if [ $FORMAT_ERRORS -eq 0 ]; then
        check_info "格式检查全部通过"
    else
        check_info "发现 $FORMAT_ERRORS 个格式问题（不影响 CI 通过，建议修复）"
    fi
}

check_no_api_key_iterate_config() {
    # 无 API Key 环境下 Iterate 模式配置验证
    # 详细验证 requested_mode/effective_mode/orchestrator 输出稳定性
    print_section "无 API Key Iterate 配置验证"

    ITERATE_ERRORS=0
    ITERATE_OUTPUT_FILE="$LOG_DIR/no_api_key_iterate_config.log"
    ensure_log_dir

    # 使用子 shell 隔离环境变量
    (
        # 明确清空 API Key 环境变量（双重保障：unset + export 空值）
        unset CURSOR_API_KEY
        unset CURSOR_CLOUD_API_KEY
        export CURSOR_API_KEY=""
        export CURSOR_CLOUD_API_KEY=""

        # 测试 1: run.py --mode iterate --print-config
        # 验证输出包含正确的 requested_mode/effective_mode/orchestrator
        echo "=== TEST 1: run.py --mode iterate --print-config ===" >> "$ITERATE_OUTPUT_FILE"
        python3 run.py --mode iterate --print-config >> "$ITERATE_OUTPUT_FILE" 2>&1
        TEST1_EXIT=$?
        echo "EXIT_CODE=$TEST1_EXIT" >> "$ITERATE_OUTPUT_FILE"
        echo "" >> "$ITERATE_OUTPUT_FILE"

        # 检查关键字段
        # requested_mode 应为 auto（来自 config.yaml 默认值或 CLI）
        if grep -q "requested_mode: auto" "$ITERATE_OUTPUT_FILE"; then
            echo "ITERATE_TEST_1A=pass"
        else
            echo "ITERATE_TEST_1A=fail"
        fi

        # effective_mode 应为 cli（无 API Key 时回退到 cli）
        if grep -q "effective_mode: cli" "$ITERATE_OUTPUT_FILE"; then
            echo "ITERATE_TEST_1B=pass"
        else
            echo "ITERATE_TEST_1B=fail"
        fi

        # orchestrator 应为 basic（requested_mode=auto 强制 basic）
        if grep -q "orchestrator: basic" "$ITERATE_OUTPUT_FILE"; then
            echo "ITERATE_TEST_1C=pass"
        else
            echo "ITERATE_TEST_1C=fail"
        fi

        # 测试 2: scripts/run_iterate.py --minimal --execution-mode auto
        # 验证 minimal 模式能正常结束（不触网/不写入/不执行 Orchestrator.run）
        echo "=== TEST 2: scripts/run_iterate.py --minimal --execution-mode auto ===" >> "$ITERATE_OUTPUT_FILE"
        # 设置超时避免卡死（30 秒足够 minimal 模式完成）
        # macOS 兼容：优先使用 gtimeout，否则直接运行（无超时保护）
        if command -v timeout &> /dev/null; then
            timeout 30 python3 -m scripts.run_iterate --minimal --execution-mode auto "分析代码结构" >> "$ITERATE_OUTPUT_FILE" 2>&1
            TEST2_EXIT=$?
        elif command -v gtimeout &> /dev/null; then
            gtimeout 30 python3 -m scripts.run_iterate --minimal --execution-mode auto "分析代码结构" >> "$ITERATE_OUTPUT_FILE" 2>&1
            TEST2_EXIT=$?
        else
            # 无超时工具，直接运行（依赖 minimal 模式快速完成）
            python3 -m scripts.run_iterate --minimal --execution-mode auto "分析代码结构" >> "$ITERATE_OUTPUT_FILE" 2>&1
            TEST2_EXIT=$?
        fi
        echo "EXIT_CODE=$TEST2_EXIT" >> "$ITERATE_OUTPUT_FILE"
        echo "" >> "$ITERATE_OUTPUT_FILE"

        # minimal 模式应正常退出（退出码 0 表示成功）
        if [ "$TEST2_EXIT" -eq 0 ]; then
            echo "ITERATE_TEST_2=pass"
        else
            echo "ITERATE_TEST_2=fail"
        fi

        # 测试 3: scripts/run_iterate.py --print-config --execution-mode auto
        # 验证 scripts/run_iterate.py 也支持 --print-config 且输出一致
        echo "=== TEST 3: scripts/run_iterate.py --print-config --execution-mode auto ===" >> "$ITERATE_OUTPUT_FILE"
        python3 -m scripts.run_iterate --print-config --execution-mode auto >> "$ITERATE_OUTPUT_FILE" 2>&1
        TEST3_EXIT=$?
        echo "EXIT_CODE=$TEST3_EXIT" >> "$ITERATE_OUTPUT_FILE"
        echo "" >> "$ITERATE_OUTPUT_FILE"

        # 检查输出是否包含 requested_mode: auto
        if grep -q "requested_mode: auto" "$ITERATE_OUTPUT_FILE"; then
            echo "ITERATE_TEST_3=pass"
        else
            echo "ITERATE_TEST_3=fail"
        fi

    ) > /tmp/iterate_test_results.txt 2>/dev/null

    # 解析测试 1A 结果: requested_mode
    if grep -q "ITERATE_TEST_1A=pass" /tmp/iterate_test_results.txt 2>/dev/null; then
        check_pass "无 API Key: requested_mode: auto"
    else
        check_fail "无 API Key: requested_mode 不为 auto"
        ((ITERATE_ERRORS++))
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${YELLOW}⚠${NC} 日志: $ITERATE_OUTPUT_FILE"
        fi
    fi

    # 解析测试 1B 结果: effective_mode
    if grep -q "ITERATE_TEST_1B=pass" /tmp/iterate_test_results.txt 2>/dev/null; then
        check_pass "无 API Key: effective_mode: cli"
    else
        check_fail "无 API Key: effective_mode 不为 cli（预期无 Key 时回退）"
        ((ITERATE_ERRORS++))
    fi

    # 解析测试 1C 结果: orchestrator
    if grep -q "ITERATE_TEST_1C=pass" /tmp/iterate_test_results.txt 2>/dev/null; then
        check_pass "无 API Key: orchestrator: basic"
    else
        check_fail "无 API Key: orchestrator 不为 basic（预期 auto 模式强制 basic）"
        ((ITERATE_ERRORS++))
    fi

    # 解析测试 2 结果: minimal 模式
    if grep -q "ITERATE_TEST_2=pass" /tmp/iterate_test_results.txt 2>/dev/null; then
        check_pass "无 API Key: --minimal 模式正常结束"
    else
        check_fail "无 API Key: --minimal 模式执行失败或超时"
        ((ITERATE_ERRORS++))
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${YELLOW}⚠${NC} 日志: $ITERATE_OUTPUT_FILE"
            echo -e "  ${YELLOW}⚠${NC} 提示: minimal 模式应跳过网络请求和 Orchestrator.run"
        fi
    fi

    # 解析测试 3 结果: scripts/run_iterate.py --print-config
    if grep -q "ITERATE_TEST_3=pass" /tmp/iterate_test_results.txt 2>/dev/null; then
        check_pass "无 API Key: scripts/run_iterate.py --print-config"
    else
        check_fail "无 API Key: scripts/run_iterate.py --print-config 失败"
        ((ITERATE_ERRORS++))
    fi

    rm -f /tmp/iterate_test_results.txt

    if [ $ITERATE_ERRORS -eq 0 ]; then
        check_info "无 API Key Iterate 配置验证全部通过"
    else
        check_warn "有 $ITERATE_ERRORS 个 Iterate 配置验证失败"
        if [ "$JSON_OUTPUT" != true ]; then
            check_info "详细日志: $ITERATE_OUTPUT_FILE"
        fi
    fi
}

# ============================================================
# 模式检查辅助函数
# ============================================================

# 检查是否应运行指定模式
# 用法: should_run_mode "import" && run_import_checks
should_run_mode() {
    local mode="$1"
    
    # 未指定模式时运行所有检查
    if [ "$CHECK_MODE_SET" = false ]; then
        return 0
    fi
    
    # 检查是否包含 "all" 模式
    for m in "${CHECK_MODES[@]}"; do
        if [ "$m" = "all" ]; then
            return 0
        fi
    done
    
    # 检查是否包含指定模式
    for m in "${CHECK_MODES[@]}"; do
        if [ "$m" = "$mode" ]; then
            return 0
        fi
    done
    
    return 1
}

# 运行 import 模式检查（对齐 import-test.yml）
# 包括：语法检查 + 核心模块导入
run_import_mode() {
    run_check "Python 语法检查" check_syntax
    run_check "模块结构检查" check_imports
    run_check "预提交检查 (Python)" check_pre_commit
}

# 运行 lint 模式检查（对齐 lint.yml）
# 包括：flake8-critical + ruff + mypy
run_lint_mode() {
    run_check "代码风格检查" check_code_style
    run_check "类型检查 (mypy)" check_type_hints
}

# 运行 test 模式检查（对齐 ci.yml test/e2e）
# 包括：单元测试 + 覆盖率 + E2E 测试
run_test_mode() {
    # 准备 pytest 环境
    prepare_pytest_env
    
    # 根据 FULL_CHECK 和 SPLIT_TESTS 选择测试方式
    if [ "$SPLIT_TESTS" = true ]; then
        if [ "$SPLIT_COVERAGE" = true ]; then
            run_check "单元测试 + 覆盖率" check_unit_tests_with_coverage
        else
            run_check "单元测试" check_unit_tests
        fi
    else
        run_check "单元测试 + 覆盖率" check_unit_tests_with_coverage
    fi
    
    run_check "E2E 测试" check_e2e_tests
    
    # 网络隔离测试
    if [ "$RUN_NETWORK_ISOLATION" = true ]; then
        run_check "网络隔离测试" check_network_isolation_tests
    fi
}

# 运行 minimal 模式检查（轻量检查，无额外依赖）
# 仅依赖 bash + python3，用于快速验证基础环境
run_minimal_mode() {
    print_section "轻量检查 (minimal)"
    
    local mode_start_ms
    mode_start_ms=$(now_ms)
    
    # 检查 1: Python 版本检查（pass）
    local py_version
    py_version=$(python3 --version 2>&1 || echo "未安装")
    if [[ "$py_version" == Python* ]]; then
        check_pass "Python 版本检查" "$py_version"
    else
        check_fail "Python 版本检查" "Python3 未安装或不可用"
        return 1
    fi
    
    # 检查 2: 项目根目录验证（info）
    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        check_info "项目结构验证" "pyproject.toml 存在"
    else
        check_info "项目结构验证" "pyproject.toml 不存在（非 Python 项目或未初始化）"
    fi
    
    # 记录 minimal 模式总耗时
    local mode_end_ms
    mode_end_ms=$(now_ms)
    local mode_duration_ms=$((mode_end_ms - mode_start_ms))
    record_section_duration "轻量检查 (minimal)" "$mode_duration_ms"
}

# ============================================================
# 主程序
# ============================================================

main() {
    # JSON 模式不输出头部
    if [ "$JSON_OUTPUT" != true ]; then
        print_header "项目健康检查 - $(basename "$PROJECT_ROOT")"
        echo -e "  ${BLUE}时间:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
        echo -e "  ${BLUE}路径:${NC} $PROJECT_ROOT"
        if [ "$CI_MODE" = true ]; then
            echo -e "  ${BLUE}模式:${NC} CI"
        fi
        if [ "$FAIL_FAST" = true ]; then
            echo -e "  ${BLUE}选项:${NC} fail-fast"
        fi
    fi

    # ============================================================
    # 根据模式执行检查
    # ============================================================
    
    # 指定模式时仅运行对应检查
    if [ "$CHECK_MODE_SET" = true ]; then
        # 显示运行模式
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}运行模式:${NC} ${CHECK_MODES[*]}"
        fi
        
        # import 模式：语法 + 核心导入（对齐 import-test.yml）
        if should_run_mode "import"; then
            run_import_mode
        fi
        
        # lint 模式：flake8-critical + ruff + mypy（对齐 lint.yml）
        if should_run_mode "lint"; then
            run_lint_mode
        fi
        
        # test 模式：单元测试 + E2E（对齐 ci.yml）
        if should_run_mode "test"; then
            run_test_mode
        fi
        
        # minimal 模式：轻量检查（无额外依赖）
        if should_run_mode "minimal"; then
            run_minimal_mode
        fi
    else
        # 未指定模式时：执行所有检查（原有逻辑）
        run_check "Python 环境检查" check_python_version
        run_check "超时机制检查" check_timeout_backend
        run_check "依赖检查" check_dependencies
        run_check "依赖一致性检查" check_dependency_consistency
        run_check "Python 语法检查" check_syntax
        run_check "模块结构检查" check_imports
        run_check "目录结构检查" check_directories
        run_check "入口文件检查" check_main_entries
        run_check "运行模式检查" check_run_modes
        run_check "无 API Key Smoke 测试" check_no_api_key_smoke
        run_check "无 API Key Iterate 配置验证" check_no_api_key_iterate_config
        run_check "过时模式检测" check_deprecated_patterns
        run_check "配置文件检查" check_config
        run_check "Git 状态检查" check_git
        run_check "Agent CLI 检查" check_agent_cli
        run_check "知识库验证" check_knowledge_base
        run_check "预提交检查 (Python)" check_pre_commit

        # 可选的深度检查
        if [ "$FULL_CHECK" = true ]; then
            run_check "类型检查 (mypy)" check_type_hints
            run_check "代码风格检查" check_code_style
            run_check "格式检查" check_format
            # 综合：单元测试 + 覆盖率（一次完成），再跑一次 E2E（若无用例不失败）
            # --split-tests 用于排查/定位，避免覆盖率聚合带来的重复执行，split 模式下跳过覆盖率阈值检查
            if [ "$SPLIT_TESTS" = true ]; then
                if [ "$SPLIT_COVERAGE" = true ]; then
                    run_check "单元测试 + 覆盖率" check_unit_tests_with_coverage
                else
                    check_warn "--split-tests 模式下跳过覆盖率阈值检查（避免重复执行），建议单独再跑一次 --full 获取覆盖率"
                    run_check "单元测试" check_unit_tests
                fi
            else
                run_check "单元测试 + 覆盖率" check_unit_tests_with_coverage
            fi
            run_check "E2E 测试" check_e2e_tests
            
            # 网络隔离测试（--run-network-isolation 或 --full 时运行）
            if [ "$RUN_NETWORK_ISOLATION" = true ]; then
                run_check "网络隔离测试" check_network_isolation_tests
            fi
        else
            # 默认：快速 + 核心测试集合
            run_check "核心测试集合" check_core_tests
            
            # 即使不是 FULL_CHECK，如果显式指定 --run-network-isolation 也运行
            if [ "$RUN_NETWORK_ISOLATION" = true ]; then
                run_check "网络隔离测试" check_network_isolation_tests
            fi
            
            if [ "$JSON_OUTPUT" != true ]; then
                print_section "跳过的检查 (使用 --full 启用)"
                check_info "类型检查 (mypy)"
                check_info "代码风格检查 (flake8/ruff)"
                check_info "格式检查 (ruff format/isort)"
                check_info "单元测试 + 覆盖率 (pytest-cov, 阈值 ${COV_FAIL_UNDER}%)"
                check_info "E2E 测试 (-m e2e)"
                if [ "$RUN_NETWORK_ISOLATION" != true ]; then
                    check_info "网络隔离测试 (--run-network-isolation)"
                fi
            fi
        fi
    fi

    # 计算退出码
    local EXIT_CODE=0
    if [ $FAIL_COUNT -gt 0 ]; then
        EXIT_CODE=1
    fi

    # JSON 输出
    if [ "$JSON_OUTPUT" = true ]; then
        _CHECK_ALL_COMPLETED=true
        output_json_result $EXIT_CODE
        exit $EXIT_CODE
    fi

    # 汇总报告
    print_header "检查结果汇总"
    echo ""
    echo -e "  ${GREEN}✓ 通过:${NC} $PASS_COUNT"
    echo -e "  ${RED}✗ 失败:${NC} $FAIL_COUNT"
    echo -e "  ${YELLOW}⚠ 警告:${NC} $WARN_COUNT"
    echo -e "  ${BLUE}○ 跳过:${NC} $SKIP_COUNT"
    echo ""

    TOTAL=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT + SKIP_COUNT))

    # 成功时可清理日志（避免污染）
    if [ "$JSON_OUTPUT" != true ] && [ "$KEEP_LOGS" != true ] && [ "$LOG_DIR_SET" != true ] && \
       [ "$FAIL_COUNT" -eq 0 ] && [ "$LOG_DIR_CREATED" = true ]; then
        rm -rf "$LOG_DIR" 2>/dev/null || true
    fi

    _CHECK_ALL_COMPLETED=true
    if [ $FAIL_COUNT -eq 0 ]; then
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}${BOLD}  所有检查通过! 项目状态良好${NC}"
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        exit 0
    else
        echo -e "${RED}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}${BOLD}  发现 $FAIL_COUNT 个问题需要修复${NC}"
        echo -e "${RED}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        exit 1
    fi
}

main
