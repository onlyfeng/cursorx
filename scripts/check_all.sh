#!/bin/bash
# check_all.sh - 一键检查脚本
# 执行全面的项目健康检查
#
# 支持选项:
#   --ci          CI 模式（非交互式、无颜色输出）
#   --json        JSON 输出格式（便于 CI 解析）
#   --fail-fast   遇到失败立即退出
#   --full, -f    执行完整检查
#   --help, -h    显示帮助信息
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
        --split-tests)
            SPLIT_TESTS=true
            shift
            ;;
        --test-chunk-size)
            if [ -z "${2:-}" ]; then
                echo "参数错误: --test-chunk-size 需要数值"
                exit 1
            fi
            TEST_CHUNK_SIZE="$2"
            shift 2
            ;;
        --test-timeout)
            if [ -z "${2:-}" ]; then
                echo "参数错误: --test-timeout 需要数值 (秒)"
                exit 1
            fi
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --case-timeout)
            if [ -z "${2:-}" ]; then
                echo "参数错误: --case-timeout 需要数值 (秒)"
                exit 1
            fi
            CASE_TIMEOUT="$2"
            shift 2
            ;;
        --coverage-timeout)
            if [ -z "${2:-}" ]; then
                echo "参数错误: --coverage-timeout 需要数值 (秒)"
                exit 1
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
                echo "参数错误: --pytest-args 需要参数"
                exit 1
            fi
            PYTEST_ARGS_STR="$2"
            shift 2
            ;;
        --pytest-arg)
            if [ -z "${2:-}" ]; then
                echo "参数错误: --pytest-arg 需要参数"
                exit 1
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
                echo "参数错误: --log-dir 需要路径"
                exit 1
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
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --ci          CI 模式（非交互式、无颜色输出）"
            echo "  --json        JSON 输出格式（便于 CI 解析）"
            echo "  --fail-fast   遇到失败立即退出"
            echo "  -f, --full    执行完整检查（包括类型、风格和测试）"
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
            echo "  -h, --help    显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  CI=true       自动启用 CI 模式"
            echo "  NO_COLOR=1    禁用颜色输出"
            echo ""
            echo "示例:"
            echo "  $0                 快速检查"
            echo "  $0 --full          完整检查"
            echo "  $0 --full --split-tests --test-chunk-size 6"
            echo "  $0 --full --split-tests --test-timeout 600"
            echo "  $0 --full --coverage-timeout 900"
            echo "  $0 --full --split-tests --split-coverage"
            echo "  $0 --full --diagnose-hang"
            echo "  $0 --full --pytest-parallel 4"
            echo "  $0 --ci --json     CI 环境 JSON 输出"
            echo "  $0 --fail-fast     遇到失败立即退出"
            echo "  CI=true $0         自动 CI 模式"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 分组测试默认超时（避免卡死）
if [ "$SPLIT_TESTS" = true ] && [ "$TEST_TIMEOUT" -le 0 ]; then
    TEST_TIMEOUT=600
fi

# 完整检查默认启用卡住定位诊断（可用 --no-diagnose-hang 关闭）
if [ "$FULL_CHECK" = true ] && [ "$DIAGNOSE_HANG_SET" = false ]; then
    DIAGNOSE_HANG=true
fi

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

# JSON 字符串转义
json_escape() {
    echo "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g'
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
    python3 - <<'PY'
import json
import sys

text = sys.stdin.read()
for i in range(len(text) - 1, -1, -1):
    if text[i] != "{":
        continue
    try:
        data = json.loads(text[i:])
    except Exception:
        continue
    print(json.dumps(data, ensure_ascii=False))
    sys.exit(0)
sys.exit(1)
PY
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

# 测试分层（与 pyproject.toml 的 markers 定义保持一致）
UNIT_MARKER_EXPR="not e2e and not slow and not integration"
E2E_MARKER_EXPR="e2e and not slow"

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

    if [ ! -f "requirements.txt" ]; then
        check_fail "requirements.txt 不存在"
        return 1
    fi

    check_pass "requirements.txt 存在"

    # 使用 Python 直接导入检查（比 pip list 更快）
    DEPS_MISSING=0

    # 核心依赖 (包名:导入名)
    CORE_DEPS=(
        "pydantic:pydantic"
        "loguru:loguru"
        "pyyaml:yaml"
        "aiofiles:aiofiles"
        "beautifulsoup4:bs4"
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

    # 检查可选依赖
    OPTIONAL_DEPS=(
        "sentence-transformers:sentence_transformers"
        "chromadb:chromadb"
    )

    for dep in "${OPTIONAL_DEPS[@]}"; do
        pkg="${dep%%:*}"
        import_name="${dep##*:}"
        if python3 -c "import $import_name" 2>/dev/null; then
            check_pass "已安装 (可选): $pkg"
        else
            check_warn "未安装 (可选): $pkg"
        fi
    done

    if [ $DEPS_MISSING -gt 0 ]; then
        check_info "运行 'pip install -r requirements.txt' 安装依赖"
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
        if mypy core/ agents/ --ignore-missing-imports --no-error-summary 2>/dev/null; then
            check_pass "类型检查通过"
        else
            check_warn "类型检查有警告 (非致命)"
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
            check_warn "ruff 发现 $RUFF_ERRORS 个问题"
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
            python3 -m coverage report --fail-under=80 > "$COV_REPORT_FILE" 2>&1
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
            --cov=knowledge --cov=indexing --cov=process --cov-report=term-missing --cov-fail-under=80 \
            tests/
        COV_EXIT=$?
        set -e
        local duration_ms=$(( $(now_ms) - start_ms ))
        COV_OUTPUT=$(cat "$COV_OUTPUT_FILE" 2>/dev/null || echo "")
        COV_CMD_STR=$(format_pytest_command "${PYTEST_COMMON_ARGS[@]}" -m "$UNIT_MARKER_EXPR" \
            --cov=run --cov=core --cov=agents --cov=coordinator --cov=cursor --cov=tasks \
            --cov=knowledge --cov=indexing --cov=process --cov-report=term-missing --cov-fail-under=80 tests/)
        if [ "$COV_EXIT" -ne 0 ]; then
            COV_LAST_TEST=$(extract_last_test_case "$COV_OUTPUT_FILE")
            COV_META_JSON=$(build_meta_json "$COV_LAST_TEST" "$COV_CMD_STR")
        fi
    fi

    # 提取覆盖率百分比
    TOTAL_COV=$(echo "$COV_OUTPUT" | grep -E "^TOTAL" | awk '{print $NF}' | tr -d '%')

    if [ -n "$TOTAL_COV" ]; then
        if [ "$COV_EXIT" -eq 0 ]; then
            check_pass "单元测试通过，覆盖率: ${TOTAL_COV}% (阈值: 80%)" "" "$COV_OUTPUT_FILE" "$duration_ms"
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
                    check_fail "代码覆盖率不足: ${TOTAL_COV}% (阈值: 80%)" "coverage" "$COV_OUTPUT_FILE" "$duration_ms" "$COV_META_JSON"
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
        check_warn "未找到 E2E 测试用例（不计为失败）"
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
            check_warn "有 $CHANGES 个未提交的更改"
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
    if [ -f "scripts/pre_commit_check.py" ]; then
        check_pass "预提交检查脚本存在"

        # 运行预提交检查
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 运行预提交检查..."
        fi

        # 捕获输出并检查结果
        set +e
        PRE_COMMIT_OUTPUT=$(PYTHONPATH=. python3 scripts/pre_commit_check.py --json 2>&1)
        PRE_COMMIT_EXIT=$?
        set -e

        PRE_COMMIT_JSON=$(echo "$PRE_COMMIT_OUTPUT" | extract_json_from_output 2>/dev/null || echo "")

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
    else
        check_warn "预提交检查脚本不存在"
        check_info "运行 'python scripts/pre_commit_check.py' 进行检查"
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

    # 执行所有检查
    run_check "Python 环境检查" check_python_version
    run_check "超时机制检查" check_timeout_backend
    run_check "依赖检查" check_dependencies
    run_check "Python 语法检查" check_syntax
    run_check "模块结构检查" check_imports
    run_check "目录结构检查" check_directories
    run_check "入口文件检查" check_main_entries
    run_check "运行模式检查" check_run_modes
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
    else
        # 默认：快速 + 核心测试集合
        run_check "核心测试集合" check_core_tests
        if [ "$JSON_OUTPUT" != true ]; then
            print_section "跳过的检查 (使用 --full 启用)"
            check_info "类型检查 (mypy)"
            check_info "代码风格检查 (flake8/ruff)"
            check_info "单元测试 + 覆盖率 (pytest-cov, 阈值 80%)"
            check_info "E2E 测试 (-m e2e)"
        fi
    fi

    # 计算退出码
    local EXIT_CODE=0
    if [ $FAIL_COUNT -gt 0 ]; then
        EXIT_CODE=1
    fi

    # JSON 输出
    if [ "$JSON_OUTPUT" = true ]; then
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
