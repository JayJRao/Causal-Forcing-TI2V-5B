#!/bin/bash

VIDEOS_DIR="${1:-videos}"
SHOW_ALL=0
[ "${2}" = "-a" ] && SHOW_ALL=1
RED='\033[31m'
RESET='\033[0m'

if [ ! -d "$VIDEOS_DIR" ]; then
    echo "目录不存在: $VIDEOS_DIR"
    exit 1
fi

# 根据子目录名判断预期数量
expected_count() {
    local subname="$1"
    case "$subname" in
        demos*)               echo 100 ;;
        all_dimension_extended*) echo 946 ;;
        *vbench*)             echo 30  ;;
        *gemini*)             echo 20  ;;
        *)                    echo -1  ;;  # 未知类型，不检查
    esac
}

# 收集所有行到数组，用于对齐
rows=()

for subdir in "$VIDEOS_DIR"/*/; do
    [ -d "$subdir" ] || continue
    model=$(basename "$subdir")

    for subsubdir in "$subdir"*/; do
        [ -d "$subsubdir" ] || continue
        subname=$(basename "$subsubdir")

        for leaf in "$subsubdir"*/; do
            [ -d "$leaf" ] || continue
            leafname=$(basename "$leaf")
            count=$(find "$leaf" -name "*.mp4" | wc -l)
            rows+=("$model|$subname/$leafname|$count")
        done
    done
done

# 计算各列最大宽度
w1=0; w2=0
for row in "${rows[@]}"; do
    col1=$(echo "$row" | cut -d'|' -f1)
    col2=$(echo "$row" | cut -d'|' -f2)
    [ ${#col1} -gt $w1 ] && w1=${#col1}
    [ ${#col2} -gt $w2 ] && w2=${#col2}
done

# 打印表头
echo "统计 $VIDEOS_DIR 下 mp4 文件数量："
printf "%-${w1}s  %-${w2}s  %s\n" "模型" "子目录" "mp4数量"
printf '%0.s-' $(seq 1 $((w1 + w2 + 12))); echo

# 打印数据
total=0
prev_model=""
for row in "${rows[@]}"; do
    col1=$(echo "$row" | cut -d'|' -f1)
    col2=$(echo "$row" | cut -d'|' -f2)
    col3=$(echo "$row" | cut -d'|' -f3)
    expected=$(expected_count "$col2")
    is_bad=0
    [ "$expected" -ne -1 ] && [ "$col3" -ne "$expected" ] && is_bad=1
    [ "$is_bad" -eq 0 ] && [ "$SHOW_ALL" -eq 0 ] && continue
    if [ -n "$prev_model" ] && [ "$col1" != "$prev_model" ]; then
        echo
    fi
    prev_model="$col1"
    if [ "$is_bad" -eq 1 ]; then
        printf "${RED}%-${w1}s  %-${w2}s  %d${RESET}\n" "$col1" "$col2" "$col3"
    else
        printf "%-${w1}s  %-${w2}s  %d\n" "$col1" "$col2" "$col3"
    fi
    total=$((total + col3))
done

printf '%0.s-' $(seq 1 $((w1 + w2 + 12))); echo
echo "总计: $total 个 mp4 文件"
