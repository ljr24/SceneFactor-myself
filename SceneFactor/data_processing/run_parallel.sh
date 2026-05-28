#!/bin/bash
# 并行运行 compute_sdf_3dfront_hjm.py
# 用法: bash run_parallel.sh [进程数]
# 默认 3 个进程并行

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scenefactor

NUM_PROC=${1:-3}

echo "启动 $NUM_PROC 个并行进程处理 SDF 计算..."

# 启动所有并行进程
for i in $(seq 0 $((NUM_PROC - 1))); do
    log_file="proc${i}.log"
    python -u compute_sdf_3dfront_hjm.py -n "$NUM_PROC" -p "$i" > "$log_file" 2>&1 &
    echo "  进程 $i 已启动 → $log_file (PID: $!)"
done

echo ""
echo "所有进程已在后台启动。"
echo "查看进度: tail -f proc0.log"
echo "等待全部完成: wait"

# 等待所有进程完成
wait
echo ""
echo "全部进程已完成！"
