#!/usr/bin/env python3
"""并行运行 compute_sdf_3dfront_hjm.py 并统一显示进度。

用法:
    python run_parallel.py [进程数]

默认 3 个进程并行。
"""

import subprocess
import sys
import os
import time
import re
import signal

NUM_PROC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SCRIPT = "compute_sdf_3dfront_hjm.py"

# 总场景数（固定值，用于进度百分比估算）
TOTAL_SCENES = 50

processes = []
log_files = []

def cleanup(signum=None, frame=None):
    print("\n正在终止所有进程...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
    # 等待进程退出
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print(f"启动 {NUM_PROC} 个并行进程...")

for i in range(NUM_PROC):
    log_file = f"proc{i}.log"
    # 清空旧日志
    open(log_file, 'w').close()
    f = open(log_file, 'w')
    log_files.append(f)
    p = subprocess.Popen(
        [sys.executable, "-u", SCRIPT, "-n", str(NUM_PROC), "-p", str(i)],
        stdout=f, stderr=subprocess.STDOUT
    )
    processes.append(p)
    print(f"  进程 {i} (PID: {p.pid}) → {log_file}")

print("\n" + "=" * 70)

proc_start_time = time.time()

try:
    while True:
        # 检查是否所有进程都已结束
        running = [p for p in processes if p.poll() is None]
        if not running:
            break

        # 读取各进程日志的最新进度行
        lines = []
        for i in range(NUM_PROC):
            log_file = f"proc{i}.log"
            try:
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()
            except FileNotFoundError:
                all_lines = []

            # 从后往前找带场景名或 tqdm 进度条的行
            scene = ""
            progress = ""
            for line in reversed(all_lines):
                line = line.rstrip()
                if "开始处理场景:" in line:
                    scene = line.split("开始处理场景:")[-1].strip()[:20]
                    break
            for line in reversed(all_lines):
                if "%|" in line:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        progress = parts[0]  # e.g. "12%"
                    break

            status = "已完成" if processes[i].poll() is not None else "运行中"
            lines.append((i, status, scene, progress, len(all_lines)))

        # 统一显示
        status_line = ""
        done_count = 0
        total_done = 0
        for i, status, scene, progress, line_count in lines:
            if status == "已完成":
                done_count += 1
                marker = "✅"
                scene_text = ""
                progress_text = "100%"
            else:
                marker = "▶"
                scene_text = f" [{scene}]" if scene else ""
                progress_text = progress if progress else "?"

            status_line += f"进程{i} {marker} {progress_text}{scene_text}  "

        # 估算总进度
        try:
            for i in range(NUM_PROC):
                log_file = f"proc{i}.log"
                with open(log_file, 'r') as f:
                    content = f.read()
                # 统计已完成场景数（"开始处理场景:" 出现次数 - 正在处理的1个）
                done_scenes = content.count("开始处理场景:")
                if processes[i].poll() is not None:
                    total_done += done_scenes
                else:
                    total_done += max(0, done_scenes - 1)
        except:
            pass

        elapsed = time.time() - proc_start_time
        pct = total_done / TOTAL_SCENES * 100 if TOTAL_SCENES > 0 else 0

        print(f"\r[{total_done}/{TOTAL_SCENES} {pct:.0f}%  {elapsed/60:.0f}分] | {status_line}", end="", flush=True)

        time.sleep(2)

except KeyboardInterrupt:
    cleanup()

print("\n\n" + "=" * 70)
print("所有进程已完成!")
