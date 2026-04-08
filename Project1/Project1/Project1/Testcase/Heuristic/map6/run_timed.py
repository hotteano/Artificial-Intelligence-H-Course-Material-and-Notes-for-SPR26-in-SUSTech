#!/usr/bin/env python3
"""
计时脚本：在 map6 上运行启发式算法并记录时间（实时输出）
"""

import subprocess
import time
import sys
import os
import threading

# 配置
NETWORK_FILE = "dataset_tle"
SEED_FILE = "seed"
OUTPUT_FILE = "result.txt"
BUDGET = 1

# 启发式算法路径
HEURISTIC_SCRIPT = "../IEMP_Heur.py"

def stream_output(pipe, prefix, lines_buffer):
    """实时读取并打印输出"""
    for line in iter(pipe.readline, ''):
        timestamp = time.perf_counter() - start_global
        formatted_line = f"[{timestamp:8.2f}s] {prefix} {line.rstrip()}"
        print(formatted_line)
        lines_buffer.append(line)
    pipe.close()

def progress_reporter(stop_event):
    """每隔50秒打印一次进度"""
    while not stop_event.is_set():
        time.sleep(50)
        if not stop_event.is_set():
            elapsed = time.perf_counter() - start_global
            print(f"\n[*** PROGRESS ***] Elapsed: {elapsed:.2f}s - Still running...\n")

def run_with_timing():
    """运行启发式算法并实时计时"""
    global start_global

    print("=" * 70)
    print("Heuristic Algorithm Timer - Map6 (Real-time Output)")
    print("=" * 70)
    print(f"Network: {NETWORK_FILE}")
    print(f"Seed: {SEED_FILE}")
    print(f"Budget: {BUDGET}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)

    # 尝试查找正确的启发式脚本
    script_paths = [
        "../IEMP_Heur.py",
        "../../IEMP_Heur.py",
        "../../../Downloads/Project 1 - 12412115/IEMP_Heur.py",
        "../../../../Downloads/Project 1 - 12412115/IEMP_Heur.py",
    ]

    script_path = None
    for path in script_paths:
        if os.path.exists(path):
            script_path = path
            break

    if not script_path:
        print("Error: Cannot find IEMP_Heur.py")
        print("Searched paths:")
        for p in script_paths:
            print(f"  - {p} (exists: {os.path.exists(p)})")
        return False

    print(f"\nUsing script: {script_path}")

    # 构建命令
    cmd = [
        sys.executable,
        script_path,
        "-n", NETWORK_FILE,
        "-i", SEED_FILE,
        "-b", OUTPUT_FILE,
        "-k", str(BUDGET),
    ]

    print(f"Command: {' '.join(cmd)}")
    print("-" * 70)
    print("Starting execution...\n")

    # 记录总时间
    start_global = time.perf_counter()
    stdout_lines = []
    stderr_lines = []

    # 启动进度报告线程
    stop_progress = threading.Event()
    progress_thread = threading.Thread(target=progress_reporter, args=(stop_progress,))
    progress_thread.daemon = True
    progress_thread.start()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            universal_newlines=True
        )

        # 启动线程实时读取输出
        stdout_thread = threading.Thread(
            target=stream_output,
            args=(process.stdout, "[STDOUT]", stdout_lines)
        )
        stderr_thread = threading.Thread(
            target=stream_output,
            args=(process.stderr, "[STDERR]", stderr_lines)
        )

        stdout_thread.start()
        stderr_thread.start()

        # 等待进程完成
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()

        total_time = time.perf_counter() - start_global

        # 解析结果
        full_output = ''.join(stdout_lines)
        solve_time = None
        eval_time = None
        score = None

        for line in full_output.split('\n'):
            if 'solve=' in line.lower():
                try:
                    solve_time = float(line.split('solve=')[1].split()[0].replace('s', ''))
                except:
                    pass
            elif 'eval=' in line.lower():
                try:
                    eval_time = float(line.split('eval=')[1].split()[0].replace('s', ''))
                except:
                    pass
            elif 'score=' in line.lower():
                try:
                    score = float(line.split('score=')[1].split()[0])
                except:
                    pass

        print("\n" + "=" * 70)
        print("Final Timing Results")
        print("=" * 70)

        if solve_time and eval_time:
            print(f"  Solve time:   {solve_time:.4f}s")
            print(f"  Eval time:    {eval_time:.4f}s")
            print(f"  Algorithm:    {solve_time + eval_time:.4f}s")
        print(f"  Total time:   {total_time:.4f}s")

        if score:
            print(f"  Score:        {score:.4f}")

        print(f"  Return code:  {return_code}")
        print("=" * 70)

        stop_progress.set()
        return return_code == 0

    except KeyboardInterrupt:
        stop_progress.set()
        print("\n\n[!] Interrupted by user")
        return False

    except Exception as e:
        stop_progress.set()
        print(f"Error running heuristic: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_with_timing()
    sys.exit(0 if success else 1)
