# Kaggle script kernel：全网格扫参 + 分析，产物落 /kaggle/working
# selftest -> smoke(独立文件) -> 全网格 -> 分析+图
# dataset 挂载可能比 kernel 启动慢，递归查找 + 重试

import glob
import shutil
import subprocess
import sys
import time

def find_script(name: str) -> str:
    """在 /kaggle/input 下递归找脚本，dataset 处理慢就等（最多 5 分钟）"""
    deadline = time.time() + 300
    while time.time() < deadline:
        hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if hits:
            return hits[0]
        print(f"[wait] {name} 未挂载，10 秒后重试…", flush=True)
        time.sleep(10)
    sys.exit(f"5 分钟内没找到 {name}，dataset 挂载异常")

def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    ret = subprocess.run(cmd, shell=True).returncode
    if ret != 0:
        sys.exit(f"{cmd} 失败，退出码 {ret}")

for f in ("bench.py", "analyze.py"):
    shutil.copy(find_script(f), f)

# 1) selftest：不过就停，别浪费 GPU 时
run(f"{sys.executable} bench.py --selftest")
# 2) smoke 落独立文件，避免污染全网格的断点续跑跳表
run(f"{sys.executable} bench.py --smoke --out smoke.csv")
# 3) 全网格（54 配置，预计 5-10 分钟）
run(f"{sys.executable} bench.py --iters 50")
# 4) 分析 + 边界图（matplotlib Kaggle 自带）
run(f"{sys.executable} analyze.py results.csv --plot /kaggle/working/figures")

print("\n产物：/kaggle/working/ 下的 results.csv, smoke.csv, figures/", flush=True)
