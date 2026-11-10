# 分析：临界 N* 实测 vs 预测、边界图、权重/KV 带宽交叉点表

# 用法：python3 analyze.py results.csv [--plot out_dir]

import csv
import os
from collections import defaultdict

# 与 bench.py 一致的字节模型
D_KV_BYTES = 576
D_IDX_BYTES = 128
SLOPE = 1 - D_IDX_BYTES / D_KV_BYTES   # 0.778
LAYERS = 60                            # 层乘数（K3 量级）

def load(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["batch"] = int(row["batch"])
            row["N"] = int(row["N"])
            row["k"] = int(row["k"])
            row["dense_ms"] = float(row["dense_ms"])
            row["sparse_ms"] = float(row["sparse_ms"])
            row["measured_ratio"] = float(row["measured_ratio"])
            row["predicted_ratio"] = float(row["predicted_ratio"])
            row["dense_gbps"] = float(row["dense_gbps"])
            row["bw_ok"] = row["bw_ok"] == "True"
            rows.append(row)
    return rows

def boundary_table(rows):
    """每个 (k, B)：实测 sparse<dense 的最小 N，和理论 N* = k/SLOPE 对比
    只用 bw_ok=True 的行——launch 开销污染的读数不参与判决"""
    print("\n== 临界点：sparse 何时反超 dense（仅带宽有效行）==")
    print(f"{'k':>6} {'B':>3} {'实测N*':>8} {'理论N*':>8} {'偏差':>6}")
    groups = defaultdict(dict)   # (k,B) -> {N: measured_ratio}
    for r in rows:
        if not r["bw_ok"]:
            continue
        groups[(r["k"], r["batch"])][r["N"]] = r["measured_ratio"]
    for (k, B), by_n in sorted(groups.items()):
        ns = sorted(by_n)
        measured_star = next((n for n in ns if by_n[n] < 1.0), None)
        theo = k / SLOPE
        if measured_star is None:
            print(f"{k:>6} {B:>3} {'未达':>8} {theo:>8.0f} {'':>6}")
        else:
            dev = measured_star / theo
            mark = "*" if measured_star == max(ns) else ""   # * = 撞网格上限，删失值
            print(f"{k:>6} {B:>3} {measured_star}{mark:>1} {theo:>8.0f} {dev:>6.2f}x")

def ratio_convergence(rows):
    """P1 验证：大 N 处实测比是否收敛到预测比"""
    print("\n== P1：大 N 比值收敛（N>=65536 的行）==")
    print(f"{'k':>6} {'B':>3} {'N':>7} {'实测比':>7} {'预测比':>7}")
    for r in sorted(rows, key=lambda r: (r["k"], r["batch"], r["N"])):
        if r["N"] >= 65536:
            flag = "" if abs(r["measured_ratio"] - r["predicted_ratio"]) / r["predicted_ratio"] < 0.2 else "  [!]"
            print(f"{r['k']:>6} {r['batch']:>3} {r['N']:>7} "
                  f"{r['measured_ratio']:>7.3f} {r['predicted_ratio']:>7.3f}{flag}")

def bandwidth_check(rows):
    print("\n== 带宽 sanity（<30% 峰值的行数）==")
    bad = [r for r in rows if not r["bw_ok"]]
    print(f"共 {len(rows)} 行，可疑 {len(bad)} 行")
    if bad:
        for r in bad[:10]:
            print(f"  B={r['batch']} N={r['N']} k={r['k']} dense_bw={r['dense_gbps']:.0f}GB/s")

def weight_kv_crossover():
    """纯算术：batch 多大时 KV 读取带宽超过权重带宽（这 3/4 何时变成真收益）"""
    print("\n== 权重 vs KV 带宽交叉点（L=%d，d=%dB/层）==" % (LAYERS, D_KV_BYTES))
    print("每步 KV 读取 = B × L × d × N；超过权重读取 W 时 KV 成为瓶颈")
    print(f"{'W':>8} {'N':>8} {'B*':>6}")
    for W in (50e9, 100e9):
        for N in (32768, 131072, 1048576):
            b_star = W / (LAYERS * D_KV_BYTES * N)
            print(f"{W/1e9:>7.0f}G {N:>8} {b_star:>6.1f}")

def plot(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体：有 Noto CJK 就用，没有则回退英文标签
    from matplotlib import font_manager
    for font_path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                      "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Noto Sans CJK SC"
            break
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(out_dir, exist_ok=True)
    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        groups[r["batch"]][r["k"]][r["N"]] = r["measured_ratio"]
    for B, by_k in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, by_n in sorted(by_k.items()):
            ns = sorted(by_n)
            ax.plot(ns, [by_n[n] for n in ns], marker="o", label=f"top-k={k}")
        ax.axhline(1.0, color="black", ls="--", lw=1, label="sparse=dense")
        # 理论临界 N* 竖线
        for k, by_n in sorted(by_k.items()):
            ax.axvline(k / SLOPE, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("KV 长度 N（竖虚线 = 理论临界 k/0.778）")
        ax.set_ylabel("时间比 sparse/dense（<1 = 稀疏更快）")
        ax.set_title(f"batch={B}")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, f"boundary_B{B}.png")
        fig.savefig(path, dpi=150)
        print(f"图 -> {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", default="results.csv", nargs="?")
    parser.add_argument("--plot", default="figures")
    args = parser.parse_args()

    rows = load(args.csv_path)
    assert rows, "CSV 为空"
    bandwidth_check(rows)
    ratio_convergence(rows)
    boundary_table(rows)
    weight_kv_crossover()
    try:
        plot(rows, args.plot)
    except ImportError:
        print("[!] 无 matplotlib，跳过画图（pip install matplotlib 后重跑）")

if __name__ == "__main__":
    main()
