# 稀疏 vs dense attention 微基准：测内存流量物理，不是生产 kernel

# 用法：
#   python3 bench.py --selftest          # 先跑：k=N 必须还原 dense
#   python3 bench.py --smoke             # 本机亚分钟级 smoke
#   python3 bench.py                     # 全网格（云端跑）
#   python3 bench.py --out results.csv   # 结果追加到指定 CSV（断点续跑）

import argparse
import csv
import math
import os

import torch

# ---------- 字节模型（DSA 口径，fp8） ----------
# fp8 的字节数用 fp16 元素数模拟：元素维数 = 目标字节数 / 2
D_KV_BYTES = 576    # MLA KV：c_KV 512 + k_R 64
D_IDX_BYTES = 128   # Lightning Indexer 键
D_KV = D_KV_BYTES // 2    # fp16 元素数 288
D_IDX = D_IDX_BYTES // 2  # fp16 元素数 64

# 常见卡的理论带宽 GB/s，用于 sanity 标记（不是精确值，够判 30% 红线即可）
BW_PEAK = [
    ("3060", 360), ("3090", 936), ("4090", 1008),
    ("A100", 1555), ("H100", 3350), ("MI300", 5300), ("T4", 300), ("P100", 750),
    ("W7900", 960),  # Radeon Pro W7900D (gfx1100, 48GB)
]
BW_DEFAULT = 800

GRID_N = [4096, 8192, 16384, 32768, 65536, 131072]
GRID_K = [512, 1024, 2048]
GRID_B = [1, 8, 32]


def peak_bw(device_name: str) -> int:
    for key, bw in BW_PEAK:
        if key in device_name:
            return bw
    return BW_DEFAULT


# ---------- 两条 attention 路径 ----------

def dense_step(q, K, V):
    """dense：每步读全量 KV。q (B,1,D) / K,V (B,N,D)"""
    scores = torch.bmm(q, K.transpose(1, 2)) / math.sqrt(D_KV)
    attn = torch.softmax(scores, dim=-1)
    return torch.bmm(attn, V)


def sparse_step(q, K, V, q_idx, K_idx, k):
    """索引式稀疏：读 N×d_I（索引扫描）+ k×d_KV（gather 后的小 attention）"""
    # 1) 索引打分（便宜的小键，扫全量）
    scores_idx = torch.bmm(q_idx, K_idx.transpose(1, 2))            # (B,1,N)
    # 2) top-k 选择
    idx = torch.topk(scores_idx, k, dim=-1).indices.squeeze(1)        # (B,k)
    # 3) gather 主 KV（只读选中的 k 条）
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, D_KV)
    K_sel = K.gather(1, gather_idx)
    V_sel = V.gather(1, gather_idx)
    # 4) 小 attention
    scores = torch.bmm(q, K_sel.transpose(1, 2)) / math.sqrt(D_KV)
    attn = torch.softmax(scores, dim=-1)
    return torch.bmm(attn, V_sel)


# ---------- 计时 ----------

def time_path(fn, iters: int, warmup: int) -> float:
    """返回每步毫秒（median）。CUDA event 计时。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def alloc(N: int, B: int, device):
    """按字节模型建张量。generator 固定种子，同配置可复现。"""
    gen = torch.Generator(device="cpu").manual_seed(42 + N + B)
    K = torch.randn(B, N, D_KV, generator=gen, dtype=torch.float16).to(device)
    V = torch.randn(B, N, D_KV, generator=gen, dtype=torch.float16).to(device)
    K_idx = torch.randn(B, N, D_IDX, generator=gen, dtype=torch.float16).to(device)
    q = torch.randn(B, 1, D_KV, generator=gen, dtype=torch.float16).to(device)
    q_idx = torch.randn(B, 1, D_IDX, generator=gen, dtype=torch.float16).to(device)
    return q, K, V, q_idx, K_idx


# ---------- selftest：k=N 必须还原 dense（RED/GREEN 里的 GREEN 闸） ----------

def selftest(device) -> None:
    B, N, k = 2, 1024, 1024
    q, K, V, q_idx, K_idx = alloc(N, B, device)
    out_dense = dense_step(q, K, V)
    out_sparse = sparse_step(q, K, V, q_idx, K_idx, k)
    diff = (out_dense - out_sparse).abs().max().item()
    assert diff < 1e-3, f"selftest 失败：k=N 时 sparse 与 dense 最大差 {diff}"
    print(f"[selftest] PASS：k=N 还原 dense，最大差 {diff:.2e}")

    # 反证（RED 闸）：k < N 时输出必须不同，否则两条路径根本没分开
    out_small = sparse_step(q, K, V, q_idx, K_idx, k // 2)
    diff_small = (out_dense - out_small).abs().max().item()
    assert diff_small > 1e-3, "selftest 失败：k<N 时 sparse 竟然与 dense 相同，路径没生效"
    print(f"[selftest] PASS：k={k // 2} 时输出偏离 dense（差 {diff_small:.2e}），稀疏路径真实生效")


# ---------- 主扫参 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="本机小网格")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    device = "cuda"
    name = torch.cuda.get_device_name(0)
    bw_peak = peak_bw(name)
    print(f"device: {name}, 假设峰值带宽 {bw_peak} GB/s")

    if args.selftest:
        selftest(device)
        return

    # smoke：砍到 6GB 放得下、一分钟内跑完的最小网格
    n_list = [4096, 16384, 65536] if args.smoke else GRID_N
    k_list = [512, 2048] if args.smoke else GRID_K
    b_list = [1, 8] if args.smoke else GRID_B
    iters = 20 if args.smoke else args.iters

    header = ["device", "batch", "N", "k", "dense_ms", "sparse_ms",
              "measured_ratio", "predicted_ratio",
              "dense_gbps", "sparse_gbps", "bw_peak_assumed", "bw_ok"]
    exists = os.path.exists(args.out)
    # 断点续跑：跳过 CSV 里已有的 (batch, N, k) 组合
    done = set()
    if exists:
        with open(args.out) as f:
            for row in csv.DictReader(f):
                done.add((int(row["batch"]), int(row["N"]), int(row["k"])))

    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        for B in b_list:
            for N in n_list:
                q, K, V, q_idx, K_idx = alloc(N, B, device)
                dense_ms = time_path(lambda: dense_step(q, K, V), iters, args.warmup)
                for k in k_list:
                    if (B, N, k) in done:
                        continue
                    sparse_ms = time_path(
                        lambda: sparse_step(q, K, V, q_idx, K_idx, k), iters, args.warmup)
                    # 字节量（每步每层）：dense 全读 vs 稀疏 = 索引扫全量 + 取 k 条
                    dense_bytes = B * N * D_KV_BYTES
                    sparse_bytes = B * N * D_IDX_BYTES + B * k * D_KV_BYTES
                    dense_gbps = dense_bytes / dense_ms / 1e6
                    sparse_gbps = sparse_bytes / sparse_ms / 1e6
                    predicted = sparse_bytes / dense_bytes
                    measured = sparse_ms / dense_ms
                    bw_ok = dense_gbps > 0.3 * bw_peak
                    writer.writerow([name, B, N, k,
                                     f"{dense_ms:.4f}", f"{sparse_ms:.4f}",
                                     f"{measured:.4f}", f"{predicted:.4f}",
                                     f"{dense_gbps:.1f}", f"{sparse_gbps:.1f}",
                                     bw_peak, bw_ok])
                    f.flush()
                    print(f"B={B:>2} N={N:>6} k={k:>4}  dense={dense_ms:8.3f}ms "
                          f"sparse={sparse_ms:8.3f}ms  实测比 {measured:.3f} "
                          f"预测 {predicted:.3f}  dense_bw={dense_gbps:.0f}GB/s"
                          f"{'' if bw_ok else '  [!] 带宽利用率<30%，读数可疑'}")
                del q, K, V, q_idx, K_idx
                torch.cuda.empty_cache()
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
