# kv-index-boundary

稀疏注意力（索引式 top-k）vs dense 的边界曲线实验。机制解释见 `DESIGN.md`。

## 本机 smoke（亚分钟级）

```bash
cd kv-index-boundary
python3 bench.py --selftest    # 必须先过：k=N 还原 dense + k<N 真的分叉
python3 bench.py --smoke       # 小网格，~1 分钟
python3 analyze.py results.csv
```

## 云端全网格

### Kaggle（T4 16GB，免费）

Notebook cell：

```python
!cp /kaggle/input/<你的dataset>/kv-index-boundary/*.py .
!python3 bench.py --selftest && python3 bench.py --iters 50
!python3 analyze.py results.csv --plot /kaggle/working/figures
```

预计 5–8 分钟（54 配置，dense 大 N 行最重）。`results.csv` 逐行 flush，
中断后把 CSV 带回去重跑即可续（已测组合自动跳过）。

### AMD MI300X 实例（ssh，验证 P3 跨硬件一致性）

```bash
ssh <amd-instance>
cd ~/dsh-scratch/<jobId>
python3 bench.py --selftest && python3 bench.py --iters 30
python3 analyze.py results.csv
# 拉回：把 results.csv、figures/ 拷回本地 analyses/remote/<jobId>/
```

## 判读

- `bw_ok=False` 的行丢弃重测（launch 开销污染，不是带宽-bound）。
- P1：大 N 实测比 ≈ 预测比（±20%）→ 字节模型成立。
- P2：实测临界 N* ≈ k/0.778（±1.5×内）→ 交叉点公式成立。
- P3：T4 与 MI300X 的比值曲线重合 → 结论可跨硬件引用。
- 附加：`analyze.py` 末尾的权重/KV 交叉点表回答「batch 多大时这 3/4
  才变成端到端收益」。

## 下一步（博客）

分析输出全部来自 CSV，不手打数字。骨架：
机制（4 个杠杆表）→ 字节模型 → 边界图（本目录产物）→
交叉点表 → 简化声明（traffic-proxy，生产 kernel 只会更差）。
