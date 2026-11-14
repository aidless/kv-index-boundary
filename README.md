# kv-index-boundary

稀疏注意力（索引式 top-k）与 dense attention 的读取量交叉边界，怎么算、在真显卡上落在哪。

起因是一条抖音视频的说法：Kimi K3.1 Pro 多存了一份索引，长上下文下相关注意力层的读取量反而少约四分之三。我们用字节模型把这笔账算出来，再在 T4、RTX 3060、Radeon Pro W7900D 三张卡上实测验证。完整的机制解释和实测叙事见 [blog.md](blog.md)。

## 核心结论

字节模型（fp8 口径）：每 token 每层存 576 B 主 KV + 128 B 索引键 = 704 B（存储 +22.2%）；decode 每步每层读 `d_i·N + d·k` 字节。N=128K、k=2048 时读取 18.0 MB vs dense 75.5 MB，**−76.2%**，即「约四分之三」。

理论交叉点：稀疏便宜过 dense 要求 `k < N·(1 − d_i/d) = 0.778·N`，即 k=2048 需 N > 2633。

实测（Kaggle T4，54 配置，`cloud-t4/`）：

- **P1 成立**：N=131072 九组实测比 0.181–0.227，预测 0.226–0.238，偏差 < 15%。
- **P2 方向成立、边界右移 6–12 倍**：k=2048 理论 2633、实测 16384 才反超；k=512 理论 658、实测 8192。朴素实现 8 次 kernel launch 的固定开销摊不平。
- **P4 成立**：比值 0.181–0.227 同窄带，不随 batch 变。
- **P3 部分成立**：k=2048 大 N 行三卡互差 < 10%（W7900D 0.206–0.215 / T4 0.216–0.227），临界点表逐格一致（8192/8192/16384）；但 k=512/1024 行 AMD 系统性低于 T4 8–20%（ROCm 的 gather 更快）→ 比值不是普适常数，它同时吃字节模型和 gather kernel 的相对效率。W7900D 上 42/54 行 bw_ok 未过（代理打不满 30% 红线），AMD 数值整体作参考档。

最重要的一条：**字节模型给出的交叉点是下界，真实边界是实现质量的函数**。vLLM/SGLang 在短上下文默认跳过 indexer，正是这条边界的工程体现。

## 复现

```bash
python3 bench.py --selftest    # 先过双向闸：k=N 还原 dense + k<N 真分叉
python3 bench.py --smoke       # 12 配置，~1 分钟
python3 bench.py --iters 50    # 全网格 54 配置；CSV 逐行 flush，中断可续跑
python3 analyze.py results.csv --plot figures
```

Kaggle 免费跑法（T4）：把本目录四个脚本上传为 dataset，notebook 里：

```python
!cp /kaggle/input/<dataset>/kv-index-boundary/*.py .
!python3 bench.py --selftest && python3 bench.py --iters 50
!python3 analyze.py results.csv --plot /kaggle/working/figures
```

`bw_ok=False` 的行是 launch 开销污染（带宽利用率 < 30% 假设峰值），不进判决。

## 证据分级

- **官方可查**：K3.1 Pro `config.json`——93 层中 24 层 MLA+DSA（每 4 层留 1 + 末层）、69 层 KDA；`kv_lora_rank=512 + rope 64 = 576`、`index_head_dim=128`、`index_topk=2048`，与本文账本口径逐项对上。技术报告 arXiv:2607.24653。
- **文献**：DeepSeek DSA/CSA 系列、GLM IndexCache 的公开账本。
- **本文自算自测**：交叉点公式、三卡实测、54 配置网格。

## 文件

- `bench.py` / `analyze.py` — 微基准与分析器
- `blog.md` — 机制解释成稿
- `DESIGN.md` — 设计文档（预注册预测、闸门）
- `cloud-t4/` — Kaggle T4 全网格结果与边界图
- `kaggle-dataset/`、`kaggle-kernel/` — Kaggle 打包件
- `cloud-amd/` — Radeon Pro W7900D（gfx1100）全网格结果
- `blog-zhihu.md` — 知乎发布版（图片占位 + 上传清单）

## 验收标准与复算

**验收判据**：本仓判为合格的条件是 `python3 eval/run_cases.py` 通过率必须为 100%（20/20），且 `make eval` 退出码必须为 0；任何一条失败即判为不合格。

`eval/cases.jsonl` 的 20 条题按 8 个 axis 分：

| axis | 条数 | 考什么 |
|---|---|---|
| `bytes/model` | 3 | 字节模型斜率 1−d_I/d、128K/k=2048 预测比、索引键带来的存储开销 |
| `bytes/crossover` | 2 | 理论临界 N*=k/0.7778（k=2048→2633、k=512→658） |
| `data/t4` | 3 | T4 全网格 54 行、带宽有效 27 行、N=131072 最小实测比 0.1813 |
| `data/amd` | 2 | W7900D 54 行、bw_ok 未过 42 行 |
| `data/local` | 3 | 本机 3060 12 行、带宽有效 6 行、B=8/N=65536/k=512 实测比 0.191 |
| `analyze/api` | 3 | `load()` 类型归一、`weight_kv_crossover()` 交叉点表、`boundary_table()` 临界点输出 |
| `analyze/errors` | 2 | 缺文件→`FileNotFoundError`、缺列→`KeyError` |
| `docs/claims` | 2 | README 的 −76.2% 与 DESIGN 的 N*≈2633 两处公开断言 |

```bash
python3 eval/run_cases.py            # 跑全部，实跑 20/20
python3 eval/run_cases.py --save     # 顺带重建 eval/run-report.json
python3 eval/run_cases.py --selftest # 篡改 EB-01 期望值，实跑 19/20，必须变红
```

**数字如何复算**：每题 `verify` 就是复算代码——`expr` 直接 `import analyze` 调本仓 API，或 `analyze.load()` 真解析 `results.csv` / `cloud-t4/results.csv` / `cloud-amd/results50.csv`，`expected` 是现算值；`raises` 断言真实异常类型；`doc` 断言文档原文正则。`eval/run-report.json` 每条带 `id` / `axis` / `detail`，重跑 `--save` 即可重建整份报告，不需要手抄任何数。

**失败归因**：`run_cases.py` 失败时按 axis 汇总（例如 `bytes/model: 1 条 —— EB-01`），而不是只丢一个 pass_rate。axis 对应模块：`bytes/*` 坏了 → `analyze.py` 的 `D_KV_BYTES` / `D_IDX_BYTES` / `SLOPE`；`data/*` 坏了 → 对应 CSV 被改动或缺失；`analyze/*` 坏了 → `analyze.py` 的 `load` / `boundary_table` / `weight_kv_crossover` 逻辑；`docs/claims` 坏了 → README 或 DESIGN 的公开数字被改。
