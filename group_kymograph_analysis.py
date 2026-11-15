import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 用户参数
# ============================================================
FRAME_INTERVAL_S    = 10        # 每帧 10 秒 [1]
MAX_PLOT_TIME_S = 1800      # 绘图最大时间：1800 秒（30 min）
VOLUME_PER_PIXEL_UL = 0.00535   # 每像素体积 (µL) = 5.35 nL
MIN_CONSECUTIVE_255 = 5         # 连续 n 个 255 才认定液体前缘 [1]
INJECTION_FRAME     = 60        # x：DCZ 注射结束的帧号（各样品相同时用全局值）
ZERO_AT_T0          = True      # 归零：减去第一帧（x1）的体积
SMOOTH_WINDOW       = 6         # 流量平滑窗口（帧），6帧=1min

# ============================================================
# 多文件输入：每个 csv 一行 (路径, 组别, 样品编号)
# ============================================================
FILES = [
    (r"D:\Bailab\1kidney\BEHAVIOR\ureteral catheterization_analysed\26-8-25\2026-08-25 10-08-34\newMASK_new2026-08-25 10-08-34.tif (Projected Kymograph).csv", "Ctrl", "GAW228"),
    (r"D:\Bailab\1kidney\BEHAVIOR\ureteral catheterization_analysed\26-8-25\test\2026-08-25 10-49-59\MASK_2026-08-25 10-49-59.tif (Projected Kymograph).csv", "Calb1", "GAW207"),
    (r"D:\Bailab\1kidney\BEHAVIOR\ureteral catheterization_analysed\26-8-25\test\2026-08-25 11-24-38\MASK_2026-08-25 11-24-38.tif (Projected Kymograph).csv", "Calb1",  "GAW217"),
    (r"D:\Bailab\1kidney\BEHAVIOR\ureteral catheterization_analysed\26-8-25\test\2026-08-25 13-09-42\MASK_2026-08-25 13-09-1.tif (Projected Kymograph).csv", "Calb1",  "GAW213"),
    (r"D:\Bailab\1kidney\BEHAVIOR\ureteral catheterization_analysed\26-8-25\2026-08-25 13-52-08\MASK_2026-08-25 13-52-08.tif (Projected Kymograph).csv","Ctrl", "GAW234"),
    (r"D:\Bailab\1kidney\BEHAVIOR\ureteral catheterization_analysed\26-8-25\test\2026-08-25 16-31-07\MASK_2026-08-25 16-31-07 (Projected Kymograph).csv", "Calb1", "GAW220"),
     #继续添加更多样品...
]

GROUP_COLORS = {"Calb1": "tab:red", "Ctrl": "tab:blue"}   # 组别 → 颜色
LINE_ALPHA   = 0.7        # 单条曲线透明度，多条重叠时更易分辨

# ============================================================
# 处理函数（单样品）
# ============================================================
def process_one(csv_path, group, sample):
    """读入单个样品的 kymograph csv，返回该样品的逐帧结果 DataFrame"""
    mat = (pd.read_csv(csv_path, header=None)
           .apply(pd.to_numeric, errors="coerce")
           .dropna(axis=0, how="all")
           .dropna(axis=1, how="all")
           .values)
    mat = (mat > 0).astype(int) * 255          # 强制二值化兜底

    # 时间方向检查 [1]
    if (mat[-1] == 255).sum() > (mat[0] == 255).sum():
        mat = np.flipud(mat)

    n_frames = mat.shape[0]

    # 前缘提取：连续 n 个 255 的第一个 255 所在列 [1]
    def front_position(row, n):
        run = 0
        for col, val in enumerate(row):
            if val == 255:
                run += 1
                if run >= n:
                    return col - n + 1
            else:
                run = 0
        return np.nan

    front = np.array([front_position(row, MIN_CONSECUTIVE_255) for row in mat])

    # 回退校正：下降时沿用前一帧 [1]
    for i in range(1, len(front)):
        if np.isnan(front[i]) or front[i] < front[i-1]:
            front[i] = front[i-1]

    # 体积换算 + 归零 [1]
    volume_ul = front * VOLUME_PER_PIXEL_UL
    if ZERO_AT_T0:
        volume_ul = volume_ul - volume_ul[0]

    time_s = np.arange(n_frames) * FRAME_INTERVAL_S
    flow = np.diff(volume_ul, prepend=volume_ul[0]) / (FRAME_INTERVAL_S / 60.0)
    if SMOOTH_WINDOW > 1:
        flow = pd.Series(flow).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().values

    return pd.DataFrame({
        "Frame":           np.arange(n_frames),
        "Time_s":          time_s,
        "Front_pixel":     front,
        "Volume_uL":       volume_ul,
        "Flow_uL_per_min": flow,
        "Group":           group,
        "Sample":          sample,
    })

# ============================================================
# 主流程：循环处理所有文件
# ============================================================
all_results = []
for path_str, group, sample in FILES:
    csv_path = Path(path_str)
    if not csv_path.exists():
        print(f"[跳过] 文件不存在: {csv_path}")
        continue
    df = process_one(csv_path, group, sample)
    all_results.append(df)
    print(f"[完成] {sample} ({group}): {df.shape[0]} 帧")

results = pd.concat(all_results, ignore_index=True)

# 保存合并结果到第一个样品的文件夹（或改成你想要的汇总目录）
output_dir = Path(FILES[0][0]).parent
results.to_csv(output_dir / "all_samples_results.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 绘图：同一坐标轴，按组别分色
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

for group, color in GROUP_COLORS.items():
    sub = results[results["Group"] == group]
    for sample in sub["Sample"].unique():
        d = sub[sub["Sample"] == sample]
        ax.plot(d["Time_s"], d["Volume_uL"], color=color, alpha=LINE_ALPHA, lw=1.5)

# 组别图例（每组合并成一条图例项）
from matplotlib.lines import Line2D
legend_items = [Line2D([0], [0], color=c, lw=2, label=g) for g, c in GROUP_COLORS.items()]
ax.legend(handles=legend_items)

# DCZ 注射结束红线（各样品时间轴一致时可画一条）
ax.axvline(INJECTION_FRAME * FRAME_INTERVAL_S, color="red", ls="--", lw=1.5, label="DCZ injection end")
# 限制绘图时间范围：0–1800 s
ax.set_xlim(0, MAX_PLOT_TIME_S)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Volume (µL)")
ax.set_title("Volume–Time Curves by Group")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(output_dir / "volume_time_by_group.png", dpi=300, bbox_inches="tight")
plt.show()

print("全部完成")