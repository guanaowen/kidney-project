import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 用户参数
# ============================================================
FRAME_INTERVAL_S = 10          # 每帧 10 秒
VOLUME_PER_PIXEL_UL = 0.00535    # 每像素体积 (uL)
MIN_CONSECUTIVE_255 = 5        # 连续 n 个 255 才认定液体前缘 [1]
INJECTION_FRAME = 60          # x：DCZ 注射结束的帧号
ZERO_AT_T0 = True              # 归零：减去第一帧（x1）的体积
SMOOTH_WINDOW = 6

GROUP  = "Calb1"
SAMPLE = "S01"

# ============================================================
# 1. 读入数字矩阵（行为帧数，列为轨迹像素）
#    如果已是 csv：mat = pd.read_csv("xxx.csv", header=None).values
# ============================================================
mat = pd.read_csv(r"D:\Bailab\1kidney\BEHAVIOR\ureteral catheterization_analysed\26-8-25\2026-08-25 10-08-34\newMASK_new2026-08-25 10-08-34.tif (Projected Kymograph).csv", header=None).values   # ← 改成你的文件

# --- 时间方向检查：255 多的行应是后面的帧 ---
n_top, n_bottom = (mat[0] == 255).sum(), (mat[-1] == 255).sum()
if n_bottom > n_top:
    mat = np.flipud(mat)          # 使 row 0 = 第一帧 (t=0)

n_frames, n_pixels = mat.shape
print(f"矩阵: {n_frames} 帧 × {n_pixels} 列")

# ============================================================
# 2. 逐帧提取液体前缘（连续 n 个 255 的第一个 255 所在列）[1]
# ============================================================
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

# --- 回退校正：前缘不应倒流，下降时沿用前一帧 [1] ---
for i in range(1, len(front)):
    if np.isnan(front[i]) or front[i] < front[i-1]:
        front[i] = front[i-1]

# ============================================================
# 3. 像素 → 体积 (µL)，归零 [1]
# ============================================================
volume_ul = front * VOLUME_PER_PIXEL_UL
if ZERO_AT_T0:
    volume_ul = volume_ul - volume_ul[0]             # t=0 (x1) 归零

time_s = np.arange(n_frames) * FRAME_INTERVAL_S

# ============================================================
# 4. 流量 (µL/min)：体积对时间差分 + 滑动平均
# ============================================================
flow = np.diff(volume_ul, prepend=volume_ul[0]) / (FRAME_INTERVAL_S / 60.0)
if SMOOTH_WINDOW > 1:
    flow = pd.Series(flow).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().values

# ============================================================
# 5. 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time_s, volume_ul, lw=1.5, color="steelblue")
ax.axvline(INJECTION_FRAME * FRAME_INTERVAL_S, color="red", ls="--", lw=1.5,
           label="DCZ injection end")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Volume (µL)")
ax.set_title("Volume–Time Curve")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("volume_time.png", dpi=300)
plt.show()

# ============================================================
# 6. 保存结果
# ============================================================

from pathlib import Path

csv_path = Path(r"D:\Bailab\1kidney\BEHAVIOR\ureteral catheterization_analysed\26-8-25\2026-08-25 10-08-34\new2026-08-25 10-08-34.tif (Projected Kymograph).csv")
output_dir = csv_path.parent          # csv 所在文件夹

out.to_csv(output_dir / f"{SAMPLE}_results.csv", index=False, encoding="utf-8-sig")
plt.savefig(output_dir / f"{SAMPLE}_volume_flow.png", dpi=300, bbox_inches="tight")
print(f"已保存到: {output_dir}")