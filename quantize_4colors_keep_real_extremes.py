#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def luminance(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def quantize_image_4colors(
    input_path,
    output_path,
    ignore_transparent=True,
):
    # ---------- 读取 ----------
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    h, w, _ = data.shape
    rgb = data[..., :3].reshape(-1, 3)
    alpha = data[..., 3].reshape(-1)

    # ---------- 仅使用非透明像素 ----------
    if ignore_transparent:
        mask = alpha > 0
        pixels = rgb[mask]
    else:
        pixels = rgb

    if len(pixels) == 0:
        raise RuntimeError("没有非透明像素")

    pixels_f = pixels.astype(np.float32)

    # ---------- 找真实最亮 / 最暗像素（锚点） ----------
    lums = np.array([luminance(c) for c in pixels_f])

    dark_anchor = pixels_f[np.argmin(lums)]
    bright_anchor = pixels_f[np.argmax(lums)]

    # ---------- 移除锚点像素 ----------
    def is_same(a, b):
        return np.all(a == b)

    middle_pixels = np.array(
        [p for p in pixels_f if not is_same(p, dark_anchor) and not is_same(p, bright_anchor)],
        dtype=np.float32
    )

    if len(middle_pixels) < 2:
        raise RuntimeError("中间像素不足，无法计算中间色")

    # ---------- 中间两色：KMeans(k=2) ----------
    kmeans = KMeans(
        n_clusters=2,
        n_init="auto",
        random_state=0,
    )
    kmeans.fit(middle_pixels)

    mid_centers = kmeans.cluster_centers_

    # 按亮度排序
    mid_lums = np.array([luminance(c) for c in mid_centers])
    mid_centers = mid_centers[np.argsort(mid_lums)]

    # ---------- 最终调色板 ----------
    palette = np.array(
        [dark_anchor, mid_centers[0], mid_centers[1], bright_anchor],
        dtype=np.float32
    )

    print("最终 4 色（从暗到亮，RGB）:")
    for i, c in enumerate(palette):
        print(f"{i}: {tuple(np.round(c).astype(int))}, luminance={luminance(c):.1f}")

    # ---------- 映射所有像素 ----------
    rgb_all = rgb.astype(np.float32)

    diff = rgb_all[:, None, :] - palette[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    labels = np.argmin(dist, axis=1)

    new_rgb = palette[labels].astype(np.uint8)

    # ---------- 还原 RGBA ----------
    out = np.zeros((h * w, 4), dtype=np.uint8)
    out[:, :3] = new_rgb
    out[:, 3] = alpha

    Image.fromarray(out.reshape(h, w, 4), "RGBA").save(output_path)
    print("完成")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法:")
        print("  python quantize_4colors_keep_real_extremes.py input.png output.png")
        sys.exit(1)

    quantize_image_4colors(sys.argv[1], sys.argv[2])

