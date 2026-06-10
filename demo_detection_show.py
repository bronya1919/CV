"""
demo_detection_show.py — 文字检测演示（自动展示版）
随机选 4 张 query（不同 landmark），展示各自的 top-10 检索结果 + 文字检测框。
生成图片后自动打开显示。

用法：
    python demo_detection_show.py
"""

import os
import random
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ============ 配置 ============
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "demo_detection_show")
NUM_DEMO = 4
TOP_K = 10
DETECT_CONF = 0.3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
LOCATIONS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 已训练的检测模型权重
BEST_PT = os.path.join(
    os.path.dirname(__file__), "outputs", "detection_v2", "train", "weights", "best.pt"
)
# 检索结果 JSON
RETRIEVAL_JSON = os.path.join(
    os.path.dirname(__file__), "outputs", "retrieval_v2", "retrieval_results.json"
)

DATASET_ROOT = r"E:\playground\交大视觉印象数据集2026"
QUERY_DIR = os.path.join(DATASET_ROOT, "image_retrieval", "query")


def get_label(filepath):
    """提取 landmark"""
    stem = Path(filepath).stem
    if "-" in stem:
        return stem.split("-")[0].lower()
    return stem.lower()


def save_and_show(fig, save_path):
    """保存图片并打开"""
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved & opened: {save_path}")
    os.startfile(save_path)


def detect_on_images(model_path, image_paths, conf_thresh):
    """对图片列表做文字检测"""
    from ultralytics import YOLO
    model = YOLO(model_path)
    detections = {}
    total = len(image_paths)
    print(f"  Detecting on {total} images (conf={conf_thresh})...")
    for idx, fp in enumerate(image_paths):
        if (idx + 1) % 10 == 0:
            print(f"    {idx+1}/{total}")
        results = model(fp, conf=conf_thresh, verbose=False)
        name = Path(fp).name
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                boxes.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "conf": conf,
                })
        detections[name] = boxes
    return detections


def main():

    # 检查依赖
    print("\n 检查依赖")
    if not os.path.exists(BEST_PT):
        print(f"  没有找到检测模型: {BEST_PT}")
        return
    if not os.path.exists(RETRIEVAL_JSON):
        print(f"  检索结果未找到 : {RETRIEVAL_JSON}")
        return
    print("模型和检索结果都存在")

    # 加载检索结果
    print("\n 加载模型")
    with open(RETRIEVAL_JSON, "r", encoding="utf-8") as f:
        retrieval_results = json.load(f)

    query_entries = []  # (filename, fullpath)
    for f_name in sorted(os.listdir(QUERY_DIR)):
        if Path(f_name).suffix in IMAGE_EXTS:
            fp = os.path.join(QUERY_DIR, f_name)
            if os.path.getsize(fp) > 1024:
                query_entries.append((f_name, fp))

    loc_queries = defaultdict(list)
    for i, (qname, qp) in enumerate(query_entries):
        if qname in retrieval_results:
            loc_queries[get_label(qp)].append(i)

    available = [loc for loc in LOCATIONS if loc in loc_queries and loc_queries[loc]]
    if len(available) < NUM_DEMO:
        print(f"  只有 {len(available)} landmarks 可用")
        NUM_DEMO_actual = max(1, len(available))
    else:
        NUM_DEMO_actual = NUM_DEMO
    chosen_locs = random.sample(available, NUM_DEMO_actual)
    print(f" 选择 landmarks: {chosen_locs}")

    demo_queries = []
    for loc in chosen_locs:
        qi = random.choice(loc_queries[loc])
        qname, qp = query_entries[qi]
        demo_queries.append((loc, qname, qp))

    # 收集需要检测的图片
    print("\n 收集需要检测的图片")
    all_base_set = set()
    for loc, qname, qp in demo_queries:
        top_paths = retrieval_results.get(qname, [])[:TOP_K]
        all_base_set.update(p for p in top_paths if os.path.exists(p))
    all_base_paths = sorted(all_base_set)
    print(f"  {len(all_base_paths)} 个检测图片")

    # 检测可视化
    detections = detect_on_images(BEST_PT, all_base_paths, DETECT_CONF)
    n_with_text = sum(1 for v in detections.values() if v)
    print(f"  检测到的含有文本的图片: {n_with_text}/{len(detections)}")

    print(f"\n 检测 {NUM_DEMO_actual} 图片")
    NROWS, NCOLS = 2, 5

    for loc, qname, qp in demo_queries:
        top_paths = retrieval_results.get(qname, [])[:TOP_K]
        actual_k = len(top_paths)

        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(20, 9))
        axes_flat = axes.flatten()
        fig.suptitle(f"Text Detection Demo — {loc}  (query: {qname[:50]})",
                     fontsize=12, fontweight="bold")

        for i in range(len(axes_flat)):
            ax = axes_flat[i]
            if i < actual_k:
                rp = top_paths[i]
                rname = Path(rp).name
                try:
                    rimg = Image.open(rp).convert("RGB")
                    ax.imshow(rimg)
                    if rname in detections:
                        for box in detections[rname]:
                            rect = patches.Rectangle(
                                (box["x1"], box["y1"]),
                                box["x2"] - box["x1"],
                                box["y2"] - box["y1"],
                                linewidth=2, edgecolor="red", facecolor="none",
                            )
                            ax.add_patch(rect)
                            ax.text(
                                box["x1"], max(box["y1"] - 5, 0),
                                "text", color="white", fontsize=7,
                                bbox=dict(facecolor="red", alpha=0.8, pad=1),
                            )
                    ax.set_title(rname[:30], fontsize=7)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Load Error", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="gray")
                    ax.set_title(rname[:30], fontsize=7)
            else:
                ax.set_visible(False)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(OUTPUT_DIR, f"demo_detection_{loc}.png")
        save_and_show(fig, out_path)

    print(f"\n {NUM_DEMO_actual} 图片已保存")


if __name__ == "__main__":
    random.seed(42)
    main()
