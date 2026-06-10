"""
demo_retrieval_show.py — 图像检索演示（自动展示版）
随机选 4 张 query（不同 landmark），展示各自的 top-10 检索结果。
生成图片后自动打开显示。

用法：
    python demo_retrieval_show.py
"""

import os
import random
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ============ 配置 ============
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "demo_retrieval_show")
NUM_DEMO = 4
TOP_K = 10
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
LOCATIONS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

RETRIEVAL_JSON = os.path.join(
    os.path.dirname(__file__), "outputs", "retrieval_v2", "retrieval_results.json"
)

DATASET_ROOT = r"E:\playground\交大视觉印象数据集2026"
QUERY_DIR = os.path.join(DATASET_ROOT, "image_retrieval", "query")


def get_label(filepath):
    """提取landmark"""
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


def main():

    # 检查依赖
    print("\n 检查依赖")
    if not os.path.exists(RETRIEVAL_JSON):
        print(f"  没找到检索结果 : {RETRIEVAL_JSON}")
        return
    if not os.path.exists(QUERY_DIR):
        print(f"  没找到查询图片 : {QUERY_DIR}")
        return
    print("准备齐全")

    # 加载检索结果
    print("\n 加载检索结果")
    with open(RETRIEVAL_JSON, "r", encoding="utf-8") as f:
        retrieval_results = json.load(f)

    loc_queries = defaultdict(list)
    query_entries = []  # (filename, fullpath)
    for f_name in sorted(os.listdir(QUERY_DIR)):
        if Path(f_name).suffix in IMAGE_EXTS:
            fp = os.path.join(QUERY_DIR, f_name)
            if os.path.getsize(fp) > 1024:
                query_entries.append((f_name, fp))

    for i, (qname, qp) in enumerate(query_entries):
        if qname in retrieval_results:
            loc_queries[get_label(qp)].append(i)

    available = [loc for loc in LOCATIONS if loc in loc_queries and loc_queries[loc]]
    if len(available) < NUM_DEMO:
        print(f"  只有 {len(available)} landmarks可用")
        NUM_DEMO_actual = max(1, len(available))
    else:
        NUM_DEMO_actual = NUM_DEMO
    chosen_locs = random.sample(available, NUM_DEMO_actual)
    print(f"  选择 landmarks: {chosen_locs}")

    demo_queries = []
    for loc in chosen_locs:
        qi = random.choice(loc_queries[loc])
        qname, qp = query_entries[qi]
        demo_queries.append((loc, qname, qp))

    # 检索可视化
    NROWS, NCOLS = 2, 5
    TOTAL_COLS = 1 + NCOLS

    for loc, qname, qp in demo_queries:
        top_paths = retrieval_results.get(qname, [])[:TOP_K]
        actual_k = len(top_paths)

        fig = plt.figure(figsize=(24, 9))
        gs = fig.add_gridspec(NROWS, TOTAL_COLS, width_ratios=[1.8] + [1]*NCOLS,
                              hspace=0.15, wspace=0.08)

        ax_q = fig.add_subplot(gs[:, 0])
        try:
            qimg = Image.open(qp).convert("RGB")
            ax_q.imshow(qimg)
            ax_q.set_title(f"Query: {qname}\n(Landmark: {loc})",
                           fontsize=11, fontweight="bold")
        except Exception as e:
            ax_q.text(0.5, 0.5, f"Load Error\n{e}", ha="center", va="center",
                      fontsize=9, color="gray", transform=ax_q.transAxes)
        ax_q.axis("off")

        for rank in range(NROWS * NCOLS):
            row = rank // NCOLS
            col = rank % NCOLS + 1
            ax = fig.add_subplot(gs[row, col])
            if rank < actual_k:
                rp = top_paths[rank]
                try:
                    rimg = Image.open(rp).convert("RGB")
                    ax.imshow(rimg)
                    ax.set_title(f"#{rank+1}  {Path(rp).name[:20]}", fontsize=7)
                except Exception:
                    ax.text(0.5, 0.5, "Load Error", ha="center", va="center",
                            fontsize=7, color="gray", transform=ax.transAxes)
                    ax.set_title(f"#{rank+1}", fontsize=7)
                ax.axis("off")
            else:
                ax.axis("off")

        fig.suptitle(f"Image Retrieval Demo — {loc}",
                     fontsize=14, fontweight="bold", y=0.98)
        out_path = os.path.join(OUTPUT_DIR, f"demo_retrieval_{loc}.png")
        save_and_show(fig, out_path)

    print(f"\n {NUM_DEMO_actual} 图片已保存 ")


if __name__ == "__main__":
    random.seed(42)
    main()
