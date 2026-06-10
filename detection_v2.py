import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

# 配置
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "detection_v2")
YOLO_DATASET = os.path.join(OUTPUT_DIR, "yolo_dataset")

TRAIN_SPLIT = 0.8
YOLO_EPOCHS = 50
YOLO_IMSZ = 640
YOLO_BATCH = 16
DETECT_CONF = 0.3
LOCATIONS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]

QUERIES_PER_LOC = 2
TOP_K_RETRIEVAL  = 10

# 数据清洗
MIN_BOX_AREA_RATIO = 0.0005
MIN_BOX_SIDE_RATIO = 0.005
MAX_ASPECT_RATIO   = 20.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(YOLO_DATASET, exist_ok=True)

import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


DATASET_ROOT = r"E:\playground\交大视觉印象数据集2026"
DETECTION_SRC = os.path.join(DATASET_ROOT, "object_detection", "data")
QUERY_DIR = os.path.join(DATASET_ROOT, "image_retrieval", "query")

RETRIEVAL_JSON = os.path.join(os.path.dirname(__file__), "outputs", "retrieval_v2", "retrieval_results.json")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# 格式转换

def labelme_to_yolo(src_dir, output_dir, split_ratio):
    json_files = sorted(Path(src_dir).glob("*.json"))
    print(f"Found {len(json_files)} annotation files")

    pairs = []
    total_boxes_raw = 0
    total_boxes_kept = 0
    skipped_empty = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = None
        img_name = data.get("imagePath", "")
        if img_name:
            candidate = Path(src_dir) / img_name
            if candidate.exists():
                img_path = str(candidate)
        if not img_path:
            for ext in IMAGE_EXTS:
                c = Path(src_dir) / (jf.stem + ext)
                if c.exists():
                    img_path = str(c)
                    break
        if not img_path:
            print(f"  WARNING: No image found for {jf.name}, skipping")
            continue

        valid_shapes = []
        for s in data.get("shapes", []):
            if s.get("shape_type") != "rectangle":
                continue
            total_boxes_raw += 1
            pts = s["points"]
            if len(pts) < 2:
                continue
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            bw = abs(x2 - x1)
            bh = abs(y2 - y1)
            if bw < 2 or bh < 2:
                continue
            valid_shapes.append(s)
            total_boxes_kept += 1

        if not valid_shapes:
            skipped_empty += 1
            continue
        pairs.append((str(jf), img_path, data, valid_shapes))

    print(f"  Boxes: {total_boxes_raw} raw -> {total_boxes_kept} kept after initial filter")
    print(f"  Skipped {skipped_empty} images with no valid boxes; {len(pairs)} images remain")

    label2id = {"text": 0}
    print(f"  Classes: {list(label2id.keys())}")

    with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    for subset in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", subset), exist_ok=True)

    def convert(pair_list, subset):
        total = len(pair_list)
        print(f"  Converting {subset}: {total} samples...")
        filtered_box_count = 0
        for idx, (json_path, img_path, data, valid_shapes) in enumerate(pair_list):
            if idx % max(1, total // 10) == 0:
                print(f"    {idx+1}/{total}")

            try:
                img = Image.open(img_path)
                W, H = img.size
            except Exception:
                print(f"    Cannot read image: {img_path}, skipping")
                continue

            base_name = Path(img_path).stem
            txt_path = os.path.join(output_dir, "labels", subset, base_name + ".txt")

            kept_lines = []
            for s in valid_shapes:
                pts = s["points"]
                x1, y1 = pts[0]
                x2, y2 = pts[1]

                # 归一化到 [0, 1]
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = abs(x2 - x1) / W
                bh = abs(y2 - y1) / H
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0001, min(1.0, bw))
                bh = max(0.0001, min(1.0, bh))

                # 过滤：面积/边长太小 or 宽高比异常
                box_area_ratio = bw * bh
                min_side = min(bw, bh)
                aspect = max(bw, bh) / (min_side + 1e-8)

                if box_area_ratio < MIN_BOX_AREA_RATIO:
                    continue
                if min_side < MIN_BOX_SIDE_RATIO:
                    continue
                if aspect > MAX_ASPECT_RATIO:
                    continue

                kept_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                filtered_box_count += 1

            if kept_lines:
                with open(txt_path, "w") as f:
                    f.writelines(kept_lines)
            else:
                with open(txt_path, "w") as f:
                    pass

            dst_img = os.path.join(output_dir, "images", subset, Path(img_path).name)
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)

        print(f"    Kept {filtered_box_count} boxes after normalization filtering")

    convert(train_pairs, "train")
    convert(val_pairs, "val")

    # 写 data.yaml
    data_yaml = {
        "path": str(Path(output_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["text"],
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    return yaml_path, label2id


# YOLO训练

def train_yolo(data_yaml, epochs, imgsz, batch, device):
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=OUTPUT_DIR,
        name="train",
        exist_ok=True,
        label_smoothing=0.1,
        mosaic=0.3,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
    )


# 检测

def detect_on_images(model_path, image_paths, conf_thresh):
    """对图片列表做文字检测"""
    from ultralytics import YOLO

    model = YOLO(model_path)

    detections = {}
    total = len(image_paths)
    print(f"  Detecting on {total} images (conf={conf_thresh})...")
    for idx, fp in enumerate(image_paths):
        if idx % max(1, total // 10) == 0:
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
                    "conf": conf, "label": "text",
                })
        detections[name] = boxes
    return detections


#可视化

def visualize(query_detections, base_detections, output_dir):
    loc_queries = defaultdict(list)
    for qname in query_detections:
        prefix = qname.split("-")[0].lower() if "-" in qname else qname.lower()
        if prefix in LOCATIONS:
            loc_queries[prefix].append(qname)

    retrieval_results = {}
    if os.path.exists(RETRIEVAL_JSON):
        with open(RETRIEVAL_JSON, "r", encoding="utf-8") as f:
            retrieval_results = json.load(f)
        print(f"  加载模型: {RETRIEVAL_JSON}")
    else:
        print(f"  没有模型 {RETRIEVAL_JSON}")

    if not retrieval_results:
        print(" 不存在模型结果")

    NROWS, NCOLS = 2, 5
    assert NROWS * NCOLS >= TOP_K_RETRIEVAL

    total_saved = 0
    for loc in sorted(loc_queries.keys()):
        queries = loc_queries[loc][:QUERIES_PER_LOC]
        if not queries:
            continue

        for q_idx, qname in enumerate(queries):
            fig, axes = plt.subplots(NROWS, NCOLS, figsize=(20, 9))
            axes = axes.flatten()

            top_paths = retrieval_results.get(qname, [])[:TOP_K_RETRIEVAL]

            for i in range(len(axes)):
                ax = axes[i]
                if i < len(top_paths):
                    rp = top_paths[i]
                    rname = Path(rp).name
                    try:
                        rimg = Image.open(rp).convert("RGB")
                        ax.imshow(rimg)
                        if rname in base_detections:
                            for box in base_detections[rname]:
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
                    except Exception:
                        ax.text(0.5, 0.5, "Load Error", ha="center", va="center",
                                transform=ax.transAxes, fontsize=8, color="gray")
                        ax.set_title(rname[:30], fontsize=7)
                else:
                    ax.set_visible(False)
                ax.axis("off")

            qname_short = qname[:40] if len(qname) > 40 else qname
            fig.suptitle(f"Text Detection — {loc}  (query: {qname_short})",
                         fontsize=12, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            out_name = f"detection_{loc}_{q_idx+1}.png"
            plt.savefig(os.path.join(output_dir, out_name), dpi=150, bbox_inches="tight")
            plt.close()
            total_saved += 1
            print(f"  Saved: {out_name}")

    print(f"  可视化保存: {total_saved} ")



def main():
    # 格式转换
    print("\n格式转换")
    yaml_path = os.path.join(YOLO_DATASET, "data.yaml")
    if os.path.exists(yaml_path):
        with open(os.path.join(YOLO_DATASET, "classes.json"), "r", encoding="utf-8") as f:
            label2id = json.load(f)
    else:
        yaml_path, label2id = labelme_to_yolo(DETECTION_SRC, YOLO_DATASET, TRAIN_SPLIT)

    # 训练
    print("\n训练YOLOv8模型")
    best_pt = os.path.join(OUTPUT_DIR, "train", "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f" 模型已经存在: {best_pt}")
    else:
        train_yolo(yaml_path, YOLO_EPOCHS, YOLO_IMSZ, YOLO_BATCH, DEVICE)
        for i in range(1, 100):
            alt = os.path.join(OUTPUT_DIR, f"train{i}", "weights", "best.pt")
            if os.path.exists(alt):
                best_pt = alt
                print(f"  模型存在: {best_pt}")
                break

    if not os.path.exists(best_pt):
        print("不存在best_pt")
        return

    # 收集需要检测的图片
    print("\n收集需要检测的图片")

    query_paths = []
    for f in sorted(os.listdir(QUERY_DIR)):
        if Path(f).suffix in IMAGE_EXTS:
            query_paths.append(os.path.join(QUERY_DIR, f))
    print(f"  Query images: {len(query_paths)}")

    base_set = set()
    if os.path.exists(RETRIEVAL_JSON):
        with open(RETRIEVAL_JSON, "r", encoding="utf-8") as f:
            retrieval_data = json.load(f)
        for paths in retrieval_data.values():
            for p in paths:
                if os.path.exists(p):
                    base_set.add(p)
    base_paths = sorted(base_set)
    print(f"  Base  images: {len(base_paths)}")

    all_detect_paths = query_paths + base_paths

    # 检测
    print("\n检测")
    cache_all = os.path.join(OUTPUT_DIR, "detections_all.json")
    if os.path.exists(cache_all):
        with open(cache_all, "r", encoding="utf-8") as f:
            all_detections = json.load(f)
    else:
        all_detections = detect_on_images(best_pt, all_detect_paths, DETECT_CONF)
        with open(cache_all, "w", encoding="utf-8") as f:
            json.dump(all_detections, f, ensure_ascii=False, indent=2)

    query_detections = {}
    for qp in query_paths:
        qname = Path(qp).name
        query_detections[qname] = all_detections.get(qname, [])

    base_detections = {}
    for bp in base_paths:
        bname = Path(bp).name
        base_detections[bname] = all_detections.get(bname, [])

    q_has_text = sum(1 for v in query_detections.values() if v)
    b_has_text = sum(1 for v in base_detections.values() if v)
    print(f"  Query images : {q_has_text}/{len(query_detections)}")
    print(f"  Base  images : {b_has_text}/{len(base_detections)}")

    # 可视化
    print("\n可视化")
    visualize(query_detections, base_detections, OUTPUT_DIR)
    print(f"\n已保存: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
