import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "retrieval_v2")
BATCH_SIZE = 64
K_VALUES = [20, 40, 60]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
LOCATIONS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = r"E:\playground\交大视觉印象数据集2026"
QUERY_DIR   = os.path.join(DATASET_ROOT, "image_retrieval", "query")
BASE_BJTU   = os.path.join(DATASET_ROOT, "image_retrieval", "base", "BJTU")
BASE_UTIL   = os.path.join(DATASET_ROOT, "image_retrieval", "base", "util_pic")


# 工具函数

def get_label(filepath):
    stem = Path(filepath).stem
    if "-" in stem:
        return stem.split("-")[0].lower()
    return stem.lower()


def collect_images(*dirs):
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if Path(f).suffix in IMAGE_EXTS:
                fp = os.path.join(d, f)
                if os.path.getsize(fp) > 1024:  # skip empty/corrupt
                    paths.append(fp)
    return sorted(paths)


# 数据加载

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageListDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return IMG_TRANSFORM(img)


# 特征提取

@torch.no_grad()
def extract_features(image_paths):
    try:
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    except ImportError:
        model = models.resnet50(pretrained=True)

    model.fc = torch.nn.Identity()
    model = model.to(DEVICE).eval()

    dataset = ImageListDataset(image_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    features = []
    total_batches = len(loader)
    print(f"  Processing {total_batches} batches...")
    for i, batch in enumerate(loader):
        if i % max(1, total_batches // 10) == 0:
            print(f"    Batch {i+1}/{total_batches}")
        feat = model(batch.to(DEVICE))
        feat = F.normalize(feat, p=2, dim=1)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


# 检索

def search(query_feat, base_feat):
    sim = query_feat @ base_feat.T
    return np.argsort(-sim, axis=1)


# 评估与输出
def evaluate(order, query_paths, base_paths):
    query_labels = [get_label(p) for p in query_paths]
    base_labels  = [get_label(p) for p in base_paths]

    loc_queries = defaultdict(list)
    for i, lbl in enumerate(query_labels):
        loc_queries[lbl].append(i)

    max_k = max(K_VALUES)
    p_at_k = defaultdict(lambda: defaultdict(list))

    for loc, q_indices in loc_queries.items():
        for qi in q_indices:
            top_k = order[qi, :max_k]
            q_label = query_labels[qi]
            for k in K_VALUES:
                cnt = sum(1 for idx in top_k[:k] if base_labels[idx] == q_label)
                p_at_k[loc][k].append(cnt / k)

    for loc in LOCATIONS:
        if loc not in p_at_k:
            continue
        plt.figure(figsize=(6, 4))
        xs = K_VALUES
        ys = [np.mean(p_at_k[loc][k]) for k in K_VALUES]
        plt.plot(xs, ys, "o-", linewidth=2, markersize=8)
        plt.xlabel("K")
        plt.ylabel("Precision@K")
        plt.title(f"P@K — {loc}")
        plt.xticks(K_VALUES)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"P@K_{loc}.png"), dpi=150)
        plt.close()

    retrieval_results = {}
    for loc, q_indices in loc_queries.items():
        for qi in q_indices[:5]:
            q_name = Path(query_paths[qi]).name
            top10 = order[qi, :10]
            retrieval_results[q_name] = [base_paths[i] for i in top10]

    result_path = os.path.join(OUTPUT_DIR, "retrieval_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
    print(f"  Retrieval results saved to: {result_path}")

    print("\n P@K ")
    pad = max(len(loc) for loc in LOCATIONS if loc in p_at_k)
    for loc in LOCATIONS:
        if loc in p_at_k:
            vals = "  ".join(f"K={k}:{np.mean(p_at_k[loc][k]):.3f}" for k in K_VALUES)
            print(f"  {loc:<{pad}}  {vals}")



def main():
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_ROOT}")

    print("\n[1/4] 加载图片")
    base_paths  = collect_images(BASE_BJTU, BASE_UTIL)
    query_paths = collect_images(QUERY_DIR)
    print(f"  Base  images: {len(base_paths)}")
    print(f"  Query images: {len(query_paths)}")

    print("\n提取特征")
    cache_base  = os.path.join(OUTPUT_DIR, "base_features.npy")
    cache_query = os.path.join(OUTPUT_DIR, "query_features.npy")

    if os.path.exists(cache_base) and os.path.exists(cache_query):
        base_feat  = np.load(cache_base)
        query_feat = np.load(cache_query)
    else:
        base_feat  = extract_features(base_paths)
        query_feat = extract_features(query_paths)
        np.save(cache_base, base_feat)
        np.save(cache_query, query_feat)

    print(f"  base  features: {base_feat.shape}")
    print(f"  query features: {query_feat.shape}")

    print("\n检索")
    order = search(query_feat, base_feat)

    print("\n评价")
    evaluate(order, query_paths, base_paths)
    print(f"\n已保存: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
