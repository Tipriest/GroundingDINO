#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
from glob import glob
from typing import Dict, List, Tuple

import cv2
import yaml
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict, annotate


def load_config(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


def collect_images(image_dir: str, recursive: bool, extensions: List[str]) -> List[str]:
    paths: List[str] = []
    for ext in extensions:
        pattern = os.path.join(image_dir, "**", f"*.{ext}") if recursive else os.path.join(image_dir, f"*.{ext}")
        paths.extend(glob(pattern, recursive=recursive))
    return sorted(set(paths))


def split_list(items: List[str], ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if ratio <= 0 or ratio >= 1 or not items:
        return items, []
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    split_idx = int(len(items) * (1 - ratio))
    return items[:split_idx], items[split_idx:]


def write_yolo_labels(label_path: str, class_ids: List[int], boxes: List[Tuple[float, float, float, float]]) -> None:
    lines = []
    for cls_id, (cx, cy, w, h) in zip(class_ids, boxes):
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def place_image(src: str, dst: str, copy_images: bool, use_symlinks: bool) -> None:
    if os.path.exists(dst):
        return
    ensure_dir(os.path.dirname(dst))
    if use_symlinks:
        os.symlink(src, dst)
    elif copy_images:
        shutil.copy2(src, dst)


def normalize_box(box: List[float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = box
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return cx, cy, w, h


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GroundingDINO to YOLO dataset")
    parser.add_argument("-c", "--config", default="./groundingdino_to_yolo.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("inference", {})
    yolo_cfg = cfg.get("yolo", {})

    model_config_path = model_cfg.get("config_path", "groundingdino/config/GroundingDINO_SwinB_cfg.py")
    weights_path = model_cfg.get("weights_path", "weights/groundingdino_swinb_cogcoor.pth")
    device = model_cfg.get("device", "cuda")

    class_names = infer_cfg.get("class_names", [])
    if not class_names:
        raise ValueError("inference.class_names is required")
    class_names = [str(x).strip().lower() for x in class_names]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    box_threshold = float(infer_cfg.get("box_threshold", 0.35))
    text_threshold = float(infer_cfg.get("text_threshold", 0.25))

    image_dir = data_cfg.get("image_dir")
    train_dir = data_cfg.get("train_images")
    val_dir = data_cfg.get("val_images")
    output_dir = data_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("data.output_dir is required")

    recursive = bool(data_cfg.get("recursive", False))
    extensions = data_cfg.get("extensions", ["jpg", "jpeg", "png", "bmp"])
    copy_images = bool(data_cfg.get("copy_images", True))
    use_symlinks = bool(data_cfg.get("use_symlinks", False))
    skip_existing = bool(data_cfg.get("skip_existing", True))
    split_ratio = float(data_cfg.get("split_ratio", 0.0))
    split_seed = int(data_cfg.get("split_seed", 42))
    save_visualizations = bool(data_cfg.get("save_visualizations", False))
    vis_dir = data_cfg.get("vis_dir", "visualizations")

    if image_dir:
        all_images = collect_images(image_dir, recursive, extensions)
        if not all_images:
            raise ValueError(f"No images found in {image_dir}")
        train_images, val_images = split_list(all_images, split_ratio, split_seed)
    else:
        if not train_dir:
            raise ValueError("data.image_dir or data.train_images is required")
        train_images = collect_images(train_dir, recursive, extensions)
        if not train_images:
            raise ValueError(f"No images found in {train_dir}")
        if val_dir:
            val_images = collect_images(val_dir, recursive, extensions)
        else:
            train_images, val_images = split_list(train_images, split_ratio, split_seed)

    output_images_train = os.path.join(output_dir, "images", "train")
    output_images_val = os.path.join(output_dir, "images", "val")
    output_labels_train = os.path.join(output_dir, "labels", "train")
    output_labels_val = os.path.join(output_dir, "labels", "val")
    output_vis_train = os.path.join(output_dir, vis_dir, "train")
    output_vis_val = os.path.join(output_dir, vis_dir, "val")

    ensure_dir(output_images_train)
    ensure_dir(output_images_val)
    ensure_dir(output_labels_train)
    ensure_dir(output_labels_val)
    if save_visualizations:
        ensure_dir(output_vis_train)
        ensure_dir(output_vis_val)

    model = load_model(model_config_path, weights_path, device=device)
    caption = ". ".join(class_names)

    def process_split(images: List[str], split_name: str) -> None:
        if split_name == "train":
            images_dir = output_images_train
            labels_dir = output_labels_train
            vis_dir_local = output_vis_train
        else:
            images_dir = output_images_val
            labels_dir = output_labels_val
            vis_dir_local = output_vis_val

        for image_path in tqdm(images, desc=f"Processing {split_name}"):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            if skip_existing and os.path.exists(label_path):
                continue

            image_source, image = load_image(image_path)
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )

            class_ids: List[int] = []
            yolo_boxes: List[Tuple[float, float, float, float]] = []
            for box, phrase in zip(boxes.tolist(), phrases):
                label = str(phrase).strip().lower()
                if label not in class_to_id:
                    continue
                cx, cy, w, h = normalize_box(box)
                if w <= 0 or h <= 0:
                    continue
                class_ids.append(class_to_id[label])
                yolo_boxes.append((cx, cy, w, h))

            write_yolo_labels(label_path, class_ids, yolo_boxes)

            dst_image = os.path.join(images_dir, os.path.basename(image_path))
            place_image(image_path, dst_image, copy_images, use_symlinks)

            if save_visualizations:
                annotated = annotate(
                    image_source=image_source,
                    boxes=boxes,
                    logits=logits,
                    phrases=phrases,
                )
                vis_path = os.path.join(vis_dir_local, f"{base_name}.jpg")
                cv2.imwrite(vis_path, annotated)

    process_split(train_images, "train")
    if val_images:
        process_split(val_images, "val")

    dataset_yaml_name = yolo_cfg.get("dataset_yaml", "dataset.yaml")
    dataset_yaml_path = os.path.join(output_dir, dataset_yaml_name)
    dataset_yaml = {
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=False, sort_keys=False)

    print(f"Done. Dataset saved to: {output_dir}")
    print(f"Dataset YAML: {dataset_yaml_path}")


if __name__ == "__main__":
    main()
