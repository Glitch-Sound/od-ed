import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from effdet import create_model
from effdet.config import get_efficientdet_config
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).parent / "data"
ANNOTATION_FILE = DATA_DIR / "annotations" / "instances_default.json"
IMAGE_DIR = DATA_DIR / "images"
MODEL_DIR = Path(__file__).parent / "model"
PARAM_DIR = Path(__file__).parent / "param"
RESULT_DIR = Path(__file__).parent / "result"
PRETRAINED_CKPT = MODEL_DIR / "tf_efficientdet_d0.pth"
TRAINED_CKPT = PARAM_DIR / "efficientdet_d0_finetuned.pth"


def _load_coco_annotation() -> Tuple[List[Dict], Dict[int, List[Dict]], Dict[int, int]]:
    """Load COCO-format annotations."""
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    image_to_anns: Dict[int, List[Dict]] = {}
    for ann in annotations:
        image_to_anns.setdefault(ann["image_id"], []).append(ann)
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    return images, image_to_anns, cat_id_to_idx


def _summarize_dataset(
    images: List[Dict],
    image_to_anns: Dict[int, List[Dict]],
    cat_id_to_idx: Dict[int, int],
) -> None:
    num_images = len(images)
    num_boxes = sum(len(v) for v in image_to_anns.values())
    num_zero_area = sum(
        1
        for anns in image_to_anns.values()
        for a in anns
        if a["bbox"][2] <= 0 or a["bbox"][3] <= 0
    )
    cat_counts: Dict[int, int] = {}
    for anns in image_to_anns.values():
        for a in anns:
            cat_counts[a["category_id"]] = cat_counts.get(a["category_id"], 0) + 1
    print(
        f"[Dataset] images={num_images} boxes={num_boxes} zero_area_boxes={num_zero_area}"
    )
    if cat_counts:
        print(f"[Dataset] box counts per category id: {cat_counts}")
    print(f"[Dataset] cat_id_to_idx mapping: {cat_id_to_idx}")


def _split_dataset(
    images: List[Dict], seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split images into train/val/test by 7:2:1."""
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    train_imgs = shuffled[:n_train]
    val_imgs = shuffled[n_train : n_train + n_val]
    test_imgs = shuffled[n_train + n_val :]
    return train_imgs, val_imgs, test_imgs


class SimpleCocoDataset(Dataset):
    """Minimal COCO dataset with resize and bbox scaling."""

    def __init__(
        self,
        image_entries: List[Dict],
        image_to_anns: Dict[int, List[Dict]],
        cat_id_to_idx: Dict[int, int],
        image_dir: Path,
        image_size: int = 512,
        return_meta: bool = False,
    ):
        self.image_entries = image_entries
        self.image_to_anns = image_to_anns
        self.cat_id_to_idx = cat_id_to_idx
        self.image_dir = image_dir
        self.image_size = image_size
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.image_entries)

    def __getitem__(self, idx: int):
        info = self.image_entries[idx]
        img_path = self.image_dir / info["file_name"]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((self.image_size, self.image_size))
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        boxes, labels = [], []
        for ann in self.image_to_anns.get(info["id"], []):
            x, y, w, h = ann["bbox"]
            # EfficientDetは(ymin, xmin, ymax, xmax)順を想定
            boxes.append(
                [y * scale_y, x * scale_x, (y + h) * scale_y, (x + w) * scale_x]
            )
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        # ImageNet正規化
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        image_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW
        target = {
            "bbox": torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "cls": torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64),
            "img_size": torch.tensor(
                [self.image_size, self.image_size], dtype=torch.float32
            ),
            "img_scale": torch.tensor([1.0], dtype=torch.float32),
            "img_id": torch.tensor([info["id"]], dtype=torch.int64),
        }
        if self.return_meta:
            meta = {
                "file_name": info["file_name"],
                "scale_x": scale_x,
                "scale_y": scale_y,
                "orig_w": orig_w,
                "orig_h": orig_h,
            }
            return image_tensor, target, meta
        return image_tensor, target


def _collate_train(batch):
    images, targets = zip(*batch)
    out = {
        "bbox": [t["bbox"] for t in targets],
        "cls": [t["cls"] for t in targets],
        "img_size": torch.stack([t["img_size"] for t in targets]),
        "img_scale": torch.stack([t["img_scale"] for t in targets]),
        "img_id": torch.stack([t["img_id"] for t in targets]),
    }
    return torch.stack(images), out


def _collate_test(batch):
    images, targets, metas = zip(*batch)
    return torch.stack(images), list(targets), list(metas)


def _ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_pretrained_checkpoint() -> str:
    """Ensure pretrained weight exists under model/ and return path."""
    _ensure_model_dir()
    if PRETRAINED_CKPT.exists():
        return str(PRETRAINED_CKPT)

    cfg = get_efficientdet_config("tf_efficientdet_d0")
    url = getattr(cfg, "url", "")
    if not url:
        print(
            f"事前学習済みモデルのURLが取得できませんでした。{PRETRAINED_CKPT} に手動配置してください。"
        )
        return ""

    try:
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        torch.save(state, PRETRAINED_CKPT)
        print(f"事前学習済みモデルを {PRETRAINED_CKPT} に保存しました。")
        return str(PRETRAINED_CKPT)
    except Exception as exc:
        print(f"事前学習済みモデルのダウンロードに失敗しました: {exc}")
        print(f"手動で {PRETRAINED_CKPT} に配置してください。")
        return ""


def _to_device_batch(
    targets: Dict[str, List[torch.Tensor]], device: torch.device
) -> Dict[str, List[torch.Tensor]]:
    out: Dict[str, List[torch.Tensor]] = {}
    for k, v in targets.items():
        if isinstance(v, list):
            out[k] = [t.to(device) for t in v]
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def ob_efficientdet_train(
    num_epochs: int = 10, batch_size: int = 2, lr: float = 1e-4, image_size: int = 512
):
    """Train EfficientDet on train/val split and save weights to param/."""
    _ensure_model_dir()
    images, image_to_anns, cat_id_to_idx = _load_coco_annotation()
    if not images:
        print("アノテーション内に画像情報がありません。")
        return
    _summarize_dataset(images, image_to_anns, cat_id_to_idx)
    train_imgs, val_imgs, test_imgs = _split_dataset(images)
    print(f"[Split] train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")
    num_classes = len(cat_id_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SimpleCocoDataset(
        train_imgs, image_to_anns, cat_id_to_idx, IMAGE_DIR, image_size=image_size
    )
    val_ds = SimpleCocoDataset(
        val_imgs, image_to_anns, cat_id_to_idx, IMAGE_DIR, image_size=image_size
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_train
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_train
    )

    _prepare_pretrained_checkpoint()  # 確保だけ実施（create_modelでpretrained=Trueを使う）
    model = create_model(
        "tf_efficientdet_d0",
        bench_task="train",
        num_classes=num_classes,
        pretrained=True,  # 内部でheadをリセットしてバックボーンをロード
        bench_labeler=True,
        # img_size omitted for compatibility with older effdet
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images_batch, targets in train_loader:
            images_batch = images_batch.to(device)
            targets = _to_device_batch(targets, device)
            optimizer.zero_grad()
            outputs = model(images_batch, targets)
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            gt_count = sum(t.numel() for t in targets["cls"])
            print(f"[Train] batch loss={loss.item():.4f} gt_boxes={gt_count}")
        avg_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images_batch, targets in val_loader:
                images_batch = images_batch.to(device)
                targets = _to_device_batch(targets, device)
                outputs = model(images_batch, targets)
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                val_loss += loss.item()
                gt_count = sum(t.numel() for t in targets["cls"])
                print(f"[Val] batch loss={loss.item():.4f} gt_boxes={gt_count}")
            avg_val_loss = val_loss / max(len(val_loader), 1)
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f}"
        )

    torch.save(model.state_dict(), TRAINED_CKPT)
    print(f"学習済みパラメータを {TRAINED_CKPT} に保存しました。")


def ob_efficientdet_test(
    conf_thresh: float = 0.027, image_size: int = 512, draw_gt: bool = True
):
    """Load trained weights from param/ and save detection visuals to result/."""
    _ensure_model_dir()
    if not TRAINED_CKPT.exists():
        print(
            f"{TRAINED_CKPT} が見つかりません。先に ob_efficientdet_train() を実行してください。"
        )
        return
    images, image_to_anns, cat_id_to_idx = _load_coco_annotation()
    if not images:
        print("アノテーション内に画像情報がありません。")
        return
    _, _, test_imgs = _split_dataset(images)
    num_classes = len(cat_id_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = SimpleCocoDataset(
        test_imgs,
        image_to_anns,
        cat_id_to_idx,
        IMAGE_DIR,
        image_size=image_size,
        return_meta=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=_collate_test
    )

    model = create_model(
        "tf_efficientdet_d0",
        bench_task="predict",
        num_classes=num_classes,
        pretrained=False,
        # img_size omitted for compatibility with older effdet
    )
    state = torch.load(TRAINED_CKPT, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    for images_batch, targets_batch, metas in test_loader:
        images_batch = images_batch.to(device)
        with torch.no_grad():
            detections = model(images_batch)

        if isinstance(detections, dict) and "detections" in detections:
            batch_preds = detections["detections"]
        elif isinstance(detections, (list, tuple)):
            batch_preds = detections
        elif isinstance(detections, torch.Tensor):
            batch_preds = detections
        else:
            print("未知の推論出力形式です。")
            return

        meta = metas[0]
        file_name = meta["file_name"]
        scale_x, scale_y = meta["scale_x"], meta["scale_y"]
        img_orig = Image.open(IMAGE_DIR / file_name).convert("RGB")
        draw = ImageDraw.Draw(img_orig)

        detections_info = []
        max_score_all = -1.0
        if isinstance(batch_preds, torch.Tensor):
            preds = batch_preds[0].cpu().numpy()
            if preds.size > 0:
                max_score_all = float(np.max(preds[:, 4]))
            # スコア上位5件をログ用に収集
            top_idx = np.argsort(-preds[:, 4])[:5]
            top_raw = [(float(preds[i, 4]), preds[i, :4].tolist()) for i in top_idx]
            for row in preds:
                if len(row) == 7:
                    x1, y1, x2, y2, score, label, _ = row
                elif len(row) == 6:
                    x1, y1, x2, y2, score, label = row
                else:
                    continue
                if score < conf_thresh:
                    continue
                label_adj = int(label) - 1 if label >= 1 else int(label)
                draw.rectangle(
                    (x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y),
                    outline="red",
                    width=2,
                )
                draw.text(
                    (x1 / scale_x + 2, y1 / scale_y + 2),
                    f"{label_adj}:{score:.2f}",
                    fill="red",
                )
                detections_info.append((label_adj, float(score)))
        else:
            preds = batch_preds[0]
            boxes = preds.get("boxes") or preds.get("bbox") or []
            scores = preds.get("scores") or preds.get("score") or []
            labels = preds.get("labels") or preds.get("cls") or []
            if len(scores):
                max_score_all = float(max(scores))
            top_pairs = sorted(
                zip(scores, boxes), key=lambda x: float(x[0]), reverse=True
            )[:5]
            top_raw = [(float(s), b.tolist()) for s, b in top_pairs]
            for box, score, label in zip(boxes, scores, labels):
                if float(score) < conf_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                label_adj = int(label) - 1 if label >= 1 else int(label)
                draw.rectangle(
                    (x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y),
                    outline="red",
                    width=2,
                )
                draw.text(
                    (x1 / scale_x + 2, y1 / scale_y + 2),
                    f"{label_adj}:{float(score):.2f}",
                    fill="red",
                )
                detections_info.append((label_adj, float(score)))

        # GT可視化（デバッグ用）
        if draw_gt and targets_batch:
            gt = targets_batch[0]
            if isinstance(gt, dict) and "bbox" in gt:
                for box in gt["bbox"]:
                    y1, x1, y2, x2 = box.tolist()
                    draw.rectangle(
                        (x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y),
                        outline="blue",
                        width=2,
                    )
                print(f"[GT] {file_name}: gt_boxes={len(gt['bbox'])}")

        out_path = RESULT_DIR / file_name
        img_orig.save(out_path)
        det_msg = (
            ", ".join([f"(cls={c},score={s:.2f})" for c, s in detections_info[:5]])
            or "no detections"
        )
        print(
            f"推論結果を {out_path} に保存しました。検出数={len(detections_info)} 詳細={det_msg} max_score_all={max_score_all:.3f} top_raw={top_raw}"
        )
