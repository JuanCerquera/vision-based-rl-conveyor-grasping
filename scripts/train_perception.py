import os
import re
import sys
import tempfile

import mujoco
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SCENE_XML    = os.path.join(PROJECT_ROOT, "assets", "scene.xml")

CAM_NAME = "perception_cam"
CAM_SIZE = 640
N_IMAGES = 500
EPOCHS = 10
BATCH_SIZE = 16

TABLE_SURFACE_Z = 0.42
OBJ_X_RANGE     = (0.20, 0.80)
OBJ_Y_RANGE     = (-0.25, 0.25)

OBJ_CONFIGS = [
    {
        "name":      "obj_can",
        "joint":     "obj_can_joint",
        "class_id":  0,
        "z_nom":     TABLE_SURFACE_Z + 0.06,
        "aabb_half": np.array([0.033, 0.033, 0.06]),
    },
    {
        "name":      "obj_milk",
        "joint":     "obj_milk_joint",
        "class_id":  1,
        "z_nom":     TABLE_SURFACE_Z + 0.09,
        "aabb_half": np.array([0.035, 0.035, 0.09]),
    },
]

_SIGNS = np.array(
    [[s0, s1, s2] for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)],
    dtype=float,
)

def load_scene(scene_path):
    scene_path = os.path.abspath(scene_path)
    scene_dir  = os.path.dirname(scene_path)
    with open(scene_path) as f:
        xml = f.read()
    include_pattern = re.compile(r'<include\s+file="([^"]+)"\s*/>')
    match = include_pattern.search(xml)
    if not match:
        return mujoco.MjModel.from_xml_path(scene_path)
    rel_include      = match.group(1)
    abs_include      = os.path.normpath(os.path.join(scene_dir, rel_include))
    include_dir      = os.path.dirname(abs_include)
    include_basename = os.path.basename(abs_include)
    xml = include_pattern.sub(f'<include file="{include_basename}"/>', xml)
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=include_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(xml)
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.remove(tmp_path)
    return model


def _cam_intrinsics(model, cam_id):
    fovy = np.radians(model.cam_fovy[cam_id])
    fy   = (CAM_SIZE / 2) / np.tan(fovy / 2)
    return fy, fy, CAM_SIZE / 2, CAM_SIZE / 2


def _project_points(pts_world, cam_pos, cam_mat, fx, fy, cx, cy):
    p_cam = (cam_mat.T @ (pts_world - cam_pos).T).T
    depth = -p_cam[:, 2]
    valid = depth > 1e-4
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(valid, fx * p_cam[:, 0] / depth + cx, np.nan)
        v = np.where(valid, -fy * p_cam[:, 1] / depth + cy, np.nan)
    return u, v, valid


def generate(n_total, val_split=0.2, seed=42):
    rng     = np.random.default_rng(seed)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val
    print(f"Generating images total={n_total} train={n_train} val={n_val}")

    print("Loading scene...")
    model = load_scene(SCENE_XML)
    data  = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pick_home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    fx, fy, cx, cy = _cam_intrinsics(model, cam_id)

    for cfg in OBJ_CONFIGS:
        cfg["body_id"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["name"])
        jnt_id         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg["joint"])
        cfg["qpos0"]   = model.jnt_qposadr[jnt_id]
        cfg["corners"] = _SIGNS * cfg["aabb_half"]

    out_root = os.path.join(PROJECT_ROOT, "data", "perception")
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    mujoco.mj_forward(model, data)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()

    print("Initialising renderer...")
    renderer = mujoco.Renderer(model, CAM_SIZE, CAM_SIZE)

    saved   = {"train": 0, "val": 0}
    skipped = 0

    pbar = tqdm(range(n_total), desc="Generating", unit="img")
    for i in pbar:
        split = "val" if i < n_val else "train"

        cfg = OBJ_CONFIGS[int(rng.integers(0, len(OBJ_CONFIGS)))]

        x   = rng.uniform(*OBJ_X_RANGE)
        y   = rng.uniform(*OBJ_Y_RANGE)
        yaw = rng.uniform(0, 2 * np.pi)
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

        a = cfg["qpos0"]
        data.qpos[a:a+3]   = [x, y, cfg["z_nom"]]
        data.qpos[a+3:a+7] = quat

        for other in OBJ_CONFIGS:
            if other["name"] == cfg["name"]:
                continue
            b = other["qpos0"]
            data.qpos[b:b+3]   = [0.0, 0.0, -5.0]
            data.qpos[b+3:b+7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_kinematics(model, data)

        obj_pos = data.xpos[cfg["body_id"]].copy()
        u, v, valid = _project_points(obj_pos + cfg["corners"], cam_pos, cam_mat, fx, fy, cx, cy)

        if not valid.all() or u.max() < 0 or u.min() > CAM_SIZE or v.max() < 0 or v.min() > CAM_SIZE:
            skipped += 1
            pbar.set_postfix(train=saved["train"], val=saved["val"], skipped=skipped)
            continue

        u_min, u_max = max(0.0, u.min()), min(CAM_SIZE, u.max())
        v_min, v_max = max(0.0, v.min()), min(CAM_SIZE, v.max())

        cx_n = (u_min + u_max) / 2 / CAM_SIZE
        cy_n = (v_min + v_max) / 2 / CAM_SIZE
        w_n  = (u_max - u_min) / CAM_SIZE
        h_n  = (v_max - v_min) / CAM_SIZE

        renderer.update_scene(data, camera=CAM_NAME)
        frame = renderer.render()

        idx      = saved[split]
        img_path = os.path.join(out_root, "images", split, f"{idx:05d}.jpg")
        lbl_path = os.path.join(out_root, "labels", split, f"{idx:05d}.txt")
        Image.fromarray(frame).save(img_path, quality=95)
        with open(lbl_path, "w") as f:
            f.write(f"{cfg['class_id']} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

        saved[split] += 1
        pbar.set_postfix(train=saved["train"], val=saved["val"], skipped=skipped)

    renderer.close()

    yaml_path = os.path.join(out_root, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(OBJ_CONFIGS)}\n")
        f.write("names: ['soda_can', 'milk_carton']\n")

    print(f"Done. train={saved['train']}, val={saved['val']}, skipped={skipped}")
    return out_root

def _best_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def train_yolo(dataset_yaml, epochs, batch, device=None):
    from ultralytics import YOLO

    if device is None:
        device = _best_device()
    workers  = 0 if sys.platform == "darwin" else 8
    model_dir = os.path.join(PROJECT_ROOT, "models", "yolo")
    os.makedirs(model_dir, exist_ok=True)

    print(f"Training YOLOv8n epochs={epochs} batch={batch} device={device}")
    model = YOLO("yolov8n.pt")

    last_val: dict = {}
    pbar = tqdm(total=epochs, desc="Training", unit="epoch")

    def on_train_epoch_end(trainer):
        metrics = {"box": f"{trainer.loss_items[0]:.3f}", "cls": f"{trainer.loss_items[1]:.3f}"}
        metrics.update(last_val)
        pbar.set_postfix(metrics)
        pbar.update(1)

    def on_val_end(validator):
        m = validator.metrics
        last_val["mAP50"] = f"{m.box.map50:.3f}"
        last_val["P"]     = f"{m.box.p.mean():.3f}"
        last_val["R"]     = f"{m.box.r.mean():.3f}"

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end",         on_val_end)

    try:
        model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=CAM_SIZE,
            batch=batch,
            device=device,
            workers=workers,
            cache=True,
            project=model_dir,
            name="object_detector",
            exist_ok=True,
            verbose=False,
        )
    except KeyboardInterrupt:
        print("Interrupted.")
        pbar.close()
        sys.exit(0)

    pbar.close()
    weights = os.path.join(model_dir, "object_detector", "weights", "best.pt")
    print(f"Training complete. Best weights: {weights}")
if __name__ == "__main__":
    dataset_yaml = os.path.join(generate(N_IMAGES), "dataset.yaml")
    train_yolo(dataset_yaml, EPOCHS, BATCH_SIZE, device=None)
