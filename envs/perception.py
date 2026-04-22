"""Perception module: YOLO detection + 3D back-projection.

Wraps a trained YOLOv8 model. Given a frame from perception_cam, detects an
object and back-projects its 2D centroid to a 3D world position using the
known object centre height as the depth constraint.

Detection failure: returns detected=False, zeros for bbox, pos_3d, and class_id=-1.
"""

import os
import numpy as np

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "models", "yolo", "object_detector", "weights", "best.pt"
)

CAM_SIZE = 640

# Centre height (z) for each YOLO class above the table surface (z=0.42).
# YOLO class 0: soda can    — cyl  r=0.033 h=0.12 m  → 0.42 + 0.06 = 0.48
# YOLO class 1: milk carton — box  0.07×0.07×0.18 m  → 0.42 + 0.09 = 0.51
OBJ_Z_BY_CLASS = {
    0: 0.48,   # soda can
    1: 0.51,   # milk carton
}

# Maps YOLO class id → env class id (used for bin selection in deploy.py)
YOLO_TO_ENV_CLASS = {
    0: 1,   # soda can    → env class 1
    1: 2,   # milk carton → env class 2
}


class Perception:
    """YOLO-based object detector with 3D back-projection.

    Usage:
        perc = Perception()
        perc.attach_camera(model, data)          # call once after loading scene
        detected, bbox, pos_3d, class_id = perc.detect(frame)

    Args:
        weights: path to YOLOv8 .pt weights file
        conf:    minimum detection confidence threshold
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.5):
        from ultralytics import YOLO
        self._yolo = YOLO(weights)
        self.conf = conf
        self._cam_id = None
        self._intrinsics = None
        self._mj_data = None
        self.last_pos_3d = None   # updated every detect() call

    def attach_camera(self, mj_model, mj_data, cam_name="perception_cam"):
        """Cache camera id, intrinsics, and a reference to mj_data.

        Must be called once after the MuJoCo model is loaded, before detect().
        mj_data is stored by reference — the perception module always reads the
        current simulation state automatically.
        """
        import mujoco
        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        fovy = np.radians(mj_model.cam_fovy[cam_id])
        fy = (CAM_SIZE / 2) / np.tan(fovy / 2)
        fx = fy
        cx = cy = CAM_SIZE / 2.0
        self._cam_id = cam_id
        self._intrinsics = (fx, fy, cx, cy)
        self._mj_data = mj_data

    def _backproject(self, u, v, z):
        """Back-project pixel (u, v) to world (x, y, z) at known height z.

        Uses the current camera pose from mj_data (fixed camera, but reads
        dynamically so it works if the camera ever moves).
        """
        fx, fy, cx, cy = self._intrinsics
        cam_pos = self._mj_data.cam_xpos[self._cam_id].copy()
        cam_mat = self._mj_data.cam_xmat[self._cam_id].reshape(3, 3).copy()

        # Ray direction in camera frame (camera looks along -z_cam)
        r_cam = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
        r_world = cam_mat @ r_cam  # rotate to world frame

        # Intersect ray with the plane z = <object centre height>
        if abs(r_world[2]) < 1e-8:
            return np.zeros(3)
        t = (z - cam_pos[2]) / r_world[2]
        return cam_pos + t * r_world

    def detect(self, frame):
        """Run YOLO inference on an RGB frame.

        Args:
            frame: (H, W, 3) uint8 RGB array from mujoco.Renderer.render()

        Returns:
            detected (bool):       True if an object was found
            bbox (np.ndarray):     shape (4,) [u_min, v_min, u_max, v_max] in
                                   pixels; zeros if not detected
            pos_3d (np.ndarray):   shape (3,) world [x, y, z]; zeros if not detected
            class_id (int):        YOLO class index; -1 if not detected
        """
        if self._cam_id is None:
            raise RuntimeError("Call attach_camera() before detect()")

        results = self._yolo(frame, conf=self.conf, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return False, np.zeros(4), np.zeros(3), -1

        # Highest-confidence detection
        best = boxes[int(boxes.conf.argmax())]
        xyxy = best.xyxy[0].cpu().numpy()  # [u_min, v_min, u_max, v_max]
        class_id = int(best.cls[0].cpu().item())
        u_c = (xyxy[0] + xyxy[2]) / 2.0
        v_c = (xyxy[1] + xyxy[3]) / 2.0

        z = OBJ_Z_BY_CLASS.get(class_id, 0.48)
        pos_3d = self._backproject(u_c, v_c, z)
        self.last_pos_3d = pos_3d.copy()
        return True, xyxy, pos_3d, class_id
