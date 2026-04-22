"""Gymnasium environment for Franka Panda object picking — aligned with machines-13-00973.

Observation (15-dim):
  [0:7]   joint angles normalised to [-1,1] by joint range
  [7:10]  EEF (pinch site) position in world frame
  [10]    active object class label (integer 0–3, normalised to [0,1])
  [11:14] object position in world frame
  [14]    object yaw angle (rad)

Action (7-dim, paper Eq. 5):
  [0:7]   joint angle deltas (rad/step), clipped to ±MAX_DELTA
  Gripper is rule-based: closes automatically when EEF enters the pre-grasp zone.

Reward (Eq. 14 of paper):
  +3000 + 0.5*steps_remaining   terminal success: EEF within GRASP_DIST of target (no collision)
  +1000 * (prev_dist - curr_dist)            distance-reduction shaping (positive = getting closer)
  -0.01 * |eef_yaw - obj_yaw|               orientation alignment penalty per step
  -100 (+ terminate)                         collision: arm/gripper geom contacts table surface

Conveyor:
  Objects slide in the +y direction at conveyor_speed m/s.
  Episode truncates if object exits the belt (y > BELT_END).
  Stage 1 (paper): conveyor_speed=0.05 m/s, single object type (can, class 1).
  Stage 2 (paper): conveyor_speed random in [0.03, 0.22], object type random from can+carton.
"""

import os
import re
import tempfile

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SCENE_XML    = os.path.join(PROJECT_ROOT, "assets", "scene.xml")

JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
]

ARM_ACT_NAMES = [
    "actuator1", "actuator2", "actuator3", "actuator4",
    "actuator5", "actuator6", "actuator7",
]

# Franka Panda joint limits (rad)
JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

MAX_DELTA   = 0.05      # max joint delta per step (rad)
MAX_STEPS   = 200       # max episode steps
N_SUBSTEPS  = 20        # physics steps per RL step → 0.002 × 20 = 0.04 s/action

# Reward constants (paper Eq. 14)
DIST_REWARD_SCALE    = 1000.0
GRASP_TERMINAL_BONUS = 3000.0
TIME_BONUS_SCALE     = 0.5
ORIENT_PENALTY_SCALE = 0.01
COLLISION_PENALTY    = 100.0

GRASP_DIST  = 0.03      # pre-grasp zone radius (m) — paper Section 5.2 / Eq. 14
FALL_THRESH = 0.12      # object z drop below z_nom → knocked off table

# Conveyor belt bounds (y-axis, table y range is -0.30 to +0.30)
BELT_START  = -0.22     # object spawn y
BELT_END    =  0.27     # if object exceeds this, episode ends (missed)
# Object spawn x: within belt (belt centered at x=0.65, half-size 0.20, spans x=0.45–0.85)
BELT_X_MIN  =  0.48
BELT_X_MAX  =  0.82

# Object types (paper Table A2 dimensions)
OBJ_TYPES = [
    {"name": "obj_cereal", "geom": "obj_cereal_geom", "joint": "obj_cereal_joint", "z_nom": 0.480},   # triangular prism, half-h=0.06
    {"name": "obj_can",    "geom": "obj_can_geom",    "joint": "obj_can_joint",    "z_nom": 0.4735},  # soda can cylinder, half-h=0.0535
    {"name": "obj_milk",   "geom": "obj_milk_geom",   "joint": "obj_milk_joint",   "z_nom": 0.470},   # carton, half-h=0.05
    {"name": "obj_ball",   "geom": "obj_ball_geom",   "joint": "obj_ball_joint",   "z_nom": 0.443},   # apple sphere, r=0.023
]
N_CLASSES = len(OBJ_TYPES)


def _load_scene(scene_path):
    scene_path = os.path.abspath(scene_path)
    scene_dir  = os.path.dirname(scene_path)
    with open(scene_path) as f:
        xml = f.read()
    include_re = re.compile(r'<include\s+file="([^"]+)"\s*/>')
    match = include_re.search(xml)
    if not match:
        return mujoco.MjModel.from_xml_path(scene_path)
    rel  = match.group(1)
    abs_ = os.path.normpath(os.path.join(scene_dir, rel))
    xml  = include_re.sub(f'<include file="{os.path.basename(abs_)}"/>', xml)
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=os.path.dirname(abs_))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(xml)
        model = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    return model


class PickEnv(gym.Env):
    """UR5e single-object pick task with conveyor belt.

    Parameters
    ----------
    conveyor_speed : float
        Constant conveyor speed (m/s). Stage 1 = 0.05.
    active_object : int or None
        Index into OBJ_TYPES. None = randomise each episode (Stage 2).
    randomize_speed : bool
        If True, sample conveyor_speed ~ Uniform(0.03, 0.22) each episode (Stage 2).
    use_gt_pos : bool
        Use ground-truth object position instead of YOLO (True = faster training).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode=None,
        use_gt_pos=True,
        active_object: int = 1,          # 1 = can (matches paper Stage 1)
        conveyor_speed: float = 0.05,    # m/s (paper Stage 1)
        randomize_speed: bool = False,   # Stage 2: random speed each episode
        object_classes: list = None,     # Stage 2: restrict to subset e.g. [1, 2]
    ):
        super().__init__()
        self.render_mode       = render_mode
        self._use_gt_pos       = use_gt_pos
        self._active_object_arg = active_object   # None = randomise
        self._base_conveyor_speed = conveyor_speed
        self._randomize_speed  = randomize_speed
        self._conveyor_speed   = conveyor_speed   # actual speed this episode
        # Classes to sample from when active_object is None (paper Stage 2: can + carton)
        self._object_classes   = object_classes if object_classes is not None else list(range(N_CLASSES))

        self.model = _load_scene(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        # EEF site (pinch point between fingers)
        self._eef_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

        # Object body IDs, geom IDs, freejoint qpos/qvel addresses
        self._obj_ids        = []
        self._obj_geom_ids   = []
        self._obj_qpos_addrs = []
        self._obj_qvel_addrs = []
        for t in OBJ_TYPES:
            self._obj_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, t["name"]))
            self._obj_geom_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, t["geom"]))
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, t["joint"])
            self._obj_qpos_addrs.append(self.model.jnt_qposadr[jnt_id])
            self._obj_qvel_addrs.append(self.model.jnt_dofadr[jnt_id])

        # Franka finger body geoms — allowed to contact table_top during grasping
        _finger_bodies = ["left_finger", "right_finger"]
        self._pad_geom_ids = set(
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] in [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in _finger_bodies
            ]
        )

        # Table top geom — used for collision detection
        self._table_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")

        # Robot base (link0) geoms — allowed to contact table_top (base is mounted on table)
        _base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self._base_geom_ids = set(
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] == _base_body_id
        )

        # Arm joint qpos addresses
        jnt_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        ]
        self._qpos_addrs = np.array(
            [self.model.jnt_qposadr[j] for j in jnt_ids], dtype=np.intp)

        # Arm actuator IDs
        self._arm_act_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ARM_ACT_NAMES
        ], dtype=np.intp)
        self._grip_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")

        # Home keyframe
        self._home_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "pick_home")

        # Perception (lazy-load only when not using ground truth)
        self._perception  = None
        self._renderer    = None
        if not self._use_gt_pos:
            from envs.perception import Perception
            self._perception = Perception()
            self._perception.attach_camera(self.model, self.data)
        # Only allocate offscreen renderer when it will actually be used
        self._renderer = (
            mujoco.Renderer(self.model, 640, 640)
            if (not self._use_gt_pos or render_mode == "rgb_array")
            else None
        )

        # Gym spaces
        # Obs: 7 joints + 3 EEF + 1 class + 3 obj_pos + 1 obj_yaw = 15
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        # Action (paper Eq. 5): 7 joint angle deltas — gripper is rule-based
        self.action_space = spaces.Box(
            low =np.full(7, -MAX_DELTA, dtype=np.float32),
            high=np.full(7,  MAX_DELTA, dtype=np.float32),
        )

        # Episode state
        self._step_count     = 0
        self._ctrl_target    = np.zeros(7)
        self._gripper_closed = False
        self._prev_dist      = 0.0
        self._active_idx     = active_object if active_object is not None else 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obj_yaw(self) -> float:
        """Extract yaw angle from the active object's freejoint quaternion."""
        addr = self._obj_qpos_addrs[self._active_idx]
        qw, qx, qy, qz = self.data.qpos[addr+3:addr+7]
        return float(np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2)))

    def _get_eef_yaw(self) -> float:
        """Extract yaw angle from the EEF site rotation matrix."""
        R = self.data.site_xmat[self._eef_site_id].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _check_collision(self) -> bool:
        """True if an arm/wrist geom (not fingers) contacts the table surface."""
        # Allow: object geoms on table (resting), pad geoms near table (grasping low objects)
        allowed = set(self._obj_geom_ids) | set(self._pad_geom_ids) | self._base_geom_ids
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._table_geom_id or g2 == self._table_geom_id:
                other = g2 if g1 == self._table_geom_id else g1
                if other not in allowed:
                    return True
        return False

    def _get_obs(self) -> np.ndarray:
        # Joint angles normalised to [-1, 1]
        q      = self.data.qpos[self._qpos_addrs]
        lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
        q_norm = (2.0 * (q - lo) / (hi - lo) - 1.0).astype(np.float32)

        eef_world = self.data.site_xpos[self._eef_site_id].copy().astype(np.float32)

        if self._use_gt_pos:
            obj_world = self.data.xpos[self._obj_ids[self._active_idx]].copy().astype(np.float32)
        else:
            self._renderer.update_scene(self.data, camera="perception_cam")
            frame = self._renderer.render()
            detected, _, pos_world, _ = self._perception.detect(frame)
            obj_world = pos_world.astype(np.float32) if detected else \
                        self.data.xpos[self._obj_ids[self._active_idx]].copy().astype(np.float32)

        class_label = np.float32(self._active_idx / (N_CLASSES - 1))  # normalised 0-1
        obj_yaw     = np.float32(self._get_obj_yaw())

        return np.concatenate([q_norm, eef_world, [class_label], obj_world, [obj_yaw]])

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose active object
        if self._active_object_arg is None:
            self._active_idx = int(self.np_random.choice(self._object_classes))
        else:
            self._active_idx = self._active_object_arg

        # Choose conveyor speed
        if self._randomize_speed:
            self._conveyor_speed = float(self.np_random.uniform(0.03, 0.22))
        else:
            self._conveyor_speed = self._base_conveyor_speed

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)

        z_nom = OBJ_TYPES[self._active_idx]["z_nom"]

        for i, addr in enumerate(self._obj_qpos_addrs):
            if i == self._active_idx:
                # Place at conveyor start position
                x   = float(self.np_random.uniform(BELT_X_MIN, BELT_X_MAX))
                y   = BELT_START
                yaw = float(self.np_random.uniform(0.0, 2 * np.pi))
                half_yaw = yaw / 2.0
                self.data.qpos[addr:addr+3]   = [x, y, z_nom]
                self.data.qpos[addr+3:addr+7] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
                # Zero velocity
                qv = self._obj_qvel_addrs[i]
                self.data.qvel[qv:qv+6] = 0.0
            else:
                # Park inactive objects off-scene
                z_park = OBJ_TYPES[i]["z_nom"] - 0.42
                self.data.qpos[addr:addr+3]   = [5.0, i * 0.3, z_park]
                self.data.qpos[addr+3:addr+7] = [1.0, 0.0, 0.0, 0.0]

        # Park phantom marker underground so it doesn't cast shadows
        phantom_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "phantom_joint")
        if phantom_jnt >= 0:
            a = self.model.jnt_qposadr[phantom_jnt]
            self.data.qpos[a:a+3]   = [0.0, 0.0, -5.0]
            self.data.qpos[a+3:a+7] = [1.0, 0.0, 0.0, 0.0]

        self._ctrl_target = self.data.qpos[self._qpos_addrs].copy()
        self.data.ctrl[self._arm_act_ids] = self._ctrl_target

        mujoco.mj_forward(self.model, self.data)

        self._step_count     = 0
        self._gripper_closed = False

        eef_world = self.data.site_xpos[self._eef_site_id].copy()
        obj_world = self.data.xpos[self._obj_ids[self._active_idx]].copy()
        self._prev_dist = float(np.linalg.norm(eef_world - obj_world))

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        # Joint-space delta control (paper Eq. 5: 7 joint deltas)
        self._ctrl_target = np.clip(
            self._ctrl_target + action[:7],
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1],
        )
        self.data.ctrl[self._arm_act_ids] = self._ctrl_target

        # Rule-based gripper: open (255) during approach, close (0) when within grasp zone.
        # actuator8 ctrlrange 0–255 maps to finger position 0–0.04 m (0=closed, 255=open).
        self.data.ctrl[self._grip_act_id] = 0.0 if self._gripper_closed else 255.0

        # Physics substeps + conveyor velocity injection
        qv_addr = self._obj_qvel_addrs[self._active_idx]
        for _ in range(N_SUBSTEPS):
            if not self._gripper_closed:
                # Conveyor: lock object to belt — maintain y-velocity, suppress all
                # other motion so the object doesn't drift or tip during transport.
                self.data.qvel[qv_addr + 0] = 0.0   # no x drift
                self.data.qvel[qv_addr + 1] = self._conveyor_speed
                self.data.qvel[qv_addr + 2] = 0.0   # no z bounce
                self.data.qvel[qv_addr + 3] = 0.0   # no roll
                self.data.qvel[qv_addr + 4] = 0.0   # no pitch
                self.data.qvel[qv_addr + 5] = 0.0   # no yaw spin
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()

        eef_world = self.data.site_xpos[self._eef_site_id].copy()
        obj_world = self.data.xpos[self._obj_ids[self._active_idx]].copy()
        curr_dist = float(np.linalg.norm(eef_world - obj_world))

        # Rule-based gripper: close when EEF enters pre-grasp zone (paper Eq. 14)
        self._gripper_closed = curr_dist < GRASP_DIST

        # --- Reward (paper Eq. 14) ---

        # Collision check (terminate + penalty)
        if self._check_collision():
            reward = -COLLISION_PENALTY
            self._prev_dist = curr_dist
            return obs, reward, True, False, {"success": False, "is_success": False, "collision": True}

        # Distance-reduction shaping
        dist_reward = DIST_REWARD_SCALE * (self._prev_dist - curr_dist)

        # Orientation alignment penalty (shortest angular distance)
        eef_yaw = self._get_eef_yaw()
        obj_yaw = self._get_obj_yaw()
        angle_diff = (eef_yaw - obj_yaw + np.pi) % (2 * np.pi) - np.pi
        orient_penalty = -ORIENT_PENALTY_SCALE * abs(angle_diff)

        reward = dist_reward + orient_penalty

        # Success: EEF within 0.03 m of object center — paper Eq. 14
        if curr_dist < GRASP_DIST:
            steps_remaining = MAX_STEPS - self._step_count
            reward += GRASP_TERMINAL_BONUS + TIME_BONUS_SCALE * steps_remaining
            self._prev_dist = curr_dist
            return obs, reward, True, False, {"success": True, "is_success": True}

        # Object fell off table or escaped conveyor belt
        fell_off     = obj_world[2] < OBJ_TYPES[self._active_idx]["z_nom"] - FALL_THRESH
        escaped_belt = obj_world[1] > BELT_END
        if fell_off or escaped_belt:
            self._prev_dist = curr_dist
            return obs, reward, True, False, {"success": False, "is_success": False}

        self._prev_dist = curr_dist
        truncated = self._step_count >= MAX_STEPS
        return obs, reward, False, truncated, {"success": False, "is_success": False}

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data, camera="perception_cam")
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
