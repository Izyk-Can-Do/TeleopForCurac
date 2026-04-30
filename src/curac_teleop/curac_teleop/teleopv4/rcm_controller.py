"""
Constrained-motion controllers for teleopv4.

**Pose convention (must match the node):** all ``tcp_pose_mm_deg`` /
``current_pose_mm_deg`` inputs are **base-frame** poses (mm, deg) of the
**effective surgical tip** — i.e. the same point the teleop node derives via
``fk_tool`` + SDK orientation when a software virtual tip offset is used.

TipLockController
    Keeps the **effective tip** xyz fixed in **base frame** while allowing
    orientation changes (angular rates in deg/s).

EntryPointLockController
    RCM / entry-lock: pivot fixed in **base frame**; shaft pivots about it;
    insertion along **tool Z** (unit vector from tip RPY in base frame).

ASSUMPTIONS
-----------
1. ``tool_z`` = third column of R(RPY) built from the pose’s roll/pitch/yaw
   (deg); points along the instrument shaft in **base frame**.

2. ``shaft_to_pivot_mm`` is measured **along tool_Z** from the effective tip
   back toward the entry (trocar).

3. Pivot = tip_position_base − shaft_to_pivot_mm * tool_z_hat (base mm).

4. ``insertion_along_neg_tool_z``: positive depth moves the tip in −tool_Z
   when True (typical “inward” convention).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

from .math_utils import (
    clamp,
    tool_z_axis_from_rpy_deg,
    vec3_add,
    vec3_cross,
    vec3_dot,
    vec3_norm,
    vec3_scale,
    vec3_sub,
)


# ═══════════════════════════════════════════════════════════════════════
# Tip-Lock Controller
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TipLockConfig:
    max_angular_deg_s: float = 10.0
    correction_gain: float = 3.0
    correction_deadband_mm: float = 0.3


class TipLockController:
    """Keep the TCP (tool-tip) position fixed; allow orientation changes.

    The xArm SDK's Cartesian velocity controller should inherently keep
    the TCP stationary when v=[0,0,0] and w=[wx,wy,wz].  This
    controller adds a proportional position-correction term to fight
    drift caused by controller imperfections or discretisation.

    STATUS: research prototype.  Not a rigorous resolved-rate
    controller.  Accuracy depends heavily on correct TCP-offset
    calibration and robot controller quality.
    """

    def __init__(self, cfg: TipLockConfig):
        self.cfg = cfg
        self.active = False
        self.locked_pos_mm: List[float] = [0.0, 0.0, 0.0]
        self.last_pos_err_mm: float = 0.0

    def capture(self, tcp_pose_mm_deg: List[float]) -> None:
        """Record current **effective tip** xyz (base mm) as locked point."""
        self.locked_pos_mm = [
            float(tcp_pose_mm_deg[0]),
            float(tcp_pose_mm_deg[1]),
            float(tcp_pose_mm_deg[2]),
        ]
        self.active = True
        self.last_pos_err_mm = 0.0

    def clear(self) -> None:
        self.active = False
        self.last_pos_err_mm = 0.0

    def compute_velocity(
        self,
        current_pose_mm_deg: List[float],
        stick_x: float,
        stick_y: float,
    ) -> Tuple[List[float], List[float]]:
        """Compute [vx,vy,vz] (mm/s) and [wx,wy,wz] (deg/s).

        stick_x, stick_y: filtered joystick inputs in [-1, 1].
        Returns (linear_vel, angular_vel).
        """
        if not self.active:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        cur_pos = [float(current_pose_mm_deg[i]) for i in range(3)]

        # ── Position correction ──────────────────────────────────────
        err = vec3_sub(self.locked_pos_mm, cur_pos)
        err_norm = vec3_norm(err)
        self.last_pos_err_mm = err_norm

        if err_norm > self.cfg.correction_deadband_mm:
            v_corr = vec3_scale(err, self.cfg.correction_gain)
        else:
            v_corr = [0.0, 0.0, 0.0]

        # ── Angular velocity from sticks ─────────────────────────────
        max_w = self.cfg.max_angular_deg_s
        wx = -stick_y * max_w
        wy = stick_x * max_w
        wz = 0.0

        return v_corr, [wx, wy, wz]


# ═══════════════════════════════════════════════════════════════════════
# Entry-Point-Lock / RCM Controller
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EntryPointLockConfig:
    max_linear_mm_s: float = 12.0
    max_angular_deg_s: float = 12.0
    pivot_correction_gain: float = 3.0
    angular_correction_gain: float = 3.0
    lin_corr_deadband_mm: float = 0.3
    ang_corr_deadband_deg: float = 0.3
    insertion_along_neg_tool_z: bool = True


class EntryPointLockController:
    """Remote-Centre-of-Motion (RCM) constrained controller.

    The pivot (entry) point is captured once and must remain fixed in
    space while the tool rotates around it and inserts/retracts through
    it.

    Physics
    -------
    Let  r = p_tcp − p_pivot  (vector from pivot to TCP).
    For pure rotation around the pivot:
        v_tcp = omega × r

    For insertion along the tool axis:
        v_insert = speed * tool_z_axis    (through the pivot)

    Corrections fight drift:
    - Linear: proportional to the perpendicular distance of the pivot
      from the current tool axis line.
    - Angular: cross-product alignment of tool_z with the (pivot→TCP)
      direction.

    The total angular velocity omega_total (user + correction) is used
    for BOTH the angular command AND the v = omega × r computation, so
    they remain coupled.
    """

    def __init__(self, cfg: EntryPointLockConfig):
        self.cfg = cfg
        self.active = False
        self.pivot_xyz: List[float] = [0.0, 0.0, 0.0]
        self.shaft_to_pivot_mm: float = 0.0
        self.last_pivot_err_mm: float = 0.0
        self.last_r_norm: float = 0.0

    def capture_pivot(
        self,
        tcp_pose_mm_deg: List[float],
        shaft_to_pivot_mm: float,
    ) -> None:
        """Record entry pivot in **base frame** from effective tip pose.

        pivot_base_mm = tip_base_mm − shaft_to_pivot_mm * tool_z_hat_base.
        """
        tcp_xyz = [float(tcp_pose_mm_deg[i]) for i in range(3)]
        tool_z = tool_z_axis_from_rpy_deg(
            tcp_pose_mm_deg[3], tcp_pose_mm_deg[4], tcp_pose_mm_deg[5],
        )
        self.pivot_xyz = vec3_sub(
            tcp_xyz, vec3_scale(tool_z, shaft_to_pivot_mm),
        )
        self.shaft_to_pivot_mm = shaft_to_pivot_mm
        self.active = True
        self.last_pivot_err_mm = 0.0
        self.last_r_norm = 0.0

    def clear(self) -> None:
        self.active = False
        self.last_pivot_err_mm = 0.0
        self.last_r_norm = 0.0

    def compute_rcm_velocity(
        self,
        current_pose_mm_deg: List[float],
        stick_x: float,
        stick_y: float,
        depth_speed_mm_s: float,
    ) -> Tuple[List[float], List[float]]:
        """Compute [vx,vy,vz] (mm/s) and [wx,wy,wz] (deg/s).

        stick_x, stick_y: filtered joystick, [-1, 1]
        depth_speed_mm_s: insertion speed (positive = inward per config)

        Returns (linear_vel, angular_vel).
        """
        if not self.active:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        cfg = self.cfg
        tcp_xyz = [float(current_pose_mm_deg[i]) for i in range(3)]
        tool_z = tool_z_axis_from_rpy_deg(
            current_pose_mm_deg[3],
            current_pose_mm_deg[4],
            current_pose_mm_deg[5],
        )

        # r: vector from pivot to TCP
        r = vec3_sub(tcp_xyz, self.pivot_xyz)
        r_norm = vec3_norm(r)
        r_safe = max(r_norm, 30.0)  # clamp to avoid division by tiny r
        r_hat = vec3_scale(r, 1.0 / r_safe)
        self.last_r_norm = r_norm

        # ── User angular command ─────────────────────────────────────
        # Adaptive gain: full stick → max_linear_mm_s at TCP regardless of depth
        w_gain_rad = cfg.max_linear_mm_s / r_safe
        w_gain_deg = min(math.degrees(w_gain_rad), cfg.max_angular_deg_s)
        w_x_deg = -stick_y * w_gain_deg
        w_y_deg = stick_x * w_gain_deg
        w_user_rad = [math.radians(w_x_deg), math.radians(w_y_deg), 0.0]

        # ── Angular correction: align tool_z with r_hat ──────────────
        ang_err_vec = vec3_cross(tool_z, r_hat)
        ang_err_mag = vec3_norm(ang_err_vec)
        ang_deadband = math.sin(math.radians(cfg.ang_corr_deadband_deg))
        if ang_err_mag > ang_deadband:
            w_ang_corr = vec3_scale(ang_err_vec, cfg.angular_correction_gain)
        else:
            w_ang_corr = [0.0, 0.0, 0.0]

        # ── Total angular velocity ───────────────────────────────────
        w_total_rad = vec3_add(w_user_rad, w_ang_corr)

        # ── Orbital translational velocity  v = omega × r ────────────
        v_orbit = vec3_cross(w_total_rad, r)

        # ── Linear correction: perpendicular pivot error ─────────────
        to_pivot = vec3_sub(self.pivot_xyz, tcp_xyz)
        dot_along = vec3_dot(to_pivot, tool_z)
        along_tool = vec3_scale(tool_z, dot_along)
        perp_err = vec3_sub(to_pivot, along_tool)
        perp_err_norm = vec3_norm(perp_err)
        self.last_pivot_err_mm = perp_err_norm

        if perp_err_norm > cfg.lin_corr_deadband_mm:
            v_corr = vec3_scale(perp_err, cfg.pivot_correction_gain)
        else:
            v_corr = [0.0, 0.0, 0.0]

        # ── Insertion velocity along tool axis ───────────────────────
        insertion_sign = -1.0 if cfg.insertion_along_neg_tool_z else 1.0
        v_insert = vec3_scale(tool_z, insertion_sign * depth_speed_mm_s)

        # ── Total linear velocity ────────────────────────────────────
        v_total = vec3_add(vec3_add(v_orbit, v_insert), v_corr)

        # ── Clamp linear speed (scale angular equally to preserve coupling)
        v_norm = vec3_norm(v_total)
        if v_norm > cfg.max_linear_mm_s and v_norm > 1e-6:
            s = cfg.max_linear_mm_s / v_norm
            v_total = vec3_scale(v_total, s)
            w_total_rad = vec3_scale(w_total_rad, s)

        w_total_deg = [math.degrees(w) for w in w_total_rad]
        return v_total, w_total_deg
