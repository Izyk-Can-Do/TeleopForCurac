"""
TeleopV7Node

PS5-focused telemanipulation node for experiment:
- Free teleop + fixed-tip mode (Tip-Lock).
- Continuous FT-driven DualSense haptics (rumble + adaptive triggers).
"""

import math
import time
from typing import List, Optional, Tuple

import rclpy

from curac_teleop.teleopv4.ft_guard import FTDataState, StalePolicy
from curac_teleop.teleopv4.input_layer import InputSnapshot
from curac_teleop.teleopv4.math_utils import (
    clamp,
    sigmoid_shape,
    tool_z_axis_from_rpy_deg,
    vec3_add,
    vec3_cross,
    vec3_norm,
    vec3_scale,
)
from curac_teleop.teleopv4.node import TeleopV4Node
from curac_teleop.teleopv4.state_machine import TeleopState
from curac_teleop.teleopv5.input_layer import DualSenseInput, InputConfig as DSInputConfig


class TeleopV7Node(TeleopV4Node):
    def _declare_params(self):
        super()._declare_params()
        self.declare_parameter("v7_force_watchdog_zero", True)
        self.declare_parameter("v7_use_tool_z_depth_in_free", True)
        self.declare_parameter("v7_enable_rcm_placeholder", False)
        self.declare_parameter("v7_mode_switch_settle_s", 0.20)
        self.declare_parameter("v7_tip_idle_reanchor", False)
        self.declare_parameter("v7_tip_idle_reanchor_dwell_s", 0.60)
        self.declare_parameter("v7_tip_idle_reanchor_max_err_mm", 0.8)
        self.declare_parameter("v7_tip_lock_max_linear_mm_s", 12.0)
        self.declare_parameter("v7_tip_lock_recovery_err_mm", 6.0)
        self.declare_parameter("v7_tip_lock_recovery_gain_mm_s_per_mm", 0.8)
        self.declare_parameter("v7_tip_lock_min_angular_scale", 0.15)
        self.declare_parameter("v7_tip_lock_error_brake_enable", True)
        self.declare_parameter("v7_tip_lock_error_brake_mm", 2.0)
        self.declare_parameter("v7_tip_lock_error_brake_exp", 1.25)
        self.declare_parameter("v7_tip_lock_pose_source", "sdk_tcp")
        self.declare_parameter("v7_tip_lock_tcp_rotation_comp_enable", True)
        self.declare_parameter("v7_tip_lock_tcp_rotation_comp_gain", 0.8)
        self.declare_parameter("v7_tip_lock_correction_gain", 6.0)
        self.declare_parameter("v7_tip_lock_correction_deadband_mm", 0.10)
        self.declare_parameter("v7_tip_lock_reanchor_on_enter", True)
        self.declare_parameter("v7_tip_lock_debug", False)
        self.declare_parameter("v7_tip_lock_debug_interval_s", 0.20)
        self.declare_parameter("v7_tip_lock_debug_err_threshold_mm", 2.0)
        self.declare_parameter("v7_haptic_feedback_gain", 1.0)
        self.declare_parameter("v7_haptics_boot_test", True)
        self.declare_parameter("v7_haptic_event_gain", 1.0)
        self.declare_parameter("v7_haptic_event_min_duration_ms", 160)
        self.declare_parameter("v7_ft_haptic_enable", True)
        self.declare_parameter("v7_ft_haptic_warn_ratio", 0.05)
        self.declare_parameter("v7_ft_haptic_release_ratio", 0.08)
        self.declare_parameter("v7_ft_haptic_force_source", "insertion_axis")
        self.declare_parameter("v7_ft_haptic_full_scale_n", 0.0)
        self.declare_parameter("v7_ft_haptic_force_alpha", 0.08)
        self.declare_parameter("v7_ft_haptic_deadband_n", 1.5)
        self.declare_parameter("v7_ft_haptic_shape_exp", 1.35)
        self.declare_parameter("v7_ft_haptic_smooth_alpha", 0.18)
        self.declare_parameter("v7_ft_haptic_max_trigger_resistance", 0.65)
        self.declare_parameter("v7_ft_haptic_rumble_enable", False)
        self.declare_parameter("v7_ft_haptic_rumble_interval_s", 0.14)
        self.declare_parameter("v7_ft_poll_fallback_enable", True)
        self.declare_parameter("v7_ft_poll_min_interval_s", 0.04)
        self.declare_parameter("v7_ft_poll_auto_init", True)
        self.declare_parameter("v7_ft_modbus_baudrate", 2000000)
        self.declare_parameter("v7_ft_global_stop_enable", True)
        self.declare_parameter("v7_ft_global_soft_slowdown", True)
        self.declare_parameter("v7_ft_allow_retreat_on_limit", False)
        self.declare_parameter("v7_ft_retreat_speed_scale", 0.40)
        self.declare_parameter("v7_ft_retreat_min_cos", 0.15)
        self.declare_parameter("v7_motion_haptic_enable", False)
        self.declare_parameter("v7_motion_haptic_gain", 0.45)
        self.declare_parameter("v7_motion_haptic_deadband_ratio", 0.03)
        self.declare_parameter("v7_motion_haptic_rumble_interval_s", 0.12)
        self.declare_parameter("dualsense_startup_grace_s", 1.0)
        self.declare_parameter("target_vendor_id", 0x054C)
        self.declare_parameter("target_product_id", 0x0CE6)

    def _load_params(self):
        super()._load_params()
        self.v7_force_watchdog_zero = bool(
            self.get_parameter("v7_force_watchdog_zero").value
        )
        self.v7_use_tool_z_depth_in_free = bool(
            self.get_parameter("v7_use_tool_z_depth_in_free").value
        )
        self.v7_enable_rcm_placeholder = bool(
            self.get_parameter("v7_enable_rcm_placeholder").value
        )
        self.v7_mode_switch_settle_s = max(
            0.0, float(self.get_parameter("v7_mode_switch_settle_s").value)
        )
        self.v7_tip_idle_reanchor = bool(
            self.get_parameter("v7_tip_idle_reanchor").value
        )
        self.v7_tip_idle_reanchor_dwell_s = max(
            0.0, float(self.get_parameter("v7_tip_idle_reanchor_dwell_s").value)
        )
        self.v7_tip_idle_reanchor_max_err_mm = max(
            0.0, float(self.get_parameter("v7_tip_idle_reanchor_max_err_mm").value)
        )
        self.v7_tip_lock_max_linear_mm_s = float(
            self.get_parameter("v7_tip_lock_max_linear_mm_s").value
        )
        self.v7_tip_lock_recovery_err_mm = max(
            0.1, float(self.get_parameter("v7_tip_lock_recovery_err_mm").value)
        )
        self.v7_tip_lock_recovery_gain_mm_s_per_mm = max(
            0.0, float(self.get_parameter("v7_tip_lock_recovery_gain_mm_s_per_mm").value)
        )
        self.v7_tip_lock_min_angular_scale = clamp(
            float(self.get_parameter("v7_tip_lock_min_angular_scale").value), 0.0, 1.0
        )
        self.v7_tip_lock_error_brake_enable = bool(
            self.get_parameter("v7_tip_lock_error_brake_enable").value
        )
        self.v7_tip_lock_error_brake_mm = max(
            0.1, float(self.get_parameter("v7_tip_lock_error_brake_mm").value)
        )
        self.v7_tip_lock_error_brake_exp = max(
            0.2, float(self.get_parameter("v7_tip_lock_error_brake_exp").value)
        )
        self.v7_tip_lock_pose_source = str(
            self.get_parameter("v7_tip_lock_pose_source").value
        ).strip().lower()
        if self.v7_tip_lock_pose_source not in ("sdk_tcp", "kdl_effective"):
            self.get_logger().warn(
                f"[V7] invalid v7_tip_lock_pose_source={self.v7_tip_lock_pose_source!r}; using sdk_tcp"
            )
            self.v7_tip_lock_pose_source = "sdk_tcp"
        self.v7_tip_lock_tcp_rotation_comp_enable = bool(
            self.get_parameter("v7_tip_lock_tcp_rotation_comp_enable").value
        )
        self.v7_tip_lock_tcp_rotation_comp_gain = clamp(
            float(self.get_parameter("v7_tip_lock_tcp_rotation_comp_gain").value), 0.0, 2.0
        )
        self.v7_tip_lock_correction_gain = max(
            0.0, float(self.get_parameter("v7_tip_lock_correction_gain").value)
        )
        self.v7_tip_lock_correction_deadband_mm = max(
            0.0, float(self.get_parameter("v7_tip_lock_correction_deadband_mm").value)
        )
        self.v7_tip_lock_reanchor_on_enter = bool(
            self.get_parameter("v7_tip_lock_reanchor_on_enter").value
        )
        self.v7_tip_lock_debug = bool(
            self.get_parameter("v7_tip_lock_debug").value
        )
        self.v7_tip_lock_debug_interval_s = max(
            0.05, float(self.get_parameter("v7_tip_lock_debug_interval_s").value)
        )
        self.v7_tip_lock_debug_err_threshold_mm = max(
            0.0, float(self.get_parameter("v7_tip_lock_debug_err_threshold_mm").value)
        )
        self.v7_haptic_feedback_gain = float(
            self.get_parameter("v7_haptic_feedback_gain").value
        )
        self.v7_haptics_boot_test = bool(
            self.get_parameter("v7_haptics_boot_test").value
        )
        self.v7_haptic_event_gain = float(
            self.get_parameter("v7_haptic_event_gain").value
        )
        self.v7_haptic_event_min_duration_ms = int(
            self.get_parameter("v7_haptic_event_min_duration_ms").value
        )
        self.v7_ft_haptic_enable = bool(
            self.get_parameter("v7_ft_haptic_enable").value
        )
        self.v7_ft_haptic_warn_ratio = float(
            self.get_parameter("v7_ft_haptic_warn_ratio").value
        )
        self.v7_ft_haptic_release_ratio = float(
            self.get_parameter("v7_ft_haptic_release_ratio").value
        )
        self.v7_ft_haptic_force_source = str(
            self.get_parameter("v7_ft_haptic_force_source").value
        ).strip().lower()
        if self.v7_ft_haptic_force_source not in ("insertion_axis", "norm", "max_axis"):
            self.get_logger().warn(
                f"[V7] invalid v7_ft_haptic_force_source={self.v7_ft_haptic_force_source!r}; using insertion_axis"
            )
            self.v7_ft_haptic_force_source = "insertion_axis"
        self.v7_ft_haptic_full_scale_n = max(
            0.0, float(self.get_parameter("v7_ft_haptic_full_scale_n").value)
        )
        self.v7_ft_haptic_force_alpha = clamp(
            float(self.get_parameter("v7_ft_haptic_force_alpha").value), 0.01, 1.0
        )
        self.v7_ft_haptic_deadband_n = max(
            0.0, float(self.get_parameter("v7_ft_haptic_deadband_n").value)
        )
        self.v7_ft_haptic_shape_exp = max(
            0.7, float(self.get_parameter("v7_ft_haptic_shape_exp").value)
        )
        self.v7_ft_haptic_smooth_alpha = clamp(
            float(self.get_parameter("v7_ft_haptic_smooth_alpha").value), 0.01, 1.0
        )
        self.v7_ft_haptic_max_trigger_resistance = clamp(
            float(self.get_parameter("v7_ft_haptic_max_trigger_resistance").value), 0.05, 1.0
        )
        self.v7_ft_haptic_rumble_enable = bool(
            self.get_parameter("v7_ft_haptic_rumble_enable").value
        )
        self.v7_ft_haptic_rumble_interval_s = float(
            self.get_parameter("v7_ft_haptic_rumble_interval_s").value
        )
        self.v7_ft_poll_fallback_enable = bool(
            self.get_parameter("v7_ft_poll_fallback_enable").value
        )
        self.v7_ft_poll_min_interval_s = max(
            0.01, float(self.get_parameter("v7_ft_poll_min_interval_s").value)
        )
        self.v7_ft_poll_auto_init = bool(
            self.get_parameter("v7_ft_poll_auto_init").value
        )
        self.v7_ft_modbus_baudrate = int(
            self.get_parameter("v7_ft_modbus_baudrate").value
        )
        self.v7_ft_global_stop_enable = bool(
            self.get_parameter("v7_ft_global_stop_enable").value
        )
        self.v7_ft_global_soft_slowdown = bool(
            self.get_parameter("v7_ft_global_soft_slowdown").value
        )
        self.v7_ft_allow_retreat_on_limit = bool(
            self.get_parameter("v7_ft_allow_retreat_on_limit").value
        )
        self.v7_ft_retreat_speed_scale = clamp(
            float(self.get_parameter("v7_ft_retreat_speed_scale").value), 0.05, 1.0
        )
        self.v7_ft_retreat_min_cos = clamp(
            float(self.get_parameter("v7_ft_retreat_min_cos").value), 0.0, 1.0
        )
        self.v7_motion_haptic_enable = bool(
            self.get_parameter("v7_motion_haptic_enable").value
        )
        self.v7_motion_haptic_gain = max(
            0.0, float(self.get_parameter("v7_motion_haptic_gain").value)
        )
        self.v7_motion_haptic_deadband_ratio = clamp(
            float(self.get_parameter("v7_motion_haptic_deadband_ratio").value), 0.0, 0.95
        )
        self.v7_motion_haptic_rumble_interval_s = max(
            0.05, float(self.get_parameter("v7_motion_haptic_rumble_interval_s").value)
        )
        if self.v7_force_watchdog_zero and self.velocity_watchdog_s != 0.0:
            self.get_logger().warn(
                f"[V7] forcing velocity_watchdog_s from {self.velocity_watchdog_s:.3f} to 0.0 for smooth stream"
            )
            self.velocity_watchdog_s = 0.0

    def __init__(self):
        super().__init__()
        self._v7_constrained_settle_until_s = 0.0
        self._v7_idle_since_s = 0.0
        self._v7_prev_ft_feedback = 0.0
        self._v7_last_ft_rumble_s = 0.0
        self._v7_last_ft_poll_s = 0.0
        # Keep adaptive feedback aligned with live operator mapping:
        # current robot behavior uses R2 for inward/down.
        self._v7_depth_active_side = "right"
        self._v7_last_tip_dbg_s = 0.0
        self._v7_ft_feedback_filtered = 0.0
        self._v7_ft_force_filtered_n = 0.0
        self._v7_direct_ft_logged = False
        self._v7_ft_auto_init_attempted = False
        self._v7_last_motion_rumble_s = 0.0
        self._v7_last_motion_feedback = 0.0
        if not self.dry_run:
            cfg = DSInputConfig(
                deadzone=float(self.get_parameter("deadzone").value),
                filter_alpha=float(self.get_parameter("input_filter_alpha").value),
                release_alpha=float(self.get_parameter("input_release_alpha").value),
                zero_snap=float(self.get_parameter("input_zero_snap").value),
                debounce_s=float(self.get_parameter("button_debounce_s").value),
                startup_grace_s=float(self.get_parameter("dualsense_startup_grace_s").value),
                lx_sign=float(self.get_parameter("lx_sign").value),
                ly_sign=float(self.get_parameter("ly_sign").value),
                rx_sign=float(self.get_parameter("rx_sign").value),
                ry_sign=float(self.get_parameter("ry_sign").value),
                target_vendor_id=int(self.get_parameter("target_vendor_id").value),
                target_product_id=int(self.get_parameter("target_product_id").value),
            )
            self.input = DualSenseInput(cfg)
            self.get_logger().info("[V7] PS5 DualSense input layer enabled")
            self._log_dualsense_haptics_status()
        self.get_logger().info(
            "[Teleop ready | R1 deadman | Cross fixed-tip toggle | "
            "Square FT tare | L2/R2 depth in free mode"
        )

    def _tip_lock_feedback_pose_mm_deg(self, joints_rad: Optional[List[float]] = None) -> Optional[List[float]]:
        """Pose source used by Tip-Lock capture/feedback loop."""
        if self.v7_tip_lock_pose_source == "sdk_tcp":
            try:
                code_p, pose_sdk = self.arm.get_position(is_radian=False)
            except Exception:
                code_p, pose_sdk = -1, None
            if code_p == 0 and pose_sdk:
                return [float(pose_sdk[i]) for i in range(6)]
            # Fall back to model pose if SDK read fails.
        if joints_rad is None:
            try:
                code_j, joints = self.arm.get_servo_angle(is_radian=True)
            except Exception:
                code_j, joints = -1, None
            if code_j != 0 or not joints:
                return None
            joints_rad = list(joints)
        return self._effective_tip_pose_mm_deg(joints_rad)

    def _on_state_enter(self, old: TeleopState, new: TeleopState, reason: str):
        super()._on_state_enter(old, new, reason)
        if new in (TeleopState.TIP_LOCK_ACTIVE, TeleopState.ENTRY_LOCK_ACTIVE):
            self._v7_constrained_settle_until_s = time.monotonic() + self.v7_mode_switch_settle_s
        else:
            self._v7_constrained_settle_until_s = 0.0
        self._v7_idle_since_s = 0.0
        if new == TeleopState.TIP_LOCK_ACTIVE:
            # Tighten tip-lock correction for v7 and re-anchor on state entry
            # to avoid visible step/shift right after capture.
            self.tip_lock.cfg.correction_gain = float(self.v7_tip_lock_correction_gain)
            self.tip_lock.cfg.correction_deadband_mm = float(self.v7_tip_lock_correction_deadband_mm)
            if self.v7_tip_lock_reanchor_on_enter:
                try:
                    code_j, joints = self.arm.get_servo_angle(is_radian=True)
                except Exception:
                    code_j, joints = -1, None
                if code_j == 0 and joints:
                    pose = self._tip_lock_feedback_pose_mm_deg(list(joints))
                    if pose:
                        self.tip_lock.locked_pos_mm = [float(pose[0]), float(pose[1]), float(pose[2])]
            self.rate_limiter.reset()

    def _planar_sticks(self, inp: InputSnapshot) -> Tuple[float, float]:
        # Operator-facing convention used in v6:
        # return (forward_cmd, right_cmd)
        return float(inp.ly), float(inp.lx)

    def _planar_sticks_raw(self, inp: InputSnapshot) -> Tuple[float, float]:
        return float(inp.raw_ly), float(inp.raw_lx)

    def _operator_is_idle(self, inp: InputSnapshot) -> bool:
        dead = 0.10
        hard_dead = dead * 1.35
        fwd_cmd, right_cmd = self._planar_sticks(inp)
        raw_fwd_cmd, raw_right_cmd = self._planar_sticks_raw(inp)
        return (
            abs(fwd_cmd) < dead
            and abs(right_cmd) < dead
            and abs(raw_fwd_cmd) < hard_dead
            and abs(raw_right_cmd) < hard_dead
            and abs(float(inp.lt)) < dead
            and abs(float(inp.rt)) < dead
            and abs(float(inp.ry)) < dead
        )

    def _clamp_linear_speed(self, v_xyz: List[float], max_mm_s: float) -> List[float]:
        v_norm = vec3_norm(v_xyz)
        lim = max(float(max_mm_s), 1e-6)
        if v_norm <= lim:
            return v_xyz
        s = lim / max(v_norm, 1e-9)
        return vec3_scale(v_xyz, s)

    def _ft_force_ratio(self) -> float:
        r = self.ft_guard.last_reading
        cfg = self.ft_guard.cfg
        force_norm = math.sqrt(r.fx ** 2 + r.fy ** 2 + r.fz ** 2)
        ratios = [
            abs(r.fx) / max(cfg.fx_limit_n, 1e-6),
            abs(r.fy) / max(cfg.fy_limit_n, 1e-6),
            abs(r.fz) / max(cfg.fz_limit_n, 1e-6),
            force_norm / max(cfg.force_norm_limit_n, 1e-6),
        ]
        return max(ratios)

    def _ft_haptic_force_n(self, pose_mm_deg: Optional[List[float]]) -> float:
        r = self.ft_guard.last_reading
        cfg = self.ft_guard.cfg
        f = [float(r.fx), float(r.fy), float(r.fz)]
        force_norm = math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])

        src = self.v7_ft_haptic_force_source
        if src == "norm" or pose_mm_deg is None:
            return force_norm
        if src == "max_axis":
            return max(abs(f[0]), abs(f[1]), abs(f[2]))

        # insertion_axis: project FT onto tool axis so operator feels drilling load
        # instead of unrelated lateral spikes.
        tool_z = tool_z_axis_from_rpy_deg(
            float(pose_mm_deg[3]), float(pose_mm_deg[4]), float(pose_mm_deg[5])
        )
        f_axis = abs((f[0] * tool_z[0]) + (f[1] * tool_z[1]) + (f[2] * tool_z[2]))
        return f_axis

    def _set_ft_trigger_feedback(self, force_ratio: float, side: str):
        gain = max(float(self.v7_haptic_feedback_gain), 0.0)
        ratio = clamp(
            force_ratio * gain, 0.0, max(self.v7_ft_haptic_max_trigger_resistance, 0.01)
        )
        # Route adaptive trigger resistance to the active insertion side.
        if side == "left":
            if hasattr(self.input, "set_drilling_feedback_left"):
                self.input.set_drilling_feedback_left(ratio)
            elif hasattr(self.input, "set_trigger_profile_left"):
                self.input.set_trigger_profile_left(start_position=0, strength=int(round(ratio * 8.0)))
            if hasattr(self.input, "set_drilling_feedback"):
                self.input.set_drilling_feedback(0.0)
        else:
            if hasattr(self.input, "set_drilling_feedback"):
                self.input.set_drilling_feedback(ratio)
            if hasattr(self.input, "set_drilling_feedback_left"):
                self.input.set_drilling_feedback_left(0.0)

    def _refresh_ft_fallback(self, now_s: float):
        """Update FT guard directly from SDK when ROS FT topic is stale/missing."""
        if self.dry_run or not self.v7_ft_poll_fallback_enable:
            return
        if self.ft_guard.data_state(now_s) == FTDataState.FRESH:
            return
        if (now_s - self._v7_last_ft_poll_s) < self.v7_ft_poll_min_interval_s:
            return
        self._v7_last_ft_poll_s = now_s

        try:
            code, ft_data = self.arm.get_ft_sensor_data()
        except Exception as exc:
            self._warn_throttle("v7_ft_poll_exc", f"[V7] direct FT poll exception: {exc}", 2.0)
            return

        if code != 0 or not ft_data or len(ft_data) < 6:
            if self.v7_ft_poll_auto_init and not self._v7_ft_auto_init_attempted:
                self._v7_ft_auto_init_attempted = True
                try:
                    if hasattr(self.arm, "set_tgpio_modbus_baudrate"):
                        self.arm.set_tgpio_modbus_baudrate(int(self.v7_ft_modbus_baudrate))
                    if hasattr(self.arm, "ft_sensor_enable"):
                        self.arm.ft_sensor_enable(1)
                    self.get_logger().warn(
                        "[V7] direct FT poll unavailable; attempted FT sensor auto-init"
                    )
                except Exception as exc:
                    self._warn_throttle(
                        "v7_ft_poll_init_exc",
                        f"[V7] FT auto-init attempt failed: {exc}",
                        2.0,
                    )
            self._warn_throttle(
                "v7_ft_poll_bad",
                f"[V7] direct FT poll unavailable (code={code})",
                2.0,
            )
            return

        self.ft_guard.update(
            fx=float(ft_data[0]),
            fy=float(ft_data[1]),
            fz=float(ft_data[2]),
            tx=float(ft_data[3]),
            ty=float(ft_data[4]),
            tz=float(ft_data[5]),
            now_s=now_s,
        )
        if not self._v7_direct_ft_logged:
            self._v7_direct_ft_logged = True
            self.get_logger().info("[V7] FT fallback active: using direct SDK FT polling")

    def _update_motion_haptics(self, cmd: List[float], inp: InputSnapshot):
        if not self.v7_motion_haptic_enable:
            return
        if self._v7_prev_ft_feedback > 0.05:
            # Keep force-based haptics dominant during contact.
            self._v7_last_motion_feedback = 0.0
            return

        dead = 0.05
        # In current live setup, route inward tactile cue to R2 side.
        if float(inp.rt) > float(inp.lt) + dead:
            self._v7_depth_active_side = "right"
        elif float(inp.lt) > float(inp.rt) + dead:
            self._v7_depth_active_side = "left"

        lin_norm = math.sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1] + cmd[2] * cmd[2])
        lin_ratio = lin_norm / max(self.max_linear_mm_s, 1e-6)
        depth_ratio = abs(float(inp.lt - inp.rt))
        motion_ratio = clamp(max(lin_ratio, depth_ratio), 0.0, 1.0)

        if motion_ratio <= self.v7_motion_haptic_deadband_ratio:
            if self._v7_last_motion_feedback > 0.0:
                try:
                    if hasattr(self.input, "set_trigger_vibration_left"):
                        self.input.set_trigger_vibration_left(intensity=0.0)
                    if hasattr(self.input, "set_trigger_vibration"):
                        self.input.set_trigger_vibration(intensity=0.0)
                except Exception:
                    pass
            self._v7_last_motion_feedback = 0.0
            return

        intensity = clamp(self.v7_motion_haptic_gain * math.sqrt(motion_ratio), 0.0, 1.0)
        now_s = self.get_clock().now().nanoseconds / 1e9
        if now_s - self._v7_last_motion_rumble_s >= self.v7_motion_haptic_rumble_interval_s:
            self._v7_last_motion_rumble_s = now_s
            try:
                self.input.rumble(strength=clamp(0.20 + 0.60 * intensity, 0.0, 1.0), duration_ms=80)
            except Exception:
                pass
        try:
            if self._v7_depth_active_side == "left" and hasattr(self.input, "set_trigger_vibration_left"):
                self.input.set_trigger_vibration_left(intensity=intensity)
                if hasattr(self.input, "set_trigger_vibration"):
                    self.input.set_trigger_vibration(intensity=0.0)
            elif hasattr(self.input, "set_trigger_vibration"):
                self.input.set_trigger_vibration(intensity=intensity)
                if hasattr(self.input, "set_trigger_vibration_left"):
                    self.input.set_trigger_vibration_left(intensity=0.0)
        except Exception:
            pass
        self._v7_last_motion_feedback = intensity

    def _update_ft_force_haptics(self, inp: InputSnapshot):
        if not self.v7_ft_haptic_enable:
            return
        now_s = self.get_clock().now().nanoseconds / 1e9
        self._refresh_ft_fallback(now_s)
        if self.ft_guard.data_state(now_s) != FTDataState.FRESH:
            self._warn_throttle("v7_ft_stale_haptic", "[V7] FT data stale for haptics", 2.0)
            self._set_ft_trigger_feedback(0.0, self._v7_depth_active_side)
            self._v7_ft_feedback_filtered = 0.0
            self._v7_ft_force_filtered_n = 0.0
            self._v7_prev_ft_feedback = 0.0
            return

        dead = 0.05
        # In current live setup, route inward tactile cue to R2 side.
        if float(inp.rt) > float(inp.lt) + dead:
            self._v7_depth_active_side = "right"
        elif float(inp.lt) > float(inp.rt) + dead:
            self._v7_depth_active_side = "left"

        try:
            code_p, pose_sdk = self.arm.get_position(is_radian=False)
        except Exception:
            code_p, pose_sdk = -1, None
        pose_for_haptic = [float(pose_sdk[i]) for i in range(6)] if (code_p == 0 and pose_sdk) else None
        force_n = self._ft_haptic_force_n(pose_for_haptic)
        a_force = self.v7_ft_haptic_force_alpha
        self._v7_ft_force_filtered_n = ((1.0 - a_force) * self._v7_ft_force_filtered_n) + (a_force * force_n)
        force_eff_n = max(0.0, self._v7_ft_force_filtered_n - self.v7_ft_haptic_deadband_n)
        full_scale_n = self.v7_ft_haptic_full_scale_n if self.v7_ft_haptic_full_scale_n > 0.0 else float(self.ft_guard.cfg.fz_limit_n)
        full_scale_n = max(full_scale_n, 1e-6)
        force_ratio = clamp(force_eff_n / full_scale_n, 0.0, 1.0)
        warn = clamp(self.v7_ft_haptic_warn_ratio, 0.0, 0.95)
        release = clamp(self.v7_ft_haptic_release_ratio, 0.0, 0.50)
        span = max(1.0 - warn, 1e-6)
        feedback_raw = clamp((force_ratio - warn) / span, 0.0, 1.0)
        # Shape >1.0 makes resistance build-up gradual (no abrupt hard wall).
        feedback_raw = feedback_raw ** self.v7_ft_haptic_shape_exp
        # Smooth trigger resistance to avoid sudden jumps under noisy FT readings.
        a = self.v7_ft_haptic_smooth_alpha
        feedback_ratio = (1.0 - a) * self._v7_ft_feedback_filtered + a * feedback_raw
        feedback_ratio = clamp(feedback_ratio, 0.0, self.v7_ft_haptic_max_trigger_resistance)
        self._v7_ft_feedback_filtered = feedback_ratio
        self._set_ft_trigger_feedback(feedback_ratio, self._v7_depth_active_side)

        if self.v7_ft_haptic_rumble_enable and feedback_ratio > 0.08:
            if now_s - self._v7_last_ft_rumble_s >= max(self.v7_ft_haptic_rumble_interval_s, 0.05):
                self._v7_last_ft_rumble_s = now_s
                try:
                    self.input.rumble(
                        strength=clamp(0.18 + 0.55 * feedback_ratio, 0.0, 1.0),
                        duration_ms=80,
                    )
                except Exception:
                    pass
            try:
                vib = clamp(0.10 + 0.55 * feedback_ratio, 0.0, 1.0)
                if self._v7_depth_active_side == "left" and hasattr(self.input, "set_trigger_vibration_left"):
                    self.input.set_trigger_vibration_left(intensity=vib)
                elif hasattr(self.input, "set_trigger_vibration"):
                    self.input.set_trigger_vibration(intensity=vib)
            except Exception:
                pass

        if self._v7_prev_ft_feedback >= 0.25 and feedback_ratio <= release:
            self._pulse_haptic(0.75, 120, 0.0)
        self._v7_prev_ft_feedback = feedback_ratio

    def _retreat_cmd_from_force_limit(self, cmd: List[float], fx: float, fy: float, fz: float) -> List[float]:
        lin = [float(cmd[0]), float(cmd[1]), float(cmd[2])]
        lin_norm = math.sqrt(lin[0] * lin[0] + lin[1] * lin[1] + lin[2] * lin[2])
        f_norm = math.sqrt(fx * fx + fy * fy + fz * fz)
        if lin_norm < 1e-6 or f_norm < 1e-6:
            return [0.0] * 6
        dot = fx * lin[0] + fy * lin[1] + fz * lin[2]
        retreat_cos = (-dot) / max(f_norm * lin_norm, 1e-9)
        if retreat_cos < self.v7_ft_retreat_min_cos:
            return [0.0] * 6
        s = self.v7_ft_retreat_speed_scale * clamp(retreat_cos, 0.0, 1.0)
        return [lin[0] * s, lin[1] * s, lin[2] * s, 0.0, 0.0, 0.0]

    def _log_dualsense_haptics_status(self):
        dev = getattr(self.input, "bound_device_path", None)
        writable = bool(getattr(self.input, "_haptics_writable", False))
        conn_err = getattr(self.input, "_connect_error", None)
        if conn_err:
            self.get_logger().warn(f"[V7][HAPTICS] DualSense connect warning: {conn_err}")
        self.get_logger().info(
            f"[V7][HAPTICS] DualSense device={dev or 'unknown'} write_access={'yes' if writable else 'no'}"
        )
        if self.v7_haptics_boot_test and writable:
            self._pulse_haptic(1.0, 220, 0.0)
            self._set_ft_trigger_feedback(0.55, self._v7_depth_active_side)
            if hasattr(self.input, "set_trigger_vibration"):
                try:
                    self.input.set_trigger_vibration(intensity=0.45)
                except Exception:
                    pass
            self.get_logger().info("[V7][HAPTICS] Boot test pulse sent")

    def _pulse_haptic(self, strength: float = 0.7, duration_ms: int = 100, cooldown_s: float = 0.4):
        now = time.monotonic()
        if now - self._last_haptic_s < cooldown_s:
            return
        self._last_haptic_s = now
        eff_strength = clamp(max(float(strength), 0.30) * max(self.v7_haptic_event_gain, 0.0), 0.0, 1.0)
        eff_duration = max(int(duration_ms), int(self.v7_haptic_event_min_duration_ms))
        try:
            self.input.rumble(strength=eff_strength, duration_ms=eff_duration)
        except Exception:
            pass
        try:
            if hasattr(self.input, "set_trigger_vibration"):
                self.input.set_trigger_vibration(intensity=eff_strength)
            if hasattr(self.input, "set_trigger_profile"):
                s = int(round(eff_strength * 8.0))
                self.input.set_trigger_profile(start_position=0, strength=s)
        except Exception:
            pass

    def _apply_global_ft_guard(self, cmd: List[float]) -> List[float]:
        if not self.v7_ft_global_stop_enable:
            return cmd
        if all(abs(c) < 1e-6 for c in cmd):
            return cmd
        now_s = self.get_clock().now().nanoseconds / 1e9
        self._refresh_ft_fallback(now_s)
        ds = self.ft_guard.data_state(now_s)
        cfg = self.ft_guard.cfg
        if ds != FTDataState.FRESH:
            if cfg.stale_policy == StalePolicy.BLOCK_INWARD:
                return [0.0] * 6
            return cmd

        r = self.ft_guard.last_reading
        force_norm = math.sqrt(r.fx * r.fx + r.fy * r.fy + r.fz * r.fz)
        torque_norm = math.sqrt(r.tx * r.tx + r.ty * r.ty + r.tz * r.tz)
        ratios = [
            abs(r.fx) / max(cfg.fx_limit_n, 1e-6),
            abs(r.fy) / max(cfg.fy_limit_n, 1e-6),
            abs(r.fz) / max(cfg.fz_limit_n, 1e-6),
            abs(r.tx) / max(cfg.tx_limit_nm, 1e-6),
            abs(r.ty) / max(cfg.ty_limit_nm, 1e-6),
            abs(r.tz) / max(cfg.tz_limit_nm, 1e-6),
            force_norm / max(cfg.force_norm_limit_n, 1e-6),
            torque_norm / max(cfg.torque_norm_limit_nm, 1e-6),
        ]
        max_ratio = max(ratios)
        if max_ratio >= 1.0:
            self._warn_throttle("v7_ft_global_block", f"[V7] FT global stop ratio={max_ratio:.2f}", 0.35)
            retreat_cmd = [0.0] * 6
            if self.v7_ft_allow_retreat_on_limit:
                retreat_cmd = self._retreat_cmd_from_force_limit(cmd, r.fx, r.fy, r.fz)
            if any(abs(c) > 1e-6 for c in retreat_cmd):
                self._warn_throttle(
                    "v7_ft_global_retreat",
                    f"[V7] FT limit active: retreat motion allowed (ratio={max_ratio:.2f})",
                    0.7,
                )
                return retreat_cmd
            self._pulse_haptic(0.95, 110, 0.25)
            return [0.0] * 6
        if self.v7_ft_global_soft_slowdown and max_ratio >= cfg.warning_ratio:
            span = max(1.0 - cfg.warning_ratio, 1e-6)
            scale = clamp(1.0 - (max_ratio - cfg.warning_ratio) / span, 0.0, 1.0)
            return [c * scale for c in cmd]
        return cmd

    def _vel_free_teleop(self, inp: InputSnapshot, scale: float, joints_rad: List[float]) -> List[float]:
        sg = self.sigmoid_gain
        fwd_cmd, right_cmd = self._planar_sticks(inp)
        vx = sigmoid_shape(fwd_cmd, sg) * self.max_linear_mm_s * scale
        vy = sigmoid_shape(right_cmd, sg) * self.max_linear_mm_s * scale
        # Physical mapping in current setup: R2 = insertion/down (IN), L2 = extraction/up (OUT).
        depth_input = sigmoid_shape(float(inp.lt - inp.rt), sg)
        depth_speed = depth_input * self.max_z_mm_s
        insertion_sign = (
            -1.0 if getattr(self.entry_lock.cfg, "insertion_along_neg_tool_z", True) else 1.0
        )
        if abs(depth_speed) < 1e-6:
            return [vx, vy, 0.0, 0.0, 0.0, 0.0]
        if self.v7_use_tool_z_depth_in_free:
            pose = self._effective_tip_pose_mm_deg(joints_rad)
            if pose:
                tool_z = tool_z_axis_from_rpy_deg(pose[3], pose[4], pose[5])
                v_ins = vec3_scale(tool_z, insertion_sign * depth_speed * scale)
                return [vx + v_ins[0], vy + v_ins[1], v_ins[2], 0.0, 0.0, 0.0]
        return [vx, vy, insertion_sign * depth_speed * scale, 0.0, 0.0, 0.0]

    def _vel_tip_lock(self, inp: InputSnapshot, scale: float, joints_rad: List[float]) -> List[float]:
        if time.monotonic() < self._v7_constrained_settle_until_s:
            return [0.0] * 6
        pose = self._tip_lock_feedback_pose_mm_deg(joints_rad)
        if not pose:
            return [0.0] * 6
        dead = 0.10
        hard_dead = dead * 1.25
        fwd_cmd, right_cmd = self._planar_sticks(inp)
        raw_fwd_cmd, raw_right_cmd = self._planar_sticks_raw(inp)
        rt = float(inp.rt) if float(inp.rt) > dead else 0.0
        lt = float(inp.lt) if float(inp.lt) > dead else 0.0
        # Physical mapping in current setup: R2 = insertion/down (IN), L2 = extraction/up (OUT).
        depth_axis = lt - rt
        if (
            abs(fwd_cmd) < dead
            and abs(right_cmd) < dead
            and abs(raw_fwd_cmd) < hard_dead
            and abs(raw_right_cmd) < hard_dead
            and abs(depth_axis) < dead
        ):
            now_mono = time.monotonic()
            if self._v7_idle_since_s <= 0.0:
                self._v7_idle_since_s = now_mono
            idle_for_s = now_mono - self._v7_idle_since_s
            # Re-anchor only if explicitly enabled, operator stayed idle long enough,
            # and lock error is already small. This prevents lock-point jumps.
            if (
                self.v7_tip_idle_reanchor
                and self.tip_lock.active
                and idle_for_s >= self.v7_tip_idle_reanchor_dwell_s
                and float(getattr(self.tip_lock, "last_pos_err_mm", 0.0)) <= self.v7_tip_idle_reanchor_max_err_mm
            ):
                self.tip_lock.locked_pos_mm = [float(pose[0]), float(pose[1]), float(pose[2])]
            self.rate_limiter.reset()
            return [0.0] * 6
        self._v7_idle_since_s = 0.0

        lin_v, ang_v = self.tip_lock.compute_velocity(pose, fwd_cmd, right_cmd)
        tip_lin_cap = float(self.v7_tip_lock_max_linear_mm_s)
        rec_err = float(self.v7_tip_lock_recovery_err_mm)
        rec_gain = float(self.v7_tip_lock_recovery_gain_mm_s_per_mm)
        min_ang_scale = float(self.v7_tip_lock_min_angular_scale)
        tip_err = float(getattr(self.tip_lock, "last_pos_err_mm", 0.0))
        if tip_err > rec_err:
            over = tip_err - rec_err
            dynamic_cap = tip_lin_cap + rec_gain * over
            ang_scale = max(min_ang_scale, rec_err / max(tip_err, 1e-6))
            ang_v = [ang_scale * a for a in ang_v]
        else:
            dynamic_cap = tip_lin_cap

        # Additional "error brake": if tip drift grows, strongly damp angular
        # command so linear correction can recenter the lock point.
        if self.v7_tip_lock_error_brake_enable and tip_err > self.v7_tip_lock_error_brake_mm:
            ratio = self.v7_tip_lock_error_brake_mm / max(tip_err, 1e-6)
            brake_scale = clamp(ratio ** self.v7_tip_lock_error_brake_exp, 0.02, 1.0)
            ang_v = [brake_scale * a for a in ang_v]
        dynamic_cap = min(self.max_linear_mm_s, max(dynamic_cap, 0.5))

        # Compensate rotation-induced tip translation caused by nonzero TCP offset.
        # When the low-level controller effectively rotates about flange, this term
        # cancels the induced tip motion: v = -omega x r_tcp.
        if self.v7_tip_lock_tcp_rotation_comp_enable:
            eff_tcp = self._effective_tcp_translation_mm()
            off_z_mm = float(eff_tcp[2]) if len(eff_tcp) >= 3 else 0.0
            if abs(off_z_mm) > 1e-6:
                tool_z = tool_z_axis_from_rpy_deg(pose[3], pose[4], pose[5])
                r_tcp = vec3_scale(tool_z, off_z_mm)
                w_rad = [
                    math.radians(float(ang_v[0])),
                    math.radians(float(ang_v[1])),
                    math.radians(float(ang_v[2])),
                ]
                v_rot_comp = vec3_scale(
                    vec3_cross(w_rad, r_tcp), -float(self.v7_tip_lock_tcp_rotation_comp_gain)
                )
                lin_v = vec3_add(lin_v, v_rot_comp)

        try:
            code_p, pose_sdk = self.arm.get_position(is_radian=False)
        except Exception:
            code_p, pose_sdk = -1, None
        if code_p == 0 and pose_sdk:
            d = [
                float(pose[0] - pose_sdk[0]),
                float(pose[1] - pose_sdk[1]),
                float(pose[2] - pose_sdk[2]),
            ]
            d_norm = vec3_norm(d)
            w_rad = [
                math.radians(float(ang_v[0])),
                math.radians(float(ang_v[1])),
                math.radians(float(ang_v[2])),
            ]
            w_norm = vec3_norm(w_rad)
            if d_norm > 1e-3 and w_norm > 1e-6:
                w_cap = (0.85 * dynamic_cap) / d_norm
                if w_norm > w_cap and w_cap > 1e-6:
                    s = w_cap / w_norm
                    ang_v = [a * s for a in ang_v]
                    w_rad = [w * s for w in w_rad]
                v_ff = vec3_scale(vec3_cross(w_rad, d), -1.0)
                lin_v = vec3_add(lin_v, v_ff)
        # Allow insertion/extraction while fixed-tip mode is active by shifting
        # the lock point along tool axis with trigger command.
        depth_input = sigmoid_shape(depth_axis, self.sigmoid_gain)
        depth_speed = depth_input * self.max_depth_mm_s * self.depth_gain
        if abs(depth_speed) > 0.001:
            tool_z = tool_z_axis_from_rpy_deg(pose[3], pose[4], pose[5])
            insertion_sign = (
                -1.0 if getattr(self.entry_lock.cfg, "insertion_along_neg_tool_z", True) else 1.0
            )
            v_ins = vec3_scale(tool_z, insertion_sign * depth_speed)
            lin_v = vec3_add(lin_v, v_ins)
            if self.tip_lock.active:
                self.tip_lock.locked_pos_mm = vec3_add(
                    self.tip_lock.locked_pos_mm,
                    vec3_scale(tool_z, insertion_sign * depth_speed * self.dt),
                )
        lin_v = self._clamp_linear_speed(lin_v, dynamic_cap)
        if self.v7_tip_lock_debug:
            now_s = self.get_clock().now().nanoseconds / 1e9
            if now_s - self._v7_last_tip_dbg_s >= self.v7_tip_lock_debug_interval_s:
                self._v7_last_tip_dbg_s = now_s
                lock = list(getattr(self.tip_lock, "locked_pos_mm", [0.0, 0.0, 0.0]))
                err_v = [float(lock[i] - pose[i]) for i in range(3)]
                err_n = math.sqrt(err_v[0] * err_v[0] + err_v[1] * err_v[1] + err_v[2] * err_v[2])
                if err_n >= self.v7_tip_lock_debug_err_threshold_mm:
                    try:
                        code_sdk, pose_sdk = self.arm.get_position(is_radian=False)
                    except Exception:
                        code_sdk, pose_sdk = -1, None
                    if code_sdk == 0 and pose_sdk:
                        model_vs_sdk = [
                            float(pose[0] - pose_sdk[0]),
                            float(pose[1] - pose_sdk[1]),
                            float(pose[2] - pose_sdk[2]),
                        ]
                        model_vs_sdk_n = math.sqrt(
                            model_vs_sdk[0] * model_vs_sdk[0]
                            + model_vs_sdk[1] * model_vs_sdk[1]
                            + model_vs_sdk[2] * model_vs_sdk[2]
                        )
                    else:
                        model_vs_sdk = [0.0, 0.0, 0.0]
                        model_vs_sdk_n = -1.0
                    self.get_logger().warn(
                        "[V7][TIPDBG] "
                        f"err_mm={err_n:.2f} "
                        f"err_v=({err_v[0]:+.2f},{err_v[1]:+.2f},{err_v[2]:+.2f}) "
                        f"lock=({lock[0]:.1f},{lock[1]:.1f},{lock[2]:.1f}) "
                        f"tip=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
                        f"lin_cmd=({lin_v[0]:+.2f},{lin_v[1]:+.2f},{lin_v[2]:+.2f}) "
                        f"ang_cmd=({ang_v[0]:+.2f},{ang_v[1]:+.2f},{ang_v[2]:+.2f}) "
                        f"m_vs_sdk_mm={model_vs_sdk_n:.2f} "
                        f"m_vs_sdk_v=({model_vs_sdk[0]:+.2f},{model_vs_sdk[1]:+.2f},{model_vs_sdk[2]:+.2f}) "
                        f"depth_axis={depth_axis:+.3f}"
                    )
        return [
            lin_v[0] * scale, lin_v[1] * scale, lin_v[2] * scale,
            ang_v[0] * scale, ang_v[1] * scale, ang_v[2] * scale,
        ]

    def _compute_velocity(self, inp: InputSnapshot, scale: float, joints_rad: List[float]) -> List[float]:
        st = self.sm.state
        if st in (
            TeleopState.FREE_TELEOP,
            TeleopState.TIP_LOCK_ACTIVE,
            TeleopState.ENTRY_LOCK_ACTIVE,
        ) and self._operator_is_idle(inp):
            if self._v7_idle_since_s <= 0.0:
                self._v7_idle_since_s = time.monotonic()
            self.rate_limiter.reset()
            cmd = [0.0] * 6
            self._update_ft_force_haptics(inp)
            self._update_motion_haptics(cmd, inp)
            return self._apply_global_ft_guard(cmd)
        self._v7_idle_since_s = 0.0
        if st == TeleopState.FREE_TELEOP:
            cmd = self._vel_free_teleop(inp, scale, joints_rad)
        elif st == TeleopState.TIP_LOCK_ACTIVE:
            cmd = self._vel_tip_lock(inp, scale, joints_rad)
        elif st == TeleopState.ENTRY_LOCK_ACTIVE and self.v7_enable_rcm_placeholder:
            cmd = self._vel_entry_lock(inp, scale, joints_rad)
        else:
            cmd = [0.0] * 6
        self._update_ft_force_haptics(inp)
        self._update_motion_haptics(cmd, inp)
        return self._apply_global_ft_guard(cmd)

    def _do_tip_lock_capture(self):
        """Capture Tip-Lock anchor using selected v7 pose source."""
        try:
            code_j, joints = self.arm.get_servo_angle(is_radian=True)
        except Exception:
            code_j, joints = -1, None
        joints_for_fallback = list(joints) if (code_j == 0 and joints) else None
        pose = self._tip_lock_feedback_pose_mm_deg(joints_for_fallback)
        if not pose:
            self._warn_throttle("tip_cap_v7", "[V7] Tip-Lock capture: pose unavailable", 0.5)
            self.sm.transition_to(TeleopState.FREE_TELEOP, "capture_pose_fail")
            return
        self.tip_lock.capture(pose)
        self.get_logger().info(
            f"[V7] TIP-LOCK captured ({self.v7_tip_lock_pose_source}) at "
            f"({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f})"
        )
        self._pulse_haptic(1.0, 180, 0.0)
        self.sm.transition_to(TeleopState.TIP_LOCK_ACTIVE, "tip_captured")

    def _handle_button_actions(self, inp: InputSnapshot):
        st = self.sm.state
        if st == TeleopState.FREE_TELEOP:
            if inp.x_edge:
                self._tare_sensor_action()
            elif inp.lb_edge:
                if not self.enable_initial_pose_action:
                    self._warn_throttle(
                        "v7_init_pose_disabled",
                        "[V7] Initial-pose action disabled",
                        1.0,
                    )
                    return
                self._start_initial_pose()
            elif inp.a_edge:
                if not self._constrained_modes_enabled:
                    self._warn_throttle("v7_kin_nv", "[V7] Fixed-tip blocked: kinematic model not validated", 1.0)
                    return
                self.sm.transition_to(TeleopState.TIP_LOCK_CAPTURE, "v7_btn_cross_fixed_tip")
            elif inp.b_edge:
                if self.v7_enable_rcm_placeholder:
                    if not self._constrained_modes_enabled:
                        self._warn_throttle("v7_kin_nv", "[V7] Entry-Lock blocked: kinematic model not validated", 1.0)
                        return
                    self.sm.transition_to(TeleopState.ENTRY_LOCK_CAPTURE, "v7_btn_circle_entry_lock")
                else:
                    self._warn_throttle(
                        "v7_rcm_off",
                        "[V7] RCM path disabled (v7_enable_rcm_placeholder:=false)",
                        1.0,
                    )
            elif inp.y_edge:
                self._start_alignment()
        elif st == TeleopState.TIP_LOCK_ACTIVE:
            if inp.a_edge:
                self.sm.transition_to(TeleopState.FREE_TELEOP, "v7_fixed_tip_unlock")
        elif st == TeleopState.ENTRY_LOCK_ACTIVE:
            if inp.b_edge:
                self.sm.transition_to(TeleopState.FREE_TELEOP, "v7_entry_unlock")


def main(args=None):
    if args is None:
        import sys

        args = [a for a in sys.argv if a.strip()]
    rclpy.init(args=args)
    node = TeleopV7Node()
    try:
        ctx = rclpy.get_default_context()
        ctx.on_shutdown(lambda: node._request_motion_abort())
    except AttributeError:
        pass
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_hook()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

