#!/usr/bin/env python3
"""Offline verification suite for teleopv4.

Run (after sourcing the workspace):

    python3 -m curac_teleop.teleopv4.offline_verify

File name avoids the ``test_*.py`` pattern so pytest does not collect it as
a test module during ``colcon test``.

No robot, joystick, ROS, or FT sensor required.
"""

import math
import sys
import time
import traceback

PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        msg = f"  FAIL  {name}  {detail}"
        print(msg)
        ERRORS.append(msg)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =================================================================
# 1. math_utils
# =================================================================
section("math_utils")
from curac_teleop.teleopv4.math_utils import (
    clamp, deadzone_map, sigmoid_shape,
    vec3_norm, vec3_normalize, vec3_cross, vec3_dot, vec3_scale,
    vec3_add, vec3_sub, skew_symmetric,
    rpy_deg_to_rotmat, rotmat_to_rpy_deg, tool_z_axis_from_rpy_deg,
)
import numpy as np

check("clamp basic", clamp(5, 0, 3) == 3)
check("clamp neg", clamp(-2, -1, 1) == -1)
check("clamp passthru", clamp(0.5, 0, 1) == 0.5)

check("deadzone inside", deadzone_map(0.05, 0.1) == 0.0)
check("deadzone outside", abs(deadzone_map(0.55, 0.1) - 0.5) < 1e-6,
      f"got {deadzone_map(0.55, 0.1)}")
check("deadzone neg", deadzone_map(-0.55, 0.1) < 0)
check("deadzone full", abs(deadzone_map(1.0, 0.1) - 1.0) < 1e-6)

check("sigmoid(0)=0", abs(sigmoid_shape(0.0)) < 1e-9)
check("sigmoid symmetric", abs(sigmoid_shape(0.5) + sigmoid_shape(-0.5)) < 1e-9)
check("sigmoid(1)~1", abs(sigmoid_shape(1.0) - 1.0) < 1e-6)
check("sigmoid gain=0 linear", abs(sigmoid_shape(0.5, 0.0) - 0.5) < 1e-3)

check("vec3_norm", abs(vec3_norm([3, 4, 0]) - 5.0) < 1e-9)
check("vec3_normalize", abs(vec3_norm(vec3_normalize([3, 4, 0])) - 1.0) < 1e-9)
check("vec3_normalize zero", vec3_normalize([0, 0, 0]) == [0, 0, 0])
check("vec3_cross x*y=z", vec3_cross([1, 0, 0], [0, 1, 0]) == [0, 0, 1])
check("vec3_dot", abs(vec3_dot([1, 2, 3], [4, 5, 6]) - 32) < 1e-9)
check("vec3_scale", vec3_scale([1, 2, 3], 2) == [2, 4, 6])
check("vec3_add", vec3_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9])
check("vec3_sub", vec3_sub([4, 5, 6], [1, 2, 3]) == [3, 3, 3])

S = skew_symmetric([1, 2, 3])
check("skew antisymmetric", np.allclose(S, -S.T))
w = np.array([1, 2, 3])
v = np.array([4, 5, 6])
check("skew cross product", np.allclose(S @ v, np.cross(w, v)))

R_id = rpy_deg_to_rotmat(0, 0, 0)
check("rpy identity", np.allclose(R_id, np.eye(3), atol=1e-9))

R_yaw90 = rpy_deg_to_rotmat(0, 0, 90)
check("rpy yaw90 x->-y", abs(R_yaw90[1, 0] - 1.0) < 1e-6,
      f"R[1,0]={R_yaw90[1,0]}")

# Roundtrip RPY
for rpy in [(30, 45, 60), (0, 0, 0), (180, 0, 0), (-45, 20, -170)]:
    R = rpy_deg_to_rotmat(*rpy)
    r2 = rotmat_to_rpy_deg(R)
    R2 = rpy_deg_to_rotmat(*r2)
    check(f"rpy roundtrip {rpy}", np.allclose(R, R2, atol=1e-6),
          f"got {r2}")

tz = tool_z_axis_from_rpy_deg(0, 0, 0)
check("tool_z identity", abs(tz[2] - 1.0) < 1e-9, f"got {tz}")

tz180 = tool_z_axis_from_rpy_deg(180, 0, 0)
check("tool_z roll180", abs(tz180[2] - (-1.0)) < 1e-6, f"got {tz180}")


# =================================================================
# 2. state_machine
# =================================================================
section("state_machine")
from curac_teleop.teleopv4.state_machine import (
    TeleopState, StateMachine, ALLOWED_TRANSITIONS,
    MOTION_STATES, CONSTRAINED_STATES,
)

check("8 states", len(TeleopState) == 8)
check("8 transition rules", len(ALLOWED_TRANSITIONS) == 8)

sm = StateMachine()
check("initial IDLE", sm.state == TeleopState.IDLE)
check("IDLE not motion", not sm.is_motion_allowed())
check("IDLE not constrained", not sm.is_constrained())

ok = sm.transition_to(TeleopState.FREE_TELEOP, "test")
check("IDLE->FREE_TELEOP ok", ok)
check("is_motion_allowed FREE", sm.is_motion_allowed())

bad = sm.transition_to(TeleopState.ENTRY_LOCK_ACTIVE, "illegal")
check("FREE->ENTRY_ACTIVE blocked", not bad)
check("still FREE_TELEOP", sm.state == TeleopState.FREE_TELEOP)

ok2 = sm.transition_to(TeleopState.TIP_LOCK_CAPTURE, "btn_a")
check("FREE->TIP_CAPTURE ok", ok2)
ok3 = sm.transition_to(TeleopState.TIP_LOCK_ACTIVE, "captured")
check("TIP_CAPTURE->TIP_ACTIVE ok", ok3)
check("constrained in TIP_ACTIVE", sm.is_constrained())

ok4 = sm.transition_to(TeleopState.FREE_TELEOP, "unlock")
check("TIP_ACTIVE->FREE ok", ok4)

sm2 = StateMachine()
sm2.transition_to(TeleopState.FREE_TELEOP, "test")
sm2.transition_to(TeleopState.ENTRY_LOCK_CAPTURE, "btn_b")
sm2.transition_to(TeleopState.ENTRY_LOCK_ACTIVE, "captured")
check("ENTRY_ACTIVE constrained", sm2.is_constrained())
sm2.transition_to(TeleopState.IDLE, "deadman_release")
check("ENTRY_ACTIVE->IDLE ok", sm2.state == TeleopState.IDLE)

# Zero-on-transition callbacks
exit_calls = []
enter_calls = []
sm3 = StateMachine(
    on_exit=lambda o, n, r: exit_calls.append((o, n, r)),
    on_enter=lambda o, n, r: enter_calls.append((o, n, r)),
)
sm3.transition_to(TeleopState.FREE_TELEOP, "test")
check("exit callback fired", len(exit_calls) == 1)
check("enter callback fired", len(enter_calls) == 1)
check("exit sees old state", exit_calls[0][0] == TeleopState.IDLE)
check("enter sees new state", enter_calls[0][1] == TeleopState.FREE_TELEOP)

# force_fault
sm4 = StateMachine()
sm4.transition_to(TeleopState.FREE_TELEOP, "test")
sm4.force_fault("test_fault")
check("force_fault works", sm4.state == TeleopState.FAULT_LATCHED)
bad2 = sm4.transition_to(TeleopState.FREE_TELEOP, "illegal")
check("FAULT->FREE blocked", not bad2)
ok5 = sm4.transition_to(TeleopState.IDLE, "recovery")
check("FAULT->IDLE ok", ok5)


# =================================================================
# 3. ft_guard
# =================================================================
section("ft_guard")
from curac_teleop.teleopv4.ft_guard import (
    FTGuard, FTGuardConfig, FTDataState, StalePolicy,
)

# No data, allow policy
ft = FTGuard(FTGuardConfig(stale_policy=StalePolicy.ALLOW))
now = time.monotonic()
check("data_state NO_DATA", ft.data_state(now) == FTDataState.NO_DATA)
r = ft.evaluate_depth(10.0, True, now)
check("no_data allow: not blocked", not r.blocked)
check("no_data allow: speed passes", r.allowed_speed == 10.0)

# No data, block_inward policy
ft2 = FTGuard(FTGuardConfig(stale_policy=StalePolicy.BLOCK_INWARD))
r2 = ft2.evaluate_depth(10.0, True, now)
check("no_data block_inward: blocked", r2.blocked)
r2out = ft2.evaluate_depth(-10.0, True, now)
check("no_data block_inward: outward ok", not r2out.blocked)

# Fresh data, within limits
ft3 = FTGuard(FTGuardConfig())
ft3.update(5.0, 3.0, 2.0, 0.1, 0.05, 0.02, now)
check("data_state FRESH", ft3.data_state(now) == FTDataState.FRESH)
r3 = ft3.evaluate_depth(10.0, True, now)
check("within limits: not blocked", not r3.blocked)
check("within limits: full speed", r3.scale == 1.0)

# Stale data
r_stale = ft3.evaluate_depth(10.0, True, now + 5.0)
check("data_state STALE", ft3.data_state(now + 5.0) == FTDataState.STALE)

# Hard block on exceeding limit
ft4 = FTGuard(FTGuardConfig(fx_limit_n=10.0, force_norm_limit_n=30.0))
ft4.update(11.0, 0.0, 0.0, 0.0, 0.0, 0.0, now)
r4 = ft4.evaluate_depth(10.0, True, now)
check("over limit: blocked", r4.blocked)
check("over limit: speed=0", r4.allowed_speed == 0.0)

# Hysteresis: should stay blocked until below hysteresis*limit
ft4.update(9.0, 0.0, 0.0, 0.0, 0.0, 0.0, now + 0.1)
r4b = ft4.evaluate_depth(10.0, True, now + 0.1)
check("hysteresis: still blocked at 90%", r4b.blocked)
ft4.update(7.0, 0.0, 0.0, 0.0, 0.0, 0.0, now + 0.2)
r4c = ft4.evaluate_depth(10.0, True, now + 0.2)
check("hysteresis: unblocked below 80%", not r4c.blocked)

# Soft slowdown in warning zone
ft5 = FTGuard(FTGuardConfig(fx_limit_n=10.0, force_norm_limit_n=30.0,
                             warning_ratio=0.7))
ft5.update(8.5, 0.0, 0.0, 0.0, 0.0, 0.0, now)
r5 = ft5.evaluate_depth(10.0, True, now)
check("warning zone: reduced speed", 0 < r5.scale < 1.0, f"scale={r5.scale}")

# Outward always ok
ft5.update(15.0, 0.0, 0.0, 0.0, 0.0, 0.0, now)
r5out = ft5.evaluate_depth(-10.0, True, now)
check("outward always ok", not r5out.blocked)


# =================================================================
# 4. safety (RateLimiter + JointRiskMonitor)
# =================================================================
section("safety (RateLimiter)")
from curac_teleop.teleopv4.safety import (
    RateLimiter, RateLimiterConfig, JointRiskMonitor, JointRiskConfig,
    XARM7_JOINT_LIMITS,
)

rl = RateLimiter(RateLimiterConfig(
    max_linear_acc_mm_s2=100.0, max_linear_dec_mm_s2=200.0,
    max_angular_acc_deg_s2=100.0, max_angular_dec_deg_s2=200.0,
))

# From zero, command 50mm/s in X.  dt=0.01 -> max step = 100*0.01 = 1mm/s
dt = 0.01
cmd1 = rl.limit([50, 0, 0, 0, 0, 0], dt, False)
check("rate limit: ramps up", 0 < cmd1[0] < 50, f"cmd[0]={cmd1[0]:.3f}")
check("rate limit: step<=1", cmd1[0] <= 1.001, f"cmd[0]={cmd1[0]:.3f}")

# Several steps
for _ in range(200):
    cmd1 = rl.limit([50, 0, 0, 0, 0, 0], dt, False)
check("rate limit: converges", abs(cmd1[0] - 50.0) < 0.1, f"cmd[0]={cmd1[0]:.3f}")

# Reset
rl.reset()
cmd_zero = rl.limit([0, 0, 0, 0, 0, 0], dt, False)
check("rate limit: reset to zero", all(abs(c) < 1e-9 for c in cmd_zero))

# Constrained: unified scaling
rl2 = RateLimiter(RateLimiterConfig(
    max_linear_acc_mm_s2=100.0, max_linear_dec_mm_s2=200.0,
    max_angular_acc_deg_s2=1000.0, max_angular_dec_deg_s2=2000.0,
))
cmd_c = rl2.limit([50, 0, 0, 50, 0, 0], dt, True)
# Linear step limited to 1.0. Angular step could be 10.0 but unified
# scaling should use the MORE restrictive factor.
lin_ratio = cmd_c[0] / max(cmd_c[3], 1e-12) if abs(cmd_c[3]) > 1e-9 else 0
check("unified: lin/ang ratio=1", abs(lin_ratio - 1.0) < 0.01 or abs(cmd_c[3]) < 0.01,
      f"lin={cmd_c[0]:.3f} ang={cmd_c[3]:.3f} ratio={lin_ratio:.3f}")

section("safety (JointRiskMonitor)")

from curac_teleop.teleopv4.kinematics import KDLKinModel
kin = KDLKinModel()
jrm = JointRiskMonitor(JointRiskConfig(), kin_model=kin)

# Comfortable configuration
q_ok = [0.5, -0.3, 0.2, 1.0, -0.5, 0.8, 0.1]
rr = jrm.compute_risk(q_ok)
check("comfortable: scale=1.0", rr.scale == 1.0, f"scale={rr.scale}")
check("comfortable: margin>15deg", rr.worst_joint_margin_deg > 15)

# Near J4 lower limit (-0.19198 rad ~ -11 deg)
q_near = [0.0, 0.0, 0.0, -0.10, 0.0, 0.0, 0.0]
rr2 = jrm.compute_risk(q_near)
check("near J4 limit: scale<1", rr2.scale < 1.0, f"scale={rr2.scale}")

# Past J4 lower limit
q_past = [0.0, 0.0, 0.0, -0.20, 0.0, 0.0, 0.0]
rr3 = jrm.compute_risk(q_past)
check("past J4 limit: scale very low", rr3.scale <= 0.15, f"scale={rr3.scale}")

# Joint limits count
check("7 joint limits", len(XARM7_JOINT_LIMITS) == 7)


# =================================================================
# 5. kinematics
# =================================================================
section("kinematics")

model = KDLKinModel()

# FK at zero
pos0, R0 = model.fk_flange([0.0] * 7)
check("FK zero: position defined",
      abs(pos0[0]) + abs(pos0[1]) + abs(pos0[2]) > 0.01)

# FK at non-zero
q_test = [0.5, -0.3, 0.2, 1.0, -0.5, 0.8, 0.1]
pos1, R1 = model.fk_flange(q_test)
check("FK nonzero: different from zero",
      abs(pos1[0] - pos0[0]) > 0.001 or abs(pos1[1] - pos0[1]) > 0.001)

# Rotation matrix is proper (det=1, R^T R = I)
check("FK: R orthonormal", np.allclose(R1.T @ R1, np.eye(3), atol=1e-6))
check("FK: det(R)=1", abs(np.linalg.det(R1) - 1.0) < 1e-6)

# fk_tool with offset
tcp_off = [0, 0, 0.1]
pos_t, R_t = model.fk_tool(q_test, tcp_off)
offset_vec = np.array(pos_t) - np.array(pos1)
expected_offset = R1 @ np.array(tcp_off)
check("fk_tool: offset correct",
      np.allclose(offset_vec, expected_offset, atol=1e-6))

# Jacobian shape and non-zero
J = model.jacobian(q_test)
check("Jacobian shape (6,7)", J.shape == (6, 7))
check("Jacobian non-zero", np.linalg.norm(J) > 0.01)

# Tool Jacobian
J_tool = model.tool_jacobian(q_test, tcp_off)
check("Tool Jacobian shape (6,7)", J_tool.shape == (6, 7))
check("Tool Jacobian differs from flange",
      not np.allclose(J_tool[:3, :], J[:3, :], atol=1e-9))

# Singularity measure
min_sv, cond, manip = model.singularity_measure(q_test)
check("singular value > 0 at typical", min_sv > 0.01,
      f"min_sv={min_sv}")
check("condition number reasonable", 1.0 < cond < 1000,
      f"cond={cond}")

# Zero config is singular
sv0, _, _ = model.singularity_measure([0.0] * 7)
check("zero config singular", sv0 < 1e-6, f"sv={sv0}")

# Validation helper
pos_v, _ = model.fk_flange(q_test)
fake_sdk = [pos_v[0]*1000, pos_v[1]*1000, pos_v[2]*1000, 0, 0, 0]
vr = model.validate_against_sdk(fake_sdk, q_test)
check("validation: self-consistent", vr.valid and vr.position_error_mm < 0.01)

# Validation: deliberately wrong
bad_sdk = [0, 0, 0, 0, 0, 0]
vr2 = model.validate_against_sdk(bad_sdk, q_test)
check("validation: rejects wrong pose", not vr2.valid)

# TCP offset: SDK pose as tool tip vs fk_tool with same offset (mm)
tcp_mm = [0.0, 0.0, 300.0, 0.0, 0.0, 0.0]
pt_m, _ = model.fk_tool(q_test, [0.0, 0.0, 0.3])
fake_tcp = [pt_m[0] * 1000.0, pt_m[1] * 1000.0, pt_m[2] * 1000.0, 0.0, 0.0, 0.0]
vrt = model.validate_against_sdk(
    fake_tcp, q_test, tolerance_mm=3.0, tcp_offset_mm_deg=tcp_mm,
    virtual_tip_offset_mm=[0.0, 0.0, 0.0], validate_with_tcp=True,
)
check("validation: TCP offset path self-consistent", vrt.valid, vrt.detail)
vrf = model.validate_against_sdk(fake_tcp, q_test, tcp_offset_mm_deg=None)
check("validation: zero-offset path rejects TCP pose", not vrf.valid)

virt_only = [15.0, 0.0, 0.0]
pt2_m, _ = model.fk_tool(q_test, [0.015, 0.0, 0.0])
fake_v2 = [pt2_m[0] * 1000.0, pt2_m[1] * 1000.0, pt2_m[2] * 1000.0, 0.0, 0.0, 0.0]
vr_virt = model.validate_against_sdk(
    fake_v2, q_test, tolerance_mm=3.0,
    tcp_offset_mm_deg=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    virtual_tip_offset_mm=virt_only,
)
check("validation: virtual tip only path", vr_virt.valid, vr_virt.detail)
vr_force = model.validate_against_sdk(
    fake_tcp, q_test, tolerance_mm=3.0, tcp_offset_mm_deg=tcp_mm, validate_with_tcp=False,
)
check("validation: forced flange rejects TCP pose", not vr_force.valid)

# Joint limits
limits = model.joint_limits()
check("joint_limits: 7 entries", len(limits) == 7)
for i, (lo, hi) in enumerate(limits):
    check(f"  J{i+1}: lo < hi", lo < hi, f"lo={lo} hi={hi}")


# =================================================================
# 6. rcm_controller
# =================================================================
section("rcm_controller")
from curac_teleop.teleopv4.rcm_controller import (
    TipLockController, TipLockConfig,
    EntryPointLockController, EntryPointLockConfig,
)

# TipLock
tl = TipLockController(TipLockConfig(max_angular_deg_s=10.0, correction_gain=3.0))
check("tip_lock: not active initially", not tl.active)

pose = [200.0, 100.0, 300.0, 180.0, 0.0, 0.0]
tl.capture(pose)
check("tip_lock: active after capture", tl.active)
check("tip_lock: locked pos", tl.locked_pos_mm == [200.0, 100.0, 300.0])

# No drift: zero correction
lin, ang = tl.compute_velocity(pose, 0.0, 0.0)
check("tip_lock: zero stick->zero ang", all(abs(a) < 1e-6 for a in ang))
check("tip_lock: no drift->zero lin", all(abs(v) < 1e-6 for v in lin))

# Stick input -> angular output
lin2, ang2 = tl.compute_velocity(pose, 1.0, 0.0)
check("tip_lock: stickX->wy", abs(ang2[1]) > 0)

# Drift correction
drifted = [201.0, 100.0, 300.0, 180.0, 0.0, 0.0]
lin3, ang3 = tl.compute_velocity(drifted, 0.0, 0.0)
check("tip_lock: drift correction vx<0", lin3[0] < -0.5,
      f"vx={lin3[0]}")

tl.clear()
check("tip_lock: clear deactivates", not tl.active)

# EntryPointLock
el = EntryPointLockController(EntryPointLockConfig())
pose_rcm = [200.0, 100.0, 300.0, 180.0, 0.0, 0.0]
el.capture_pivot(pose_rcm, 80.0)
check("entry_lock: active", el.active)

# tool_z at (180,0,0) is [0,0,-1], so pivot = tcp - 80*[0,0,-1] = tcp + [0,0,80]
check("entry_lock: pivot Z", abs(el.pivot_xyz[2] - 380.0) < 0.1,
      f"pz={el.pivot_xyz[2]}")

# Zero sticks, zero depth -> small correction only
vl, va = el.compute_rcm_velocity(pose_rcm, 0.0, 0.0, 0.0)
check("entry_lock: zero input->near zero", vec3_norm(vl) < 1.0,
      f"vl_norm={vec3_norm(vl)}")

# Stick input produces angular + orbital linear
vl2, va2 = el.compute_rcm_velocity(pose_rcm, 0.5, 0.0, 0.0)
check("entry_lock: stick produces motion", vec3_norm(vl2) > 0.1)

# Depth produces motion along tool axis
vl3, va3 = el.compute_rcm_velocity(pose_rcm, 0.0, 0.0, 10.0)
check("entry_lock: depth produces vz", abs(vl3[2]) > 1.0,
      f"vz={vl3[2]}")

el.clear()
check("entry_lock: clear deactivates", not el.active)


# =================================================================
# 7. input_layer (structure only, no joystick)
# =================================================================
section("input_layer (no hardware)")
from curac_teleop.teleopv4.input_layer import InputConfig, InputSnapshot

cfg = InputConfig()
check("InputConfig defaults", cfg.deadzone == 0.10)
check("InputConfig btn_deadman=5", cfg.btn_deadman_idx == 5)

snap = InputSnapshot()
check("InputSnapshot defaults", snap.lx == 0.0 and snap.connected)
check("InputSnapshot dpad", snap.dpad == (0, 0))


# =================================================================
# Summary
# =================================================================
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
if ERRORS:
    print("\nFailures:")
    for e in ERRORS:
        print(e)
    sys.exit(1)
else:
    print("\nAll offline tests PASSED.")
    sys.exit(0)
