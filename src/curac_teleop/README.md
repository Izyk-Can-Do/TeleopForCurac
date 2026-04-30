# CURAC Teleop (Standalone ROS2 Project)

CURAC Teleop is a standalone ROS2 package for xArm7 teleoperation with PlayStation 5 DualSense control, force-aware behavior, and practical clinical workflow support.

This project was split into its own workspace so you can develop, test, and publish CURAC teleoperation independently from older versions.

---

## What This Project Includes

- `teleopv7_node`: main teleoperation controller (PS5-first workflow).
- `bridge_node`: publishes force/torque sensor data from xArm SDK to ROS topic.
- `rviz_bridge_node`: publishes current joint states for RViz and other consumers.
- `gui_node`: real-time FT monitor and recording helper.
- Shared control modules copied from `teleopv4` (state machine, safety, FT guard, kinematics, math, constrained control helpers).

---

## Project Layout

```text
CURACTELEOP/
└── src/
    └── curac_teleop/
        ├── package.xml
        ├── setup.py
        ├── README.md
        ├── launch/
        │   ├── teleopv7.launch.py
        │   ├── teleopv7_with_gui.launch.py
        │   └── bridge_only.launch.py
        ├── config/
        │   ├── teleopv7_default.yaml
        │   ├── teleopv7_safe.yaml
        │   └── teleopv7_drilling.yaml
        └── curac_teleop/
            ├── teleopv7/
            ├── teleopv4/
            ├── teleopv5/
            └── nodes/
```

---

## System Requirements

- Ubuntu Linux
- ROS2 Humble (recommended)
- Python 3.10+
- xArm reachable by network
- PS5 DualSense for controller + haptics

### Python Dependencies

Declared as install requirements in `setup.py`:
- `numpy`
- `scipy`
- `pygame`
- `dualsense-controller`
- `xarm-python-sdk`

### System Dependencies

Install these with apt:
- `python3-pykdl` (needed for kinematics module)

Optional, only for GUI:
- `python3-pyqt5`
- `pyqtgraph` (pip or apt)

---

## One-Time Setup

### 1) Go to workspace

```bash
cd /home/islam/Desktop/CURACTELEOP
```

### 2) Source ROS2

```bash
source /opt/ros/humble/setup.bash
```

### 3) Build

```bash
colcon build --symlink-install
```

### 4) Source local install

```bash
source install/setup.bash
```

### 5) Install system package for KDL (important)

```bash
sudo apt update
sudo apt install -y python3-pykdl
```

---

## Launch Modes

## 1) Main Teleop only (recommended first test)

```bash
ros2 launch curac_teleop teleopv7.launch.py
```

This starts only `teleopv7_node` with default config file.

## 2) Full stack (teleop + FT bridge + RViz bridge + GUI)

```bash
ros2 launch curac_teleop teleopv7_with_gui.launch.py
```

## 3) Bridge-only mode (for diagnostics, RViz, or external tools)

```bash
ros2 launch curac_teleop bridge_only.launch.py
```

---

## Direct Node Runs (Without Launch)

### TeleopV7 with default profile

```bash
ros2 run curac_teleop teleopv7_node --ros-args --params-file /home/islam/Desktop/CURACTELEOP/src/curac_teleop/config/teleopv7_default.yaml
```

### TeleopV7 with drilling profile

```bash
ros2 run curac_teleop teleopv7_node --ros-args --params-file /home/islam/Desktop/CURACTELEOP/src/curac_teleop/config/teleopv7_drilling.yaml
```

---

## Controller Mapping (TeleopV7)

- `R1`: deadman switch; motion commands are ignored if not held.
- `Cross`: fixed-tip mode state action (capture/toggle behavior).
- `Square`: FT tare action.
- `L2` / `R2`: insertion/extraction depth control (project mapping in active code).
- Left stick: planar teleoperation in operator-perspective mapping.

If behavior feels opposite to your setup, do not hardcode immediately; prefer parameter tuning and profile updates first.

---

## Safety Behavior (Current Defaults)

The default profile is tuned for safe, practical operation:

- Global FT guard enabled.
- Soft slowdown near FT threshold enabled.
- Retreat-on-limit enabled.
  - Meaning: push direction is blocked at limit.
  - Pull-back direction remains allowed, so execution does not freeze.
- Motion rumble disabled by default to avoid noisy/annoying feedback.
- Force-based trigger feedback enabled and smoothed.

---

## Parameter Profiles

Three prepared config files:

- `config/teleopv7_default.yaml`
  - Balanced baseline.
- `config/teleopv7_safe.yaml`
  - Conservative speeds/limits for validation and first-use.
- `config/teleopv7_drilling.yaml`
  - More responsive, intended for contact-heavy workflow.

Use profile by passing `--params-file` to `ros2 run`, or pass custom config path to launch:

```bash
ros2 launch curac_teleop teleopv7.launch.py config:=/absolute/path/to/custom.yaml
```

---

## DualSense Permissions (Linux udev)

If controller input works but haptics/adaptive triggers do not, it is usually a hidraw write-permission problem.

### 1) Create rule

File: `/etc/udev/rules.d/99-dualsense.rules`

```text
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="054c", ATTRS{idProduct}=="0ce6", MODE="0666"
```

### 2) Reload rules

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 3) Reconnect controller and restart teleop

On startup, check log line similar to:
- `write_access=yes`

If it says `no`, haptics will be limited or disabled.

---

## Common Problems and Fixes

## Problem: `DistributionNotFound` at launch

Cause: package metadata requiring modules not installed in your environment.

Fix:
- Rebuild after dependency update:
```bash
cd /home/islam/Desktop/CURACTELEOP
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Problem: `PyKDL` import error

Fix:
```bash
sudo apt install -y python3-pykdl
```

## Problem: Controller connected but no adaptive trigger feedback

Checks:
- udev rule exists and loaded.
- device has write access.
- startup logs show DualSense bound device and write status.
- `v7_ft_haptic_enable` is true.

## Problem: Robot blocks completely on force limit

Enable retreat mode:
- `v7_ft_allow_retreat_on_limit:=true`

Adjust:
- `v7_ft_retreat_speed_scale`
- `v7_ft_retreat_min_cos`

---

## Bridge Node Notes

`bridge.py` and `rviz_bridge.py` currently use internal robot IP constants.  
If your robot IP changes, edit those files or add parameterization in a future revision.

---

## Recommended Daily Workflow

1. Start in `teleopv7_safe.yaml`.
2. Verify deadman, stop, and tare behaviors.
3. Verify FT feedback and trigger feel.
4. Move to `teleopv7_default.yaml`.
5. Use `teleopv7_drilling.yaml` only after safety checks pass.
6. Save tuned values into your own profile YAML and keep it versioned.

---

## Development and Contribution

If you update behavior:
- keep safety defaults conservative,
- document new parameters in this README,
- add launch/config examples for reproducibility.

---

## Final Notes

- This project is intentionally separated from older teleop versions for cleaner CURAC iteration.
- The goal is practical usability: clear startup, predictable controls, force-aware safety, and understandable logs.
- If you want, the next README revision can include screenshots, a quick reference card, and a bilingual section.
