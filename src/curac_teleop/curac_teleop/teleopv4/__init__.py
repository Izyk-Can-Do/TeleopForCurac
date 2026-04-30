"""
teleopv4 - Modular teleoperation controller for xArm7.

Research-prototype with explicit state machine, PyKDL-based kinematics,
constrained motion modes (Tip-Lock, Entry-Point-Lock / RCM), and
safety-first design.  Uses xArm SDK for robot connection and low-level
execution.

STATUS: research prototype - NOT validated for clinical or safety-critical use.
"""

__version__ = "0.1.0"
