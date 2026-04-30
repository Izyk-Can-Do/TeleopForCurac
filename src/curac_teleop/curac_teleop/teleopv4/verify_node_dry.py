#!/usr/bin/env python3
"""One-shot dry-run exercise of TeleopV4Node (no robot, no joystick timer).

Run after sourcing the workspace::

    python3 -m curac_teleop.teleopv4.verify_node_dry

Exits 0 on success, 1 on failure.  Does not register as a pytest module.
"""

import sys

import rclpy


def main() -> int:
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init(
        args=[
            'verify_node_dry',
            '--ros-args',
            '-p', 'dry_run:=true',
            '-p', 'enable_control_timer:=false',
        ],
    )
    node = None
    try:
        from curac_teleop.teleopv4.node import TeleopV4Node, _FakeArm

        node = TeleopV4Node()
        if not node.dry_run:
            print('FAIL: expected dry_run=True', file=sys.stderr)
            return 1
        if not isinstance(node.arm, _FakeArm):
            print('FAIL: expected _FakeArm', file=sys.stderr)
            return 1
        if node.timer is not None:
            print('FAIL: expected timer disabled', file=sys.stderr)
            return 1
        node._control_loop_inner()
        print('OK: TeleopV4Node dry-run construct + one control tick')
        return 0
    except Exception as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    finally:
        if node is not None:
            try:
                node.shutdown_hook()
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
