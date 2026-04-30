from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    dry_run = LaunchConfiguration("dry_run")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="192.168.1.243"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            Node(
                package="curac_teleop",
                executable="bridge_node",
                name="ft_bridge",
                output="screen",
            ),
            Node(
                package="curac_teleop",
                executable="rviz_bridge_node",
                name="rviz_bridge",
                output="screen",
            ),
            Node(
                package="curac_teleop",
                executable="teleopv7_node",
                name="teleopv7_node",
                output="screen",
                parameters=[{"robot_ip": robot_ip, "dry_run": dry_run}],
            ),
            Node(
                package="curac_teleop",
                executable="gui_node",
                name="gui_node",
                output="screen",
            ),
        ]
    )
