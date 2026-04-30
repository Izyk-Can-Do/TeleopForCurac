from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    dry_run = LaunchConfiguration("dry_run")
    config_file = LaunchConfiguration("config")
    default_config = PathJoinSubstitution(
        [FindPackageShare("curac_teleop"), "config", "teleopv7_default.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="192.168.1.243"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="curac_teleop",
                executable="teleopv7_node",
                name="teleopv7_node",
                output="screen",
                parameters=[config_file, {"robot_ip": robot_ip, "dry_run": dry_run}],
            ),
        ]
    )
