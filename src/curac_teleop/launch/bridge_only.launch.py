from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
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
        ]
    )
