from setuptools import find_packages, setup
import os
from glob import glob

package_name = "curac_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, package_name + ".*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "assets"), glob("curac_teleop/*.jpg")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "pygame",
        "dualsense-controller",
        "xarm-python-sdk",
    ],
    zip_safe=True,
    maintainer="NeuroMill / CURAC",
    maintainer_email="igabilayeff@gmail.com",
    description="Standalone CURAC teleoperation package for xArm7 with TeleopV7 and support nodes.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "teleopv4_node = curac_teleop.teleopv4.node:main",
            "teleopv7_node = curac_teleop.teleopv7.node:main",
            "bridge_node = curac_teleop.nodes.bridge:main",
            "rviz_bridge_node = curac_teleop.nodes.rviz_bridge:main",
            "gui_node = curac_teleop.nodes.gui:main",
        ],
    },
)
