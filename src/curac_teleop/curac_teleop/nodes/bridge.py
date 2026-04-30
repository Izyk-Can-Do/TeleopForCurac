#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger
from xarm.wrapper import XArmAPI
import time
import sys

# --- CONFIGURATION ---
ROBOT_IP = '192.168.1.243'
BAUDRATE = 2000000

class FtSensorBridge(Node):
    def __init__(self):
        super().__init__('ft_sensor_bridge')

        # Publisher (Message type, topic name, queue depth)
        self.publisher_ = self.create_publisher(WrenchStamped, '/xarm/ft_data', 10)

        # Service for Zeroing (Tare) if I run ros2 service call /xarm/tare_sensor std_srvs/srv/Trigger {} it is gonna call tare_callback
        self.srv = self.create_service(Trigger, '/xarm/tare_sensor', self.tare_callback)

        self.get_logger().info(f'Connecting to robot at {ROBOT_IP}...')
        self.arm = XArmAPI(ROBOT_IP)

        # --- ROBUST INITIALIZATION ---
        self.initialize_sensor()

        # Timer 5Hz (0.02 for 50Hz)
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.get_logger().info('Bridge Started. Publishing data...')

    def initialize_sensor(self):
        """Fixes baudrate and resets sensor."""
        self.get_logger().info('Initializing sensor communication...')
        self.arm.clean_error()
        time.sleep(0.5)

        # Force Baudrate
        self.arm.set_tgpio_modbus_baudrate(BAUDRATE)
        time.sleep(0.5)

        # Reset Sensor Power
        self.arm.ft_sensor_enable(0)
        time.sleep(0.5)
        self.arm.ft_sensor_enable(1)
        time.sleep(1.0) # Wait for boot

        # Initial Tare
        self.arm.ft_sensor_set_zero()
        self.get_logger().info('Sensor Initialized and Zeroed.')

    def tare_callback(self, request, response):
        """ROS2 Service callback to zero the sensor."""
        self.get_logger().info('Received Tare Request...')
        self.arm.ft_sensor_set_zero()
        response.success = True
        response.message = "Sensor Zeroed (Tare) Successfully"
        return response

    def timer_callback(self):
        code, ft_data = self.arm.get_ft_sensor_data()

        if code == 0 and ft_data:
            msg = WrenchStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "ft_sensor_link"

            # Data Mapping
            msg.wrench.force.x = float(ft_data[0])
            msg.wrench.force.y = float(ft_data[1])
            msg.wrench.force.z = float(ft_data[2])
            msg.wrench.torque.x = float(ft_data[3])
            msg.wrench.torque.y = float(ft_data[4])
            msg.wrench.torque.z = float(ft_data[5])

            self.publisher_.publish(msg)

    def shutdown(self):
        self.arm.disconnect()

def main(args=None):
    rclpy.init(args=args)
    node = FtSensorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
