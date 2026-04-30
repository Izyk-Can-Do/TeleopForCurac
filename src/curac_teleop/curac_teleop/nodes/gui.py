import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QGridLayout, QFrame, QPushButton,
                             QTabWidget, QMessageBox, QGroupBox, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QColor
import pyqtgraph as pg
import numpy as np
import subprocess
import datetime
import os
import signal

# --- STYLE SETTINGS ---
BACKGROUND_COLOR = '#1e1e1e'
TEXT_COLOR = '#ffffff'
PLOT_BG_COLOR = '#121212'

# Colors: pastel tones for Forces, warm tones for Torques
COLORS = {
    'Fx': '#ff79c6',
    'Fy': '#bd93f9',
    'Fz': '#8be9fd',
    'Mx': '#ffb86c',
    'My': '#f1fa8c',
    'Mz': '#ff5555'
}

# --- LABELS ---
LABELS = {
    'Fx': "Force X (Lateral)",
    'Fy': "Force Y (Anterior)",
    'Fz': "Force Z (Axial Pressure)",
    'Mx': "Torque X (Roll)",
    'My': "Torque Y (Pitch)",
    'Mz': "Torque Z (Yaw)"
}

# Display-only sign mapping for "human-intuitive" interpretation.
# Raw ROS topic data remains untouched; this only affects GUI numbers/plots.
INTUITIVE_DISPLAY_SIGN = {
    'Fx': -1.0, 'Fy': -1.0, 'Fz': -1.0,
    'Mx': -1.0, 'My': -1.0, 'Mz': -1.0,
}

GUIDE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "xbox_controller_guide_360.jpg")


# ============================================================
# 1) ROS 2 NODE
# ============================================================
class RosNode(Node):
    def __init__(self):
        super().__init__('gui_node')
        self.subscription = self.create_subscription(
            WrenchStamped, '/xarm/ft_data', self.listener_callback, 10)

        self.tare_client = self.create_client(Trigger, '/xarm/tare_sensor')

        self.current_data = {
            'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0,
            'Mx': 0.0, 'My': 0.0, 'Mz': 0.0,
            't': 0.0
        }

    def listener_callback(self, msg):
        self.current_data['Fx'] = msg.wrench.force.x
        self.current_data['Fy'] = msg.wrench.force.y
        self.current_data['Fz'] = msg.wrench.force.z
        self.current_data['Mx'] = msg.wrench.torque.x
        self.current_data['My'] = msg.wrench.torque.y
        self.current_data['Mz'] = msg.wrench.torque.z

        # Use the message timestamp from bridge.py
        self.current_data['t'] = (
            float(msg.header.stamp.sec) +
            float(msg.header.stamp.nanosec) * 1e-9
        )

    def send_tare_request(self):
        if not self.tare_client.wait_for_service(timeout_sec=1.0):
            return False
        req = Trigger.Request()
        self.tare_client.call_async(req)
        return True


# ============================================================
# 2) BACKGROUND WORKER THREAD
# ============================================================
class RosThread(QThread):
    """Spins ROS 2 in the background so the GUI never freezes."""
    new_data_signal = pyqtSignal(dict)

    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self._is_running = True

    def run(self):
        while rclpy.ok() and self._is_running:
            try:
                rclpy.spin_once(self.ros_node, timeout_sec=0.01)
                # Emit a copy so GUI thread gets a stable snapshot
                self.new_data_signal.emit(dict(self.ros_node.current_data))
            except Exception:
                pass

    def stop(self):
        self._is_running = False
        self.quit()
        self.wait()


# ============================================================
# 3) MAIN WINDOW GUI
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.display_intuitive = True
        self.recording_process = None
        self.recording_root = os.path.join(os.getcwd(), "recordings")
        os.makedirs(self.recording_root, exist_ok=True)
        self._log_file = None
        self.is_recording = False

        self.setWindowTitle("Neuro-Surgical Robotics Interface")
        self.resize(1400, 900)
        self.setMinimumSize(1260, 820)
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR};")

        # Main Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 10px; min-width: 150px; }
            QTabBar::tab:selected { background: #555; color: white; font-weight: bold; }
        """)
        self.setCentralWidget(self.tabs)

        # Tab 1: Monitor
        self.monitor_tab = QWidget()
        self.setup_monitor_ui()
        self.tabs.addTab(self.monitor_tab, "LIVE SURGERY MONITOR")

        # Tab 2: Xbox Guide
        self.guide_tab = QWidget()
        self.setup_guide_ui()
        self.tabs.addTab(self.guide_tab, "CONTROLLER GUIDE")

        # --- MULTITHREADING SETUP ---
        self.ros_thread = RosThread(self.ros_node)
        self.ros_thread.new_data_signal.connect(self.update_gui)
        self.ros_thread.start()

    def setup_monitor_ui(self):
        layout = QHBoxLayout(self.monitor_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # --- LEFT PANEL: NUMBERS & BUTTONS ---
        left_panel = QFrame()
        left_panel.setFixedWidth(420)
        left_panel.setStyleSheet("border-right: 2px solid #333; padding-right: 12px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)

        title = QLabel("REAL-TIME SENSOR")
        title.setFont(QFont("Arial", 17, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ecf0f1; margin-bottom: 10px;")
        left_layout.addWidget(title)

        self.value_labels = {}
        keys = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        units = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']

        for i, key in enumerate(keys):
            box = QFrame()
            box.setMinimumHeight(92)
            box.setStyleSheet(f"""
                background-color: #252526; 
                border-left: 5px solid {COLORS[key]}; 
                border-radius: 4px; 
                margin-bottom: 5px;
            """)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 6, 10, 6)
            box_layout.setSpacing(2)

            lbl_title = QLabel(LABELS[key])
            lbl_title.setWordWrap(True)
            lbl_title.setFont(QFont("Arial", 10, QFont.Bold))
            lbl_title.setStyleSheet("color: #bdc3c7; border: none;")

            val_layout = QHBoxLayout()
            val_layout.setContentsMargins(0, 0, 0, 0)
            val_layout.setSpacing(6)

            lbl_val = QLabel("0.00")
            font_size = 24
            lbl_val.setFont(QFont("Consolas", font_size, QFont.Bold))
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_val.setStyleSheet("color: white; border: none;")
            lbl_val.setMinimumWidth(160)

            lbl_unit = QLabel(units[i])
            lbl_unit.setFont(QFont("Arial", 13))
            lbl_unit.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
            lbl_unit.setStyleSheet("color: #7f8c8d; border: none; padding-bottom: 5px;")
            lbl_unit.setMinimumWidth(36)

            val_layout.addStretch()
            val_layout.addWidget(lbl_val)
            val_layout.addWidget(lbl_unit)

            self.value_labels[key] = lbl_val

            box_layout.addWidget(lbl_title)
            box_layout.addLayout(val_layout)
            left_layout.addWidget(box)

        # --- ACTIONS SECTION ---
        action_group = QGroupBox("SURGICAL CONTROLS")
        action_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555;
                margin-top: 10px;
                font-weight: bold;
                padding-top: 14px;
                border-radius: 4px;
            }
        """)
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(10, 10, 10, 10)
        action_layout.setSpacing(8)

        self.btn_tare = QPushButton("RESET SENSOR (TARE)")
        self.btn_tare.setFixedHeight(48)
        self.btn_tare.setStyleSheet("""
            QPushButton { background-color: #d35400; color: white; font-weight: bold; border-radius: 5px; font-size: 12px; padding: 4px 8px; }
            QPushButton:hover { background-color: #e67e22; }
        """)
        self.btn_tare.clicked.connect(self.handle_tare)
        action_layout.addWidget(self.btn_tare)

        self.btn_display_mode = QPushButton("DISPLAY: INTUITIVE (SIGN-CORRECTED)")
        self.btn_display_mode.setFixedHeight(44)
        self.btn_display_mode.setStyleSheet("""
            QPushButton { background-color: #2c3e50; color: white; font-weight: bold; border-radius: 5px; font-size: 11px; padding: 4px 8px; }
            QPushButton:hover { background-color: #34495e; }
        """)
        self.btn_display_mode.clicked.connect(self.toggle_display_mode)
        action_layout.addWidget(self.btn_display_mode)

        self.btn_record = QPushButton("START RECORDING")
        self.btn_record.setFixedHeight(48)
        self.btn_record.setStyleSheet("""
            QPushButton { background-color: #c0392b; color: white; font-weight: bold; border-radius: 5px; font-size: 12px; padding: 4px 8px; }
            QPushButton:hover { background-color: #e74c3c; }
        """)
        self.btn_record.clicked.connect(self.toggle_recording)
        action_layout.addWidget(self.btn_record)

        self.lbl_status = QLabel("System Status: READY")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #7f8c8d; font-size: 12px; margin-top: 5px;")
        action_layout.addWidget(self.lbl_status)

        left_layout.addWidget(action_group)
        left_layout.addStretch()
        layout.addWidget(left_panel)

        # --- RIGHT PANEL: PLOTS ---
        right_panel = QWidget()
        grid_layout = QGridLayout(right_panel)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setHorizontalSpacing(8)
        grid_layout.setVerticalSpacing(8)

        self.plots = {}
        self.curves = {}

        # Keep same structure, but now hold time + raw data
        self.time_window_sec = 5.0  # show only the last 5 seconds
        self.max_samples = 2000  # enough for high-rate raw data

        self.data_buffers = {key: [] for key in keys}
        self.time_buffers = {key: [] for key in keys}

        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for i, key in enumerate(keys):
            full_title = f"{LABELS[key]} [{units[i]}]"
            plot = pg.PlotWidget(title=full_title)
            plot.setBackground(PLOT_BG_COLOR)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setLabel('bottom', 'Time', units='s')
            plot.setLabel('left', units[i])
            plot.getPlotItem().titleLabel.setText(full_title, color="#eaeaea", size="10pt")

            # Force ranges requested by user.
            if key in ['Fx', 'Fy', 'Fz']:
                plot.setYRange(-50, 50)
            else:
                plot.setYRange(-2, 2)

            pen = pg.mkPen(color=COLORS[key], width=2)
            self.curves[key] = plot.plot(self.time_buffers[key], self.data_buffers[key], pen=pen)
            self.plots[key] = plot

            row, col = positions[i]
            grid_layout.addWidget(plot, row, col)

        layout.addWidget(right_panel)

    def setup_guide_ui(self):
        layout = QVBoxLayout(self.guide_tab)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        self.load_guide_image()

    def load_guide_image(self):
        if os.path.exists(GUIDE_IMAGE_PATH):
            pixmap = QPixmap(GUIDE_IMAGE_PATH)
            w = self.guide_tab.width() + 200
            h = self.guide_tab.height() + 200
            scaled_pixmap = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText(
                f"IMAGE NOT FOUND!\nPlease save the image as:\n'{GUIDE_IMAGE_PATH}'\nin the same folder as this script.")
            self.image_label.setFont(QFont("Arial", 14, QFont.Bold))
            self.image_label.setStyleSheet("color: #e74c3c;")

    def resizeEvent(self, event):
        if self.tabs.currentWidget() == self.guide_tab:
            self.load_guide_image()
        super().resizeEvent(event)

    def handle_tare(self):
        success = self.ros_node.send_tare_request()
        if success:
            self.lbl_status.setText("Status: SENSOR RESET (TARE) OK")
            self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            QTimer.singleShot(3000, lambda: self.lbl_status.setText("System Status: READY"))
        else:
            self.lbl_status.setText("Status: TARE FAILED!")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")

    def toggle_recording(self):
        if not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            session_name = f"surgery_{timestamp}"

            session_dir = os.path.join(self.recording_root, session_name)
            bag_dir = os.path.join(session_dir, "bag")
            os.makedirs(session_dir, exist_ok=True)

            log_path = os.path.join(session_dir, "rosbag.log")
            self._log_file = open(log_path, "w", buffering=1)

            cmd = [
                "ros2", "bag", "record",
                "-o", bag_dir,
                "/xarm/ft_data",
                "/joint_states",
                "/tf",
                "/tf_static",
                "/zed/zed_node/rgb/color/rect/image",
                "/zed/zed_node/rgb/color/rect/camera_info"
            ]

            try:
                self.recording_process = subprocess.Popen(
                    cmd,
                    preexec_fn=os.setsid,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT
                )
                self.is_recording = True
                self._current_session_dir = session_dir

                self.btn_record.setText("STOP RECORDING")
                self.btn_record.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
                self.lbl_status.setText(f"Status: RECORDING... Session: {session_name}")
                self.lbl_status.setStyleSheet("color: #f1c40f; font-weight: bold;")

            except Exception as e:
                self.lbl_status.setText(f"Status: REC FAILED: {e}")
                self.lbl_status.setStyleSheet("color: red;")
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
        else:
            if self.recording_process:
                try:
                    os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
                    self.recording_process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if self.recording_process:
                        self.recording_process.kill()
                self.recording_process = None

                if self._log_file:
                    try:
                        self._log_file.close()
                    except Exception:
                        pass
                    self._log_file = None

            self.is_recording = False
            self.btn_record.setText("START RECORDING")
            self.btn_record.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
            saved_path = getattr(self, "_current_session_dir", "recordings/")
            self.lbl_status.setText(f"Status: DATA SAVED.  ({saved_path})")
            self.lbl_status.setStyleSheet("color: #ecf0f1;")

    def _display_value(self, key: str, raw_value: float) -> float:
        if self.display_intuitive:
            return float(INTUITIVE_DISPLAY_SIGN.get(key, 1.0)) * float(raw_value)
        return float(raw_value)

    def toggle_display_mode(self):
        self.display_intuitive = not self.display_intuitive
        if self.display_intuitive:
            self.btn_display_mode.setText("DISPLAY: INTUITIVE (SIGN-CORRECTED)")
            self.lbl_status.setText("Status: DISPLAY = INTUITIVE (raw stream unchanged)")
        else:
            self.btn_display_mode.setText("DISPLAY: RAW SENSOR FRAME")
            self.lbl_status.setText("Status: DISPLAY = RAW SENSOR FRAME")
        self.lbl_status.setStyleSheet("color: #ecf0f1; font-weight: bold;")

    def update_gui(self, data):
        """Called safely by the background thread whenever new data is ready."""
        keys = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        t = float(data.get('t', 0.0))

        for key in keys:
            raw_val = float(data[key])
            val = self._display_value(key, raw_val)
            self.value_labels[key].setText(f"{val:.2f}")

            # Append newest raw sample
            self.time_buffers[key].append(t)
            self.data_buffers[key].append(val)

            # Prevent unlimited growth
            if len(self.time_buffers[key]) > self.max_samples:
                self.time_buffers[key] = self.time_buffers[key][-self.max_samples:]
                self.data_buffers[key] = self.data_buffers[key][-self.max_samples:]

            # Keep only the last self.time_window_sec seconds
            t_min = t - self.time_window_sec
            while self.time_buffers[key] and self.time_buffers[key][0] < t_min:
                self.time_buffers[key].pop(0)
                self.data_buffers[key].pop(0)

            # Plot relative time so visible x-axis starts from 0
            if self.time_buffers[key]:
                t0 = self.time_buffers[key][0]
                x = [tt - t0 for tt in self.time_buffers[key]]
                self.curves[key].setData(x, self.data_buffers[key])
                self.plots[key].setXRange(0, self.time_window_sec)

    def closeEvent(self, event):
        """Safely shuts down the background thread when the user closes the window."""
        self.ros_thread.stop()
        super().closeEvent(event)


def main():
    rclpy.init()
    ros_node = RosNode()
    app = QApplication(sys.argv)
    window = MainWindow(ros_node)
    window.show()

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        if window.recording_process:
            try:
                os.killpg(os.getpgid(window.recording_process.pid), signal.SIGINT)
                window.recording_process.wait(timeout=2)
            except Exception:
                pass
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()