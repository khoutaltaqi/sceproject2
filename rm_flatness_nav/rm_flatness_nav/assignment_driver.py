import json
import math
import threading
import time
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from robomaster import robot
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Float64MultiArray, String

from rm_flatness_nav.floor_calibration import tag_center


class AssignmentDriver(Node):
    def __init__(self):
        super().__init__('assignment_driver')

        self.declare_parameter('conn_type', 'ap')
        self.declare_parameter('enable_motion', False)
        self.declare_parameter('tof_index', 0)

        path = (
            Path.home()
            / 'robomaster_ws/src/rm_flatness_nav'
            / 'config/floor_map.json'
        )

        calibration = json.loads(path.read_text())
        self.H = np.array(calibration['H'], dtype=float)
        self.image_size = calibration['image_size']

        if (
            self.H.shape != (3, 3)
            or not np.isfinite(self.H).all()
        ):
            raise ValueError('floor_map.json tidak valid')

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        self.detector_params = (
            cv2.aruco.DetectorParameters_create()
        )

        self.bridge = CvBridge()

        self.tag_pub = self.create_publisher(
            PoseStamped, '/rm/tag', 10
        )
        self.tof_pub = self.create_publisher(
            Range, '/rm/tof', 10
        )
        self.image_pub = self.create_publisher(
            Image, '/rm/image', 1
        )
        self.status_pub = self.create_publisher(
            String, '/rm/status', 10
        )
        self.data_pub = self.create_publisher(
            Float64MultiArray, '/rm/data', 10
        )

        self.create_subscription(
            Twist, '/rm/cmd_vel', self.on_command, 10
        )

        self.ep = robot.Robot()

        self.running = False
        self.video = False
        self.thread = None

        self.frame = None
        self.frame_number = 0
        self.processed_number = -1

        self.cmd = (0.0, 0.0, 0.0)
        self.tof = (float('nan'), 0.0)
        self.velocity = (0.0, 0.0)
        self.turn_rate = (0.0, 0.0)
        self.yaw = None

        self.halt = ''
        self.last_status = ''
        self.stall_since = None

        self.previous_wheels = np.zeros(4)
        self.previous_time = time.monotonic()

        self.create_timer(0.10, self.process_image)
        self.create_timer(0.05, self.control)

    def start(self):
        self.ep.initialize(
            conn_type=self.get_parameter('conn_type').value
        )
        self.ep.set_robot_mode(mode=robot.FREE)

        action = self.ep.robotic_arm.moveto(x=105, y=60)

        if not action.wait_for_completed(timeout=8):
            raise RuntimeError(
                'Arm belum mencapai posisi tugas'
            )

        if not self.ep.sensor.sub_distance(
            freq=10, callback=self.on_tof
        ):
            raise RuntimeError('Langganan ToF gagal')

        if not self.ep.chassis.sub_velocity(
            freq=10, callback=self.on_velocity
        ):
            raise RuntimeError(
                'Langganan kecepatan gagal'
            )

        if not self.ep.chassis.sub_attitude(
            freq=10, callback=self.on_attitude
        ):
            raise RuntimeError(
                'Langganan orientasi gagal'
            )

        self.video = True

        if self.ep.camera.start_video_stream(
            display=False, resolution='720p'
        ) is False:
            raise RuntimeError('Kamera gagal dimulai')

        self.running = True
        self.thread = threading.Thread(
            target=self.grab,
            daemon=True,
        )
        self.thread.start()

    def on_command(self, msg):
        v = float(np.clip(msg.linear.x, 0.0, 0.16))
        w = float(np.clip(msg.angular.z, -3.0, 3.0))

        if math.isfinite(v) and math.isfinite(w):
            self.cmd = (v, w, time.monotonic())

    def on_tof(self, values):
        index = int(
            self.get_parameter('tof_index').value
        )

        if not 0 <= index < len(values):
            return

        mm = float(values[index])

        # Abaikan kanal kosong dan nilai penanda.
        if not 0 < mm < 10000:
            return

        self.tof = (
            mm / 1000.0,
            time.monotonic(),
        )

        msg = Range()
        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        msg.header.frame_id = 'front_tof'
        msg.radiation_type = Range.INFRARED
        msg.min_range = 0.0
        msg.max_range = 10.0
        msg.range = mm / 1000.0

        self.tof_pub.publish(msg)

        if mm <= 200:
            self.halt = 'BRAKE_TOF'

    def on_velocity(self, values):
        # SDK: vg_x, vg_y, vg_z, vb_x, vb_y, vb_z.
        self.velocity = (
            float(values[3]),
            time.monotonic(),
        )

    def on_attitude(self, values):
        now = time.monotonic()
        yaw = float(values[0])

        if self.yaw is not None:
            old_yaw, old_time = self.yaw

            delta = (
                yaw - old_yaw + 180.0
            ) % 360.0 - 180.0

            self.turn_rate = (
                -math.radians(delta)
                / max(now - old_time, 0.001),
                now,
            )

        self.yaw = (yaw, now)

    def grab(self):
        while self.running:
            try:
                frame = self.ep.camera.read_cv2_image(
                    timeout=0.5,
                    strategy='newest',
                )

                if frame is not None:
                    self.frame_number += 1

                    self.frame = (
                        frame,
                        time.monotonic(),
                        self.get_clock().now().to_msg(),
                        self.frame_number,
                    )

            except Empty:
                pass

            except Exception as exc:
                self.halt = f'FAULT_CAMERA: {exc}'
                return

            time.sleep(0.010)

    def process_image(self):
        item = self.frame

        if (
            item is None
            or item[3] == self.processed_number
        ):
            return

        frame, received, stamp, number = item
        self.processed_number = number

        if time.monotonic() - received > 0.30:
            return

        frame = frame.copy()

        if [
            frame.shape[1], frame.shape[0]
        ] != self.image_size:
            self.halt = 'FAULT_CALIBRATION_RESOLUTION'
            return

        center = tag_center(
            frame,
            self.dictionary,
            self.detector_params,
        )

        text = 'Tag 3: NOT FOUND'

        if center is not None:
            xy = cv2.perspectiveTransform(
                center.reshape(
                    1, 1, 2
                ).astype(np.float32),
                self.H,
            )[0, 0]

            x, y = map(float, xy)

            if (
                math.isfinite(x)
                and math.isfinite(y)
                and x > 0
            ):
                distance = math.hypot(x, y)

                msg = PoseStamped()
                msg.header.stamp = stamp
                msg.header.frame_id = 'front_reference'
                msg.pose.position.x = x
                msg.pose.position.y = y
                msg.pose.orientation.w = 1.0

                self.tag_pub.publish(msg)

                bearing = math.degrees(
                    math.atan2(y, x)
                )

                text = (
                    f'Tag 3: d={distance:.3f} m '
                    f'bearing={bearing:+.1f} deg'
                )

                if distance <= 0.20:
                    self.halt = 'BRAKE_CAMERA'

        cv2.putText(
            frame,
            text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        image = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8',
        )
        image.header.stamp = stamp
        image.header.frame_id = 'camera_optical_frame'

        self.image_pub.publish(image)

    def control(self):
        now = time.monotonic()
        dt = max(now - self.previous_time, 0.001)
        self.previous_time = now

        v, w, command_time = self.cmd

        ready = (
            self.frame is not None
            and now - self.frame[1] < 0.75
            and now - self.tof[1] < 0.75
            and now - self.velocity[1] < 0.75
            and now - self.turn_rate[1] < 0.75
        )

        enabled = self.get_parameter(
            'enable_motion'
        ).value

        if (
            not ready
            or not enabled
            or now - command_time > 0.25
            or self.halt
        ):
            v, w = 0.0, 0.0

        motion_requested = (
            v > 0.04 or abs(w) > 0.10
        )
        moving = (
            abs(self.velocity[0]) > 0.01
            or abs(self.turn_rate[0]) > 0.03
        )

        if motion_requested and not moving:
            if self.stall_since is None:
                self.stall_since = now

            if now - self.stall_since > 2.0:
                self.halt = 'FAULT_NO_MOTION'
                v, w = 0.0, 0.0
        else:
            self.stall_since = None

        try:
            if v == 0.0 and w == 0.0:
                stopped = self.ep.chassis.drive_wheels(
                    w1=0,
                    w2=0,
                    w3=0,
                    w4=0,
                    timeout=0.3,
                )

                if stopped is False:
                    self.halt = 'FAULT_STOP_COMMAND'

            else:
                # ROS: putar kiri positif.
                # SDK: putar kanan positif, satuan derajat/s.
                #
                # Sebagian versi SDK mengembalikan False
                # untuk jalur drive_speed tanpa ACK.
                # Gerakan diverifikasi dari feedback.
                self.ep.chassis.drive_speed(
                    x=v,
                    y=0.0,
                    z=-math.degrees(w),
                    timeout=0.3,
                )

        except Exception as exc:
            self.halt = f'FAULT_COMMAND: {exc}'

            self.ep.chassis.drive_wheels(
                w1=0,
                w2=0,
                w3=0,
                w4=0,
                timeout=0.3,
            )

            v, w = 0.0, 0.0

        # Persamaan kinematika dari PDF.
        wheels = np.array([
            v + 0.20*w,
            v - 0.20*w,
            v - 0.20*w,
            v + 0.20*w,
        ]) / 0.05

        # Estimasi tegangan dari perintah kecepatan.
        volts = (
            0.38*wheels
            + 0.06*(
                wheels - self.previous_wheels
            ) / dt
        )

        self.previous_wheels = wheels

        data = Float64MultiArray()
        data.data = [
            v,
            w,
            self.velocity[0],
            self.turn_rate[0],
            *volts.tolist(),
        ]

        self.data_pub.publish(data)

        status = self.halt or (
            'DISABLED'
            if not enabled
            else 'READY'
            if ready
            else 'WAIT_SENSORS'
        )

        self.status_pub.publish(
            String(data=status)
        )

        if status != self.last_status:
            self.get_logger().info(status)
            self.last_status = status

    def close(self):
        self.running = False

        try:
            self.ep.chassis.drive_wheels(
                w1=0,
                w2=0,
                w3=0,
                w4=0,
                timeout=0.3,
            )
        except Exception:
            pass

        if self.thread is not None:
            self.thread.join(timeout=1)

        for action in [
            self.ep.sensor.unsub_distance,
            self.ep.chassis.unsub_velocity,
            self.ep.chassis.unsub_attitude,
        ]:
            try:
                action()
            except Exception:
                pass

        if self.video:
            try:
                self.ep.camera.stop_video_stream()
            except Exception:
                pass

        self.ep.close()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = AssignmentDriver()
        node.start()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
