import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Float64MultiArray, String


def solve_flat_polynomial(z0, zdot0, zf, zdotf, T):
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, T, T*T, T*T*T],
        [0, 1, 2*T, 3*T*T],
    ], dtype=float)
    
    print (np.linalg.solve(
        matrix,
        [z0, zdot0, zf, zdotf],
    ))
    return np.linalg.solve(
        matrix,
        [z0, zdot0, zf, zdotf],
    )


def flat_velocity(ax, ay, t, T):
    if t >= T:
        return 0.0, 0.0

    dx = ax[1] + 2*ax[2]*t + 3*ax[3]*t*t
    dy = ay[1] + 2*ay[2]*t + 3*ay[3]*t*t

    ddx = 2*ax[2] + 6*ax[3]*t
    ddy = 2*ay[2] + 6*ay[3]*t

    square = dx*dx + dy*dy

    v = math.sqrt(square)

    w = (
        (dx*ddy - dy*ddx)/square
        if square > 1e-10
        else 0.0
    )

    return v, w


class FlatnessPlanner(Node):
    def __init__(self):
        super().__init__('flatness_planner')

        self.pub = self.create_publisher(
            Twist, '/rm/cmd_vel', 10
        )

        self.create_subscription(
            PoseStamped, '/rm/tag', self.on_tag, 10
        )
        self.create_subscription(
            Range, '/rm/tof', self.on_tof, 10
        )
        self.create_subscription(
            String, '/rm/status', self.on_status, 10
        )
        self.create_subscription(
            Float64MultiArray,
            '/rm/data',
            self.on_data,
            10,
        )

        self.tag = (float('nan'), 0.0, 0.0)
        self.tof = (float('nan'), 0.0)
        self.status = ('WAIT_DRIVER', 0.0)

        self.data = [0.0]*8
        self.data_time = 0.0

        self.state = 'STARTUP'
        self.direction = 1.0
        self.plan = None
        self.last_v = 0.0
        self.terminal_since = None
        self.finished_since = None
        self.done = False

        self.started = time.monotonic()
        self.previous_time = self.started
        self.next_log = 0.0

        directory = (
            Path.home()
            / 'robomaster_ws/src/rm_flatness_nav'
            / 'logs/assessment2'
            / datetime.now().strftime(
                'run_%Y%m%d_%H%M%S'
            )
        )

        directory.mkdir(
            parents=True,
            exist_ok=False,
        )
        self.directory = directory

        self.table_file = (
            directory / 'motor_voltage_log.csv'
        ).open('w', newline='')

        self.trace_file = (
            directory / 'execution.csv'
        ).open('w', newline='')

        self.table = csv.writer(self.table_file)
        self.trace = csv.writer(self.trace_file)

        header = [
            't_s',
            'V1_V',
            'V2_V',
            'V3_V',
            'V4_V',
            'state',
            'v_cmd_mps',
            'omega_cmd_radps',
            'v_measured_mps',
            'omega_measured_radps',
            'camera_m',
            'tof_m',
        ]

        self.table.writerow(header)
        self.trace.writerow(header)

        self.get_logger().info(
            f'OUTPUT: {directory}'
        )

        self.create_timer(0.05, self.step)

    def message_time(self, header):
        stamp = (
            header.stamp.sec
            + header.stamp.nanosec/1e9
        )

        age = max(
            0.0,
            self.get_clock().now().nanoseconds/1e9
            - stamp,
        )

        return time.monotonic() - age

    def on_tag(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y

        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or x <= 0
        ):
            return

        bearing = math.atan2(y, x)

        self.tag = (
            math.hypot(x, y),
            bearing,
            self.message_time(msg.header),
        )

        if abs(bearing) > 0.01:
            self.direction = (
                1.0 if bearing > 0 else -1.0
            )

    def on_tof(self, msg):
        if math.isfinite(msg.range) and msg.range > 0:
            self.tof = (
                msg.range,
                self.message_time(msg.header),
            )

    def on_status(self, msg):
        self.status = (
            msg.data,
            time.monotonic(),
        )

    def on_data(self, msg):
        if len(msg.data) == 8:
            self.data = list(msg.data)
            self.data_time = time.monotonic()

    def finish(self, state, now):
        if self.finished_since is None:
            self.finished_since = now
            self.state = state
            self.get_logger().info(state)

        # Tetap kirim nol dan catat telemetry
        # selama tiga detik setelah pengereman.
        if now - self.finished_since >= 3.0:
            self.done = True

        return 0.0, 0.0

    def decide(self, now, dt):
        d, bearing, seen = self.tag
        tof, tof_time = self.tof
        driver, heartbeat = self.status

        if self.finished_since is not None:
            return self.finish(self.state, now)

        if (
            now - seen <= 0.45 and d <= 0.20
        ) or driver == 'BRAKE_CAMERA':
            return self.finish('ARRIVED', now)

        if (
            now - tof_time < 0.75 and tof <= 0.20
        ) or driver == 'BRAKE_TOF':
            return self.finish('HALTED_TOF', now)

        if driver.startswith('FAULT'):
            return self.finish(driver, now)

        if now - self.started > 90:
            return self.finish(
                'MISSION_TIMEOUT', now
            )

        if (
            now - heartbeat > 0.75
            or driver != 'READY'
        ):
            self.state = 'WAIT_DRIVER'
            self.plan = None
            self.last_v = 0.0
            return 0.0, 0.0

        if now - seen > 0.45:
            self.state = 'SEARCHING'
            self.plan = None
            self.last_v = 0.0
            self.terminal_since = None

            return 0.0, self.direction * 0.50

        if abs(bearing) > 0.18:
            self.state = 'ALIGNING'
            self.plan = None
            self.last_v = 0.0
            self.terminal_since = None

            return 0.0, float(
                np.clip(
                    2.0*bearing,
                    -0.30,
                    0.30,
                )
            )

        self.state = 'CHASING'

        if self.plan is None and d > 0.28:
            T = max(
                4.0,
                1.6*(d - 0.20)/0.16,
            )

            ax = solve_flat_polynomial(
                0,
                0.06,
                d - 0.20,
                0,
                T,
            )
            
            print (ax)

            ay = solve_flat_polynomial(
                0,
                0,
                0,
                0,
                T,
            )

            self.plan = (
                now, d, T, ax, ay
            )

            self.get_logger().info(
                f'PLAN: d0={d:.3f} m, T={T:.2f} s'
            )

        if self.plan is not None:
            start, d0, T, ax, ay = self.plan

            t = min(now - start, T)

            vff, wff = flat_velocity(
                ax, ay, t, T
            )

            reference = (
                ax[0]
                + ax[1]*t
                + ax[2]*t*t
                + ax[3]*t*t*t
            )

            actual_progress = d0 - d
            error = reference - actual_progress

            v = float(
                np.clip(
                    vff + 0.6*error,
                    0.0,
                    0.16,
                )
            )

            if now - start > T + 10.0:
                return self.finish(
                    'TRACKING_TIMEOUT', now
                )

            terminal = t >= T

        else:
            v = 0.0
            wff = 0.0
            terminal = True

        # Koreksi akhir jika lintasan referensi
        # selesai tetapi robot belum mencapai 20 cm.
        if terminal:
            self.state = 'FINAL_APPROACH'

            if self.terminal_since is None:
                self.terminal_since = now

            if now - self.terminal_since > 10.0:
                return self.finish(
                    'TRACKING_TIMEOUT', now
                )

            v = min(
                0.04,
                max(0.01, 0.6*(d - 0.20)),
            )

        # Batasi percepatan saat mengejar.
        # Pengereman guard tetap langsung.
        v = float(
            np.clip(
                v,
                max(
                    0.0,
                    self.last_v - 0.05*dt,
                ),
                self.last_v + 0.05*dt,
            )
        )

        self.last_v = v

        w = float(
            np.clip(
                wff + 1.5*bearing,
                -0.30,
                0.30,
            )
        )

        return v, w

    def step(self):
        now = time.monotonic()
        dt = max(
            now - self.previous_time,
            0.001,
        )
        self.previous_time = now

        v, w = self.decide(now, dt)

        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)

        self.pub.publish(msg)

        elapsed = now - self.started

        values = (
            self.data
            if now - self.data_time < 0.75
            else [float('nan')]*8
        )

        (
            command_v,
            command_w,
            measured_v,
            measured_w,
            *volts,
        ) = values

        row = [
            f'{elapsed:.3f}',
            *[
                f'{value:.6f}'
                for value in volts
            ],
            self.state,
            f'{command_v:.6f}',
            f'{command_w:.6f}',
            f'{measured_v:.6f}',
            f'{measured_w:.6f}',
            f'{self.tag[0]:.6f}',
            f'{self.tof[0]:.6f}',
        ]

        # Log rinci sekitar 20 Hz.
        self.trace.writerow(row)

        # Tabel terminal dan CSV sekitar 1 Hz.
        if elapsed >= self.next_log or self.done:
            self.next_log = (
                math.floor(elapsed) + 1.0
            )

            self.table.writerow(row)
            self.table_file.flush()
            self.trace_file.flush()

            print(
                f'{elapsed:6.2f} | '
                + ' '.join(
                    f'{value:8.3f}'
                    for value in volts
                )
                + f' | {self.state}',
                flush=True,
            )

    def close(self):
        if rclpy.ok():
            self.pub.publish(Twist())

        self.table_file.close()
        self.trace_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = FlatnessPlanner()

        while rclpy.ok() and not node.done:
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
