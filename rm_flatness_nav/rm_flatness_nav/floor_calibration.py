import json
import math
import time
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from robomaster import robot


# Koordinat pusat tag dalam meter.
# X positif ke depan. Y positif ke kiri.
POINTS = [
    (0.30, 0.15),
    (0.30, -0.15),
    (1.50, -0.30),
    (1.50, 0.30),
]


def tag_center(frame, dictionary, params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=params,
    )

    if ids is None:
        return None

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    for i, marker_id in enumerate(ids.flatten()):
        if int(marker_id) != 3:
            continue

        # Proyeksikan pusat geometris tag ke gambar.
        square = np.float32([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ])

        transform = cv2.getPerspectiveTransform(
            square,
            corners[i].reshape(4, 2).astype(np.float32),
        )

        return cv2.perspectiveTransform(
            np.float32([[[0.5, 0.5]]]),
            transform,
        )[0, 0]

    return None


def main(args=None):
    rclpy.init(args=args)
    node = Node('floor_calibration')

    node.declare_parameter('conn_type', 'ap')
    node.declare_parameter('check_only', False)

    check_only = node.get_parameter('check_only').value

    package = (
        Path.home()
        / 'robomaster_ws/src/rm_flatness_nav'
    )
    output = package / 'config/floor_map.json'

    H = None
    expected_size = None

    if check_only:
        if not output.exists():
            node.destroy_node()
            rclpy.shutdown()
            raise FileNotFoundError(
                f'Kalibrasi belum ada: {output}'
            )

        saved = json.loads(output.read_text())
        H = np.array(saved['H'], dtype=np.float64)
        expected_size = saved['image_size']

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    params = cv2.aruco.DetectorParameters_create()

    ep = robot.Robot()
    connected = False
    video = False

    pixels = []
    samples = None
    next_print = 0.0

    try:
        ep.initialize(
            conn_type=node.get_parameter('conn_type').value
        )
        connected = True

        action = ep.robotic_arm.moveto(x=105, y=60)

        if not action.wait_for_completed(timeout=8):
            raise RuntimeError(
                'Arm belum menyelesaikan gerakan'
            )

        video = True

        if ep.camera.start_video_stream(
            display=False,
            resolution='720p',
        ) is False:
            raise RuntimeError('Video gagal dimulai')

        last_frame = time.monotonic()

        while rclpy.ok():
            try:
                frame = ep.camera.read_cv2_image(
                    timeout=0.5,
                    strategy='newest',
                )
            except Empty:
                frame = None

            if frame is None:
                if time.monotonic() - last_frame > 5.0:
                    raise RuntimeError(
                        'Tidak ada frame selama 5 detik'
                    )

                time.sleep(0.01)
                continue

            last_frame = time.monotonic()
            height, width = frame.shape[:2]

            if expected_size is not None:
                if [width, height] != expected_size:
                    raise RuntimeError(
                        'Resolusi berbeda dari kalibrasi'
                    )

            center = tag_center(
                frame, dictionary, params
            )

            if H is None:
                index = len(pixels)
                x, y = POINTS[index]

                text = (
                    f'POINT {index + 1}: '
                    f'X={x:.2f} Y={y:+.2f} m. '
                    'SPACE=capture'
                )

                if samples is not None and center is not None:
                    samples.append(center.copy())

                    if len(samples) == 15:
                        pixels.append(
                            np.median(samples, axis=0)
                        )
                        samples = None

                        print(
                            f'Titik {len(pixels)} tersimpan',
                            flush=True,
                        )

                        if len(pixels) == len(POINTS):
                            H, _ = cv2.findHomography(
                                np.float32(pixels),
                                np.float32(POINTS),
                                0,
                            )

                            if H is None:
                                raise RuntimeError(
                                    'Kalibrasi tidak valid'
                                )

                            if np.linalg.matrix_rank(H) < 3:
                                raise RuntimeError(
                                    'Titik kalibrasi tidak valid'
                                )

                            output.parent.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            saved = {
                                'H': H.tolist(),
                                'image_size': [width, height],
                                'ground_points_m': POINTS,
                                'pixel_points': (
                                    np.array(pixels).tolist()
                                ),
                                'arm_xy_mm': [105, 60],
                                'reference': (
                                    'front_ToF_floor_projection'
                                ),
                                'tag_id': 3,
                                'family': '36h11',
                                'tag_size_m': 0.080,
                            }

                            output.write_text(
                                json.dumps(saved, indent=2)
                            )

                            print(
                                f'Kalibrasi tersimpan: {output}',
                                flush=True,
                            )

            else:
                text = 'CHECK: show Tag 3. Q=quit'

                if center is not None:
                    xy = cv2.perspectiveTransform(
                        center.reshape(
                            1, 1, 2
                        ).astype(np.float32),
                        H,
                    )[0, 0]

                    x, y = map(float, xy)

                    distance = math.hypot(x, y)
                    bearing = math.degrees(
                        math.atan2(y, x)
                    )

                    text = (
                        f'X={x:.3f} Y={y:+.3f} '
                        f'd={distance:.3f} m '
                        f'angle={bearing:+.1f}'
                    )

                    if time.monotonic() >= next_print:
                        print(text, flush=True)
                        next_print = time.monotonic() + 1.0

            cv2.putText(
                frame,
                text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                'Floor calibration',
                cv2.resize(frame, (960, 540)),
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if (
                key == 32
                and H is None
                and center is not None
                and samples is None
            ):
                samples = []

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        if connected and video:
            try:
                ep.camera.stop_video_stream()
            except Exception:
                pass

        ep.close()
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
