#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# from visualization_msgs.msg import Maker

from geometry_msgs.msg import PointStamped

# PointStamped
# Header:
#    builtin_interfaces/Time stamp
#    string frame_id

#Point:
#    float64 x
#    float64 y
#    float64 z

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import datetime
import os

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator

detective_object_position = '/robot2/detected_object_position'
#받는 변수 이름 설정

class TurtleBotCoordFollower(Node):
    def __init__(self):
        super().__init__('turtlebot_coord_follower')

        self.navigator = TurtleBot4Navigator(namespace='/robot3')
        #네비게이터 초기화

        self.subscription = self.create_subscription(
            PointStamped,
            detective_object_position,
            self.coord_callback,
            10
        )
        # 좌표 수신용 서브스크립션

        # 이미지 관련 초기화
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # 실제 카메라 토픽
            self.image_callback,
            10
        )

        # 좌표 수신 여부 및 저장 위치 변수
        self.received_coord = False
        self.coord_position = None

        # 현재 저장 위치 출력
        self.get_logger().info(f"📂 Current working directory: {os.getcwd()}")

    def coord_callback(self, msg: PointStamped):
        if not self.received_coord:
            x = msg.point.x
            y = msg.point.y
            z = msg.point.z
            self.get_logger().info(f"Received Coord at x={x:.2f}, y={y:.2f}, z={z:.2f}")
            #수신 좌표
            self.coord_position = [x, y]
            self.received_coord = True

    def image_callback(self, msg: Image):
        self.latest_image = msg

    def take_picture_with_context(self, position):
        if self.latest_image is not None:
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                x_str = f"{position[0]:.2f}".replace('.', '_').replace('-', 'm')
                y_str = f"{position[1]:.2f}".replace('.', '_').replace('-', 'm')

                filename = f"photo_{timestamp}_x{x_str}_y{y_str}.jpg"

                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
                cv2.imwrite(filename, cv_image)

                self.get_logger().info(f"✅ Image saved to {filename}")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to save image: {e}")
        else:
            self.get_logger().warn("⚠ No image received yet.")

    def wait_for_coord_and_navigate(self):
        self.get_logger().info("Waiting for Coord message...")

        while not self.received_coord:
            ##좌표를 받지 않으면 반복
            rclpy.spin_once(self)
            #이벤트루프 한번 실행
    
        if not self.navigator.getDockedStatus():
            self.navigator.info('Docking before initializing pose')
            self.navigator.dock()
        #docking 아닐시 dock

        initial_pose = self.navigator.getPoseStamped([-3.0, -3.65], 270.0)
        #정의된 기본 함수 
        self.navigator.setInitialPose(initial_pose)

        self.navigator.waitUntilNav2Active()
        self.navigator.undock()

        # 첫 번째 목적지: sub된 좌표
        goal_pose_1 = self.navigator.getPoseStamped(self.coord_position, 0.0)
        self.navigator.startToPose(goal_pose_1)

        self.get_logger().info("Reached object. Taking picture...")
        time.sleep(1.0)  # 이미지 안정화 대기
        self.take_picture_with_context(self.coord_position)

        self.get_logger().info("Proceeding to (-4.7, -3.3).")

        time.sleep(5)

        # 두 번째 목적지: (-4.7, -3.3) 차량 보관소
        goal_pose_2 = self.navigator.getPoseStamped([-4.7, -3.3], 270.0)
        self.navigator.startToPose(goal_pose_2)
        self.get_logger().info("Navigation complete.")

def main():
    rclpy.init()
    node = TurtleBotCoordFollower()

    try:
        node.wait_for_coord_and_navigate()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()
