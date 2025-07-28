#!/usr/bin/env python3

import rclpy # ROS 2 Python 클라이언트 라이브러리
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator

INITIAL_POSE_POSITION = [-0.2612, -0.0462]
# 초기 위치 좌표 [x, y]
INITIAL_POSE_DIRECTION = 0.0
# 초기 방향 (단위: 라디안 아님, degrees. 예: 0.0은 정면)

GOAL_POSES = [
    ([-0.47, 1.92], 0.0),
    ([-0.5109, 0.5069], 0.0),
    ([-2.7932, 0.5308], 0.0),
    ([-1.79, 1.92], 0.0),
    ([-3.03, 2.01], 0.0),
    ([-2.98, -0.12], 270.0),
    ([-2.7094, 0.6813], 270.0),
    ([-2.1615, 0.1634], 180.0),
    ([-0.60, 0.06], 0.0),
]
# 목표 위치 목록: 각 항목은 (위치 [x, y], 방향 angle in degrees)
# TurtleBot이 이동할 경로

def main():
    rclpy.init()
    # ROS 2 노드 초기화
    navigator = TurtleBot4Navigator()
    # TurtleBot4Navigator 객체 생성 

    if not navigator.getDockedStatus():
        navigator.info('Docking before initializing pose')
        navigator.dock()
    # 현재 TurtleBot이 도킹 상태가 아니라면, 도킹부터 수행

    initial_pose = navigator.getPoseStamped(INITIAL_POSE_POSITION, INITIAL_POSE_DIRECTION)
    # 초기 위치를 PoseStamped 메시지로 생성
    navigator.setInitialPose(initial_pose)
    # 초기 위치를 설정 (로봇의 위치 추정에 필요)
    navigator.waitUntilNav2Active()
    # Nav2가 활성화될 때까지 대기 (Localization 및 탐색 노드 준비)

    navigator.undock()
    # 도킹 해제
    goal_pose_msgs = [navigator.getPoseStamped(position, direction) for position, direction in GOAL_POSES]
    # 목표 위치들을 PoseStamped 메시지로 변환
    navigator.startFollowWaypoints(goal_pose_msgs)
    # 경로 추종 시작 (여러 지점을 따라 이동)
    navigator.dock()

    rclpy.shutdown()
    # ROS 2 종료

if __name__ == '__main__':
    main()
