#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from rcl_interfaces.srv import GetParameters

import math
import numpy as np
import matplotlib.pyplot as plt  # MATLAB plotting functions


class PoppyController(Node):
    def __init__(self):
        super().__init__('poppy_controller')

        self.get_logger().info("Starting the controller")

        self.cmd_publisher_ = self.create_publisher(
            Float64MultiArray, '/joint_group_position_controller/commands', 10)

        self.joint_state_subscription_ = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)

        self.get_joint_cmd_names()

        self.cmd_ = Float64MultiArray()
        for _ in range(self.joint_count()):
            self.cmd_.data.append(0)

        self.wait_for_initial_position = True

        l_arm = ['l_shoulder_y','l_shoulder_x','l_arm_z','l_elbow_y']

    # Called after the a JointState message arrives if self.wait_for_initial_position is True
    def init(self):
        # get
        self.get_motor_position(l_arm[i])

        # set a non-singular initial pose
        self.set_motor_position('l_elbow_y', -0.5)
        self.set_motor_position('l_shoulder_x', 0.5)

        self.cmd_publisher_.publish(self.cmd_)

        # Wait until the initial pose is reached
        while True:
            error = 0.0
            for i in range(len(self.cmd_.data)):
                error += math.fabs(self.cmd_.data[i] -
                                   self.joint_positions_[i])
            if error < 0.01:
                break
            else:
                return

        self.wait_for_initial_position = False

        self.t0 = self.get_time()

        self.get_logger().info("Reached initial joint position (%s), starting the control loop" %
                               self.joint_positions_)

        # TODO read initial state here

        # Init done, now start the control loop
        self.run_timer = self.create_timer(0.1, self.run)

    def run(self):
        dt = self.get_time() - self.t0

        # TODO implement control loop here

        self.cmd_publisher_.publish(self.cmd_)

    def get_motor_position(self, joint_name):
        index = self.joint_state_index(joint_name)
        return self.joint_positions_[index]

    def set_motor_position(self, joint_name, joint_pos):
        index = self.joint_cmd_index(joint_name)
        self.cmd_.data[index] = joint_pos

    def joint_state_callback(self, msg):
        self.joint_names_ = msg.name
        self.joint_positions_ = msg.position
        if self.wait_for_initial_position:
            self.init()

    def joint_state_index(self, joint_name):
        return self.joint_state_names_.index(joint_name)

    def joint_cmd_index(self, joint_name):
        return self.joint_cmd_names_.index(joint_name)

    def get_joint_cmd_names(self):
        self.cmd_params = self.create_client(
            GetParameters, '/joint_group_position_controller/get_parameters')
        self.cmd_params.wait_for_service()
        srv_params = GetParameters.Request()
        srv_params.names = ["joints"]
        resp_fut = self.cmd_params.call_async(srv_params)
        rclpy.spin_until_future_complete(self, resp_fut)
        self.joint_cmd_names_ = resp_fut.result().values[0].string_array_value

    def joint_count(self):
        return len(self.joint_cmd_names_)

    def get_time(self):
        sec_nsec = self.get_clock().now().seconds_nanoseconds()
        return sec_nsec[0] + 1e-9 * sec_nsec[1]


def main(args=None):
    rclpy.init(args=args)

    poppy_controller = PoppyController()

    try:
        rclpy.spin(poppy_controller)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
