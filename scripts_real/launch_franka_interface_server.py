import zerorpc
from polymetis import RobotInterface
import scipy.spatial.transform as st
import numpy as np
import torch

from polymetis import GripperInterface  




class FrankaInterface:
    def __init__(self):
        self.robot = RobotInterface(ip_address="localhost")   #localhost
        self.gripper = GripperInterface(ip_address="localhost")   #localhost

        
    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()

    def move_to_joint_positions(self, positions, time_to_go):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go
        )

    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )

    def update_desired_ee_pose(self, pose):
        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
        )

    def terminate_current_policy(self):
        self.robot.terminate_current_policy()

    def open_gripper(self, width=0.08, speed=0.02):

        self.gripper.goto(width=width, speed=speed, force=0.0)

    def close_gripper(self, force=20.0):
        
        self.gripper.grasp(width=0.0, speed=0.02, force=force)

s = zerorpc.Server(FrankaInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()
