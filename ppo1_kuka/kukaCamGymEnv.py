import os
import random

import gym
import numpy as np
import pybullet as p
import pybullet_data as data
from gym import spaces
from gym.utils import seeding
# import data
from pkg_resources import parse_version

from kuka import Kuka

maxSteps = 1000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class KukaCamGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 block_pos=None,
                 block_orn=None
                 ):
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = 128
        self._height = 128
        self._isDiscrete = isDiscrete
        self.terminated = 0
        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        self.block_pos = block_pos
        self.block_orn = block_orn
        self.seed()
        # self.reset()
        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 4
            # self._action_bound = 1
            # action_high = np.array([self._action_bound] * action_dim)
            action_high = np.array([0.01, 0.01, 0.01, 1])

            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self._height, self._width, 4),
                                            dtype=np.uint8)
        self.viewer = None

    def reset(self):
        self.terminated = 0
        # self.Judge_arm = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -0.65])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.62500,
                   0.000000, 0.000000, 0.0, 1.0)
        block_pos = []
        block_orn = []
        if self.block_pos == None:
            xpos = 0.51 + 0.11 * random.random()
            ypos = 0.225 + 0.025 * random.random()
            block_pos = [xpos, ypos, 0.026]
            self.block_pos = block_pos
        elif len(self.block_pos) == 3:
            block_pos = self.block_pos
        else:
            assert self.block_pos + " block position is error !"
        if self.block_orn == None:
            orn_z = 1.52 + 1.52 * random.random()
            self.block_orn = [0, 0, orn_z]
            block_orn = p.getQuaternionFromEuler(self.block_orn)
        elif len(self.block_orn) == 3:
            block_orn = p.getQuaternionFromEuler(self.block_orn)
        else:
            assert self.block_orn + " block　orientation is error !"

        block_path = os.path.dirname(__file__) + "/kuka_iiwa/block.urdf"
        # block_path = "/Doc/PPO/kuka_iiwa/block.urdf"
        self.blockUid = p.loadURDF(
            os.path.join(self._urdfRoot, block_path), block_pos, block_orn)
        p.setGravity(0, 0, -10)
        self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def resetBlockPos(self, posObj, ornObj=None):
        if ornObj == None:
            ornObj = p.getQuaternionFromEuler([0, 0, 1.52])
        else:
            ornObj = p.getQuaternionFromEuler(ornObj)
        p.resetBasePositionAndOrientation(self.blockUid, posObj, ornObj)

    def getBlock_Pos_orn(self):
        pos, orn = p.getBasePositionAndOrientation(self.blockUid)
        orn = p.getEulerFromQuaternion(orn)
        return pos, orn

    def close_arm_block(self):
        closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                           self._kuka.kukaEndEffectorIndex)
        numPt = len(closestPoints)
        if numPt > 0:
            return float(closestPoints[0][8])
        return 10

    def getControlLinkState(self):
        return p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):

        viewMat = [
            -0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722,
            -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843,
            0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0
        ]
        projMatrix = [
            0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
            -0.02000020071864128, 0.0
        ]

        img_arr = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=viewMat,
                                   projectionMatrix=projMatrix)
        rgb = img_arr[2]
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        self._observation = np_img_arr
        return self._observation

    def step(self, act):
        action = act
        if len(action) == 3:
            action = np.array([action[0], action[1], action[2], 0, 0.25])
        if len(action) == 4:
            action = np.array([action[0], action[1], action[2], action[3], 0.25])
        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            # if self._termination():
            #   break
            self._envStepCounter += 1
        self._observation = self.getExtendedObservation()
        # self.Up_arm(action)
        self.Judge_Pos()
        done = self.terminated
        reward = self._reward()
        return np.array(self._observation), reward, done, {}

    # 判断手臂师傅超过指定的位置
    def Judge_Pos(self):
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
        armpPos = state[0]
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        diff = armpPos[2] - blockPos[2]
        if armpPos[2] < 0.25 or armpPos[2] > 1: self.terminated = True
        # 手臂高度低于0.225或者手臂位置大于１　
        # if diff < 0.2 or diff > 1: self.terminated = True
        # if diff < 0.2 or diff > 1: self.terminated = True

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    # 此函数暂时没有用
    def _termination(self):
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        if (self.terminated or self._envStepCounter > maxSteps):
            self._observation = self.getExtendedObservation()
            return True
        maxDist = 0.005
        closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, maxDist)

        if (len(closestPoints)):  # (actualEndEffectorPos[2] <= -0.43):
            self.terminated = 1
            fingerAngle = 0.3
            for i in range(100):
                graspAction = [0, 0, 0.0001, 0, fingerAngle]
                self._kuka.applyAction(graspAction)
                p.stepSimulation()
                fingerAngle = fingerAngle - (0.3 / 100.)
                if (fingerAngle < 0):
                    fingerAngle = 0

            for i in range(1000):
                graspAction = [0, 0, 0.001, 0, fingerAngle]
                self._kuka.applyAction(graspAction)
                p.stepSimulation()
                blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
                if (blockPos[2] > 0.23):
                    break
                state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
                actualEndEffectorPos = state[0]
                if (actualEndEffectorPos[2] > 0.5):
                    break
            return True
        return False

    def _reward(self):

        # rewards is height of target object
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                           self._kuka.kukaEndEffectorIndex)
        reward = -1000
        numPt = len(closestPoints)
        if numPt > 0:
            dis = 0.40 - float(closestPoints[0][8])
            if dis > 0:
                reward = dis * 100
            # print("距离",float(closestPoints[0][8]))
        if (blockPos[2] > 0.2):
            reward = reward + 1000
        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
