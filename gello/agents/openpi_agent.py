#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import tyro
import numpy as np
import torch
from dataclasses import dataclass
from collections import deque

# Agent の基本クラス（spacemouse_agent.py と同様）
from gello.agents.agent import Agent

# pretrained policy の読み込み（inference_pi0_policy.py を参考）
from openpi.training import config
from openpi.policies import policy_config


class OpenPiAgent(Agent):
    """
    PI0Policy を用いて環境からの観測に基づくアクションを予測するエージェントです。
    このエージェントは、観測としてカメラ画像（例："ego_rgb_image"）と、関節位置・速度（"joint_pos", "joint_vel"）
    を受け取り、PI0Policy による推論結果としてアクションを返します。
    """

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        prompt: str = "do something",
        verbose: bool = False,
        adopted_action_chunks: int = 15, # 一度の推論で得られるaction_chunkのうち、最初何個を使うか
    ):
        self.config: config.TrainConfig = config.get_config(config_name)

        # pretrained の policy を読み込む
        self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir)

        self.verbose: bool = verbose
        self.prompt: str = prompt
        self.use_delta_action: bool = True
        self.adopted_action_chunks: int = adopted_action_chunks
        self.action_queue: deque = deque(maxlen=adopted_action_chunks)

    def act(self, obs: dict) -> np.ndarray:
        """
        観測 obs から PI0Policy によりアクションを予測して返します。
        obs は以下のような構造を仮定:
        {
            "wrist_rgb": <np.ndarray shape (H, W, 3)>,
            "base_rgb": <np.ndarray shape (H, W, 3)>,
            "joint_positions": <np.ndarray shape (num_joints,)>, with gripper at the end
            "joint_velocities": <np.ndarray shape (num_joints,)>, with gripper at the end
            "ee_pos_quat": <np.ndarray shape (7,)>,
            "gripper_position": <np.ndarray shape (1,)>
        }
        """
        if self.verbose:
            print("OpenPiAgent.act invoked")
        
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            if self.use_delta_action:
                # 行動が差分である場合、元に戻す
                action = action + np.concat(
                    [obs["joint_positions"][:-1], np.array([0])]
                )  # gripperは差分を取っていないため、0
            return action

        # Policy への入力辞書を作成
        policy_input = {
            "wrist_rgb": obs["wrist_rgb"].astype(np.uint8),
            "base_rgb": obs["base_rgb"].astype(np.uint8),
            "state": obs["joint_positions"],  # 6 joints and 1 gripper
            "prompt": self.prompt,
        }

        action_chunk = self.policy.infer(policy_input)["actions"]
        self.action_queue.extend(action_chunk[1:self.adopted_action_chunks])
        action = action_chunk[0]  # 最初のアクションだけを返す

        if self.use_delta_action:
            # 行動が差分である場合、元に戻す
            action = action + np.concat(
                [obs["joint_positions"][:-1], np.array([0])]
            )  # gripperは差分を取っていないため、0

        return action


def make_dummy_obs() -> dict:
    # テスト用のダミー観測を作成
    dummy_image = np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8)  # (H, W, 3)
    num_joints = 7  # 適宜調整:gripperをjointに含めている場合は7, そうでない場合は6
    dummy_joint_pos = np.random.rand(num_joints)
    dummy_joint_vel = np.random.rand(num_joints)
    dummy_eepos = np.random.rand(7)
    dummy_gripper = np.random.rand(1)

    dummy_obs = {
        "wrist_rgb": dummy_image,
        "base_rgb": dummy_image,
        "joint_positions": dummy_joint_pos,
        "joint_velocities": dummy_joint_vel,
        "ee_pos_quat": dummy_eepos,
        "gripper_position": dummy_gripper,
    }
    return dummy_obs


def main(config_name: str, checkpoint_dir: str):
    num_steps = 10

    agent = OpenPiAgent(config_name, checkpoint_dir)
    print("Starting inference test for Pi0Agent...")
    for step in range(num_steps):
        dummy_obs = make_dummy_obs()
        action = agent.act(dummy_obs)
        print(f"Step {step}: Action = {action}")
        time.sleep(0.1)


# uv run gello/agents/openpi_agent.py --config_name pi0_lite6_low_mem_finetune --checkpoint_dir models/openpi/pi0_lite6_low_mem_finetune/lite6_pickplace/30000
if __name__ == "__main__":
    tyro.cli(main)
