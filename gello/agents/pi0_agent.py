#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass

# Agent の基本クラス（spacemouse_agent.py と同様）
from gello.agents.agent import Agent

# pretrained policy の読み込み（inference_pi0_policy.py を参考）
from lerobot.configs import parser
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


@dataclass
class Pi0AgentConfig:
    policy_path: str  # pretrained policy のパス need to be set
    task: str = "Pick up the red square block and Put it onto the white plate."
    verbose: bool = True
    use_delta_action: bool = False  # モデルが差分行動を出力する場合は True に設定
    use_joint_vel: bool = (
        False  # モデルが関節速度を入力として受け取る場合は True に設定
    )
    use_ee_pos_quat: bool = (
        False  # モデルがエンドエフェクタの位置・姿勢を入力として受け取る場合は True に設定
    )


def resize_image(np_array):
    size = (256, 256)
    """PIL を使って画像をリサイズする（RGB 向け）"""
    img = Image.fromarray(np_array)
    img = img.resize(size, Image.Resampling.BICUBIC)
    return np.array(img)


class Pi0Agent(Agent):
    """
    PI0Policy を用いて環境からの観測に基づくアクションを予測するエージェントです。
    このエージェントは、観測としてカメラ画像（例："ego_rgb_image"）と、関節位置・速度（"joint_pos", "joint_vel"）
    を受け取り、PI0Policy による推論結果としてアクションを返します。
    """

    def __init__(
        self,
        config: Pi0AgentConfig,
    ):
        self.verbose = config.verbose

        # GPU 利用可能なら GPU に、なければ CPU を使用
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.verbose:
                print("GPU is available. Using device:", self.device)
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                print("GPU is not available. Using device:", self.device)

        # pretrained の PI0Policy を読み込む（inference_pi0_policy.py を参照）
        self.policy = PI0Policy.from_pretrained(
            pretrained_name_or_path=config.policy_path, local_files_only=True
        )
        self.policy.eval()
        self.policy.to(self.device)

        # task 名は policy の入力の一部として利用
        self.task = config.task
        self.use_delta_action = config.use_delta_action
        self.use_joint_vel = config.use_joint_vel
        self.use_ee_pos_quat = config.use_ee_pos_quat

    def act(self, obs: dict) -> np.ndarray:
        """
        観測 obs から PI0Policy によりアクションを予測して返します。
        obs は以下のような構造を仮定:
        {
            "wrist_rgb": <np.ndarray shape (H, W, 3)>,
            "base_rgb": <np.ndarray shape (H, W, 3)>,
            "joint_positions": <np.ndarray shape (num_joints,)>,
            "joint_velocities": <np.ndarray shape (num_joints,)>,
            "ee_pos_quat": <np.ndarray shape (7,)>,
            "gripper_position": <np.ndarray shape (1,)>
        }
        """
        if self.verbose:
            print("Pi0Agent.act invoked")

        # カメラ画像の取得と変換 (inference_pi0_policy.py の例を参考)
        wrist_rgb: np.ndarray = obs["wrist_rgb"]
        base_rgb: np.ndarray = obs["base_rgb"]
        wrist_rgb = resize_image(wrist_rgb)
        base_rgb = resize_image(base_rgb)
        # PI0Policy では (1, 3, H, W) 形式が期待されるので軸を追加して転置&numpy から torch.Tensor に変換
        wrist_rgb = torch.from_numpy(
            wrist_rgb[np.newaxis, :].transpose(0, 3, 1, 2).astype(np.float32)
        )
        base_rgb = torch.from_numpy(
            base_rgb[np.newaxis, :].transpose(0, 3, 1, 2).astype(np.float32)
        )

        # 関節状態は position と velocity を連結
        joint_pos_np: np.ndarray = obs["joint_positions"].astype(np.float32)[
            :-1
        ]  # gripper を除く
        joint_vel_np: np.ndarray = obs["joint_velocities"].astype(np.float32)[
            :-1
        ]  # gripper を除く
        ee_pos_quat_np: np.ndarray = obs["ee_pos_quat"].astype(np.float32)  # (7,)
        gripper_pos_np: np.ndarray = (
            obs["gripper_position"].astype(np.float32).reshape([1])
        )  # (1,)

        joint_pos_pt = torch.from_numpy(joint_pos_np)  # (joint_num,)
        joint_vel_pt = torch.from_numpy(joint_vel_np)  # (joint_num,)
        ee_pos_quat_pt = torch.from_numpy(ee_pos_quat_np)  # (7,)
        gripper_pos_pt = torch.from_numpy(gripper_pos_np).reshape([1])  # (1,)

        joint_states_pt = torch.cat([joint_pos_pt, gripper_pos_pt], dim=0)[
            np.newaxis, :
        ]  # (1, joint_num*2+8)

        if self.use_joint_vel:
            joint_states_pt = torch.cat([joint_states_pt, joint_vel_pt], dim=0)
        if self.use_ee_pos_quat:
            joint_states_pt = torch.cat([joint_states_pt, ee_pos_quat_pt], dim=0)

        # Policy への入力辞書を作成
        policy_input = {
            "observation.images.base.rgb": base_rgb.to(self.device),
            "observation.images.wrist.rgb": wrist_rgb.to(self.device),
            "observation.state": joint_states_pt.to(self.device),
            "task": [self.task],
        }

        # 推論（inference_pi0_policy.py の例に則る）
        with torch.inference_mode():
            action = self.policy.select_action(policy_input)  # (1, action_dim)

        # 結果は torch.Tensor として返されるため、numpy に変換
        action = action.cpu().numpy()[0, :]  # (1, action_dim) -> (action_dim,) に変換

        if self.use_delta_action:
            # 行動が差分である場合、元に戻す
            action = action + np.concat([joint_pos_np, 0]) # gripperは差分を取っていないため、0

        return action


@parser.wrap()
def main(configs: Pi0AgentConfig):
    num_steps = 10
    configs.verbose = True

    # テスト用のダミー観測を作成（inference_pi0_policy.py の構造を参考）
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

    agent = Pi0Agent(configs)
    print("Starting inference test for Pi0Agent...")
    for step in range(num_steps):
        action = agent.act(dummy_obs)
        print(f"Step {step}: Action = {action}")
        time.sleep(0.1)


#  python gello/agents/pi0_agent.py --policy_path=models/pi0/030000_angleonly/pretrained_model
if __name__ == "__main__":
    main()
