#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Agent の基本クラス（spacemouse_agent.py と同様）
from gello.agents.agent import Agent

# trained policy の読み込み
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


def resize_image(np_array):
    size = (256, 256)
    """PIL を使って画像をリサイズする（RGB 向け）"""
    img = Image.fromarray(np_array)
    img = img.resize(size, Image.Resampling.BICUBIC)
    return np.array(img)


class DiffusionAgent(Agent):
    """
    DiffusionPolicy を用いて環境からの観測に基づくアクションを予測するエージェントです。
    このエージェントは、観測としてカメラ画像（例："ego_rgb_image"）と、関節位置・速度（"joint_pos", "joint_vel"）
    を受け取り、DiffusionPolicy による推論結果としてアクションを返します。
    """

    def __init__(
        self,
        policy_path: str,
        task: str = "Pick up the red square block and Put it onto the white plate.",
        verbose: bool = True,
    ):
        self.verbose = verbose

        # GPU 利用可能なら GPU に、なければ CPU を使用
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.verbose:
                print("GPU is available. Using device:", self.device)
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                print("GPU is not available. Using device:", self.device)

        # pretrained の DiffusionPolicy を読み込む（inference_pi0_policy.py を参照）
        self.policy = DiffusionPolicy.from_pretrained(
            pretrained_name_or_path=policy_path, local_files_only=True
        )
        self.policy.eval()
        self.policy.to(self.device)

        # task 名は policy の入力の一部として利用
        self.task = task

    def act(self, obs: dict) -> np.ndarray:
        """
        観測 obs から DiffusionPolicy によりアクションを予測して返します。
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
            print("DiffusionAgent.act invoked")

        # カメラ画像の取得と変換 (inference_pi0_policy.py の例を参考)
        wrist_rgb: np.ndarray = obs["wrist_rgb"]
        base_rgb: np.ndarray = obs["base_rgb"]
        wrist_rgb = resize_image(wrist_rgb)
        base_rgb = resize_image(base_rgb)
        # DiffusionPolicy では (1, 3, H, W) 形式が期待されるので軸を追加して転置&numpy から torch.Tensor に変換
        wrist_rgb = torch.from_numpy(
            wrist_rgb[np.newaxis, :].transpose(0, 3, 1, 2).astype(np.float32)
        )
        base_rgb = torch.from_numpy(
            base_rgb[np.newaxis, :].transpose(0, 3, 1, 2).astype(np.float32)
        )

        # 関節状態は position と velocity を連結
        joint_pos = torch.from_numpy(
            obs["joint_positions"].astype(np.float32)
        )  # (joint_num,)
        joint_vel = torch.from_numpy(
            obs["joint_velocities"].astype(np.float32)
        )  # (joint_num,)
        ee_pos_quat = torch.from_numpy(obs["ee_pos_quat"].astype(np.float32))  # (7,)
        gripper_pos = torch.from_numpy(
            obs["gripper_position"].astype(np.float32)
        ).reshape(
            [1]
        )  # (1,)

        # joint_states = torch.cat([joint_pos, joint_vel, ee_pos_quat, gripper_pos], dim=0)[np.newaxis,:] # (1, joint_num*2+8)
        joint_states = torch.cat([joint_pos, gripper_pos], dim=0)[
            np.newaxis, :
        ]  # (1, joint_num*2+8)

        # Policy への入力辞書を作成
        policy_input = {
            "observation.images.base.rgb": base_rgb.to(self.device),
            "observation.images.wrist.rgb": wrist_rgb.to(self.device),
            "observation.state": joint_states.to(self.device),
            "task": [self.task],
        }
        # 推論（inference_pi0_policy.py の例に則る）
        with torch.inference_mode():
            action = self.policy.select_action(policy_input)  # (1, action_dim)
        # 結果は torch.Tensor として返されるため、numpy に変換
        return action.cpu().numpy()[
            0, :
        ]  # (1, action_dim) -> (action_dim,) に変換して返す


# python gello/agents/diffusion_agent.py --policy_path ./models/diffusion/010000/pretrained_model --num_steps 10
def main():
    parser = argparse.ArgumentParser(description="Diffusion Agent Inference Test")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to the pretrained diffusion policy checkpoint",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="do something",
        help="Task name (used in policy input)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of inference steps"
    )
    args = parser.parse_args()

    # テスト用のダミー観測を作成（inference_pi0_policy.py の構造を参考）
    dummy_image = np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8)  # (H, W, 3)
    num_joints = 7  # 適宜調整
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

    agent = DiffusionAgent(policy_path=args.policy_path, task=args.task, verbose=True)
    print("Starting inference test for DiffusionAgent...")
    for step in range(args.num_steps):
        action = agent.act(dummy_obs)
        print(f"Step {step}: Action = {action}")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
