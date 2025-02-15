#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gello 形式の pickle ファイル群から，LeRobotDatasetV2 形式に変換するサンプルコード
"""

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import tqdm
from PIL import Image
from dataclasses import dataclass

# LeRobotDatasetV2 の作成用 API（zarr 形式の変換例を参考）
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import encode_video_frames, save_images_concurrently


@dataclass
class ConvertConfig:
    # gello データが格納されているディレクトリ
    raw_dir: str
    # 出力先ディレクトリ
    out_dir: str
    # データセットのfps
    fps: int = 10
    # 画像データを動画形式としてエンコードする場合に指定
    video: bool = True
    # デバッグモード（1 エピソードのみ処理）
    debug: bool = False
    # アクションを角度の差分に変換するかどうか
    use_delta_action: bool = False
    # 関節速度を追加するかどうか
    use_joint_vel: bool = False
    # エンドエフェクタの位置・姿勢を追加するかどうか
    use_ee_pos_quat: bool = False
    # 深度画像を追加するかどうか
    add_depth: bool = False


PUSHT_TASK = "Pick up the red square block and Put it onto the white plate."


def define_gello_features(cfg: ConvertConfig) -> Dict[str, Any]:
    # 新たなキー構成に合わせた features を定義
    gello_features = {
        "observation.images.base.rgb": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": 10.0,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": "false",
                "has_audio": "false",
            },
        },
        "observation.images.wrist.rgb": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": 10.0,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": "false",
                "has_audio": "false",
            },
        },
        "observation.state": {
            "dtype": "float32",
            # CAUTION: ここでのjoint_positionsにgripper_positionは含まれない
            "shape": (7,),  # 6 (joint_positions) + 1 (gripper_position)
            "names": {
                "joint_positions": [f"joint_pos_{i}" for i in range(6)],
                "gripper_position": ["gripper_position"],
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),  # control (7 要素)
            "names": [f"control_{i}" for i in range(7)],
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "episode_index": {
            "dtype": "int32",
            "shape": (1,),
            "names": None,
        },
    }

    if cfg.add_depth:
        gello_features["observation.images.base.depth"] = {
            "dtype": "video",
            "shape": (256, 256),
            "names": ["height", "width"],
            "video_info": {
                "video.fps": 10.0,
                "video.codec": "libx265",
                "video.pix_fmt": "gray",
                "video.is_depth_map": "true",
                "has_audio": "false",
            },
        }
        gello_features["observation.images.wrist.depth"] = {
            "dtype": "video",
            "shape": (256, 256),
            "names": ["height", "width"],
            "video_info": {
                "video.fps": 10.0,
                "video.codec": "libx265",
                "video.pix_fmt": "gray",
                "video.is_depth_map": "true",
                "has_audio": "false",
            },
        }

    if cfg.use_joint_vel:
        gello_features["observation.state"]["shape"] = gello_features["observation.state"]["shape"] + (7,)
        gello_features["observation.state"]["names"]["joint_velocities"] = [f"joint_vel_{i}" for i in range(6)]

    if cfg.use_ee_pos_quat:
        gello_features["observation.state"]["shape"] = gello_features["observation.state"]["shape"] + (7,)
        gello_features["observation.state"]["names"]["ee_pos_quat"] = [f"ee_pos_quat_{i}" for i in range(7)]

    return gello_features


def resize_image(np_array, size=(256, 256)):
    """PIL を使って画像をリサイズする（RGB 向け）"""
    img = Image.fromarray(np_array)
    img = img.resize(size, Image.Resampling.BICUBIC)
    return np.array(img)


def process_depth_image(np_array, size=(256, 256)):
    """
    深度画像 (uint16, shape=(H, W, 1)) を，
    1 チャンネルにして2で割って正規化し，uint8 化した上でリサイズする
    """
    # 1 チャンネル抽出
    depth = np_array[..., 0]
    # 2で割って正規化
    depth = (depth / 2).astype(np.uint8)

    # リサイズ
    img = Image.fromarray(depth)
    img = img.resize(size, Image.Resampling.BICUBIC)
    return np.array(img)


def load_gello_episode(cfg: ConvertConfig, episode_path: Path):
    """
    指定されたエピソードディレクトリ内の pickle ファイルを読み込み，
    各ステップのデータを前処理して，フレームごとの辞書リストとして返す
    """
    step_paths = sorted(list(episode_path.glob("*.pkl")))
    frames = []
    for step_idx, step_path in enumerate(step_paths):
        with open(step_path, "rb") as f:
            data = pickle.load(f)

        frame = {}
        # --- 画像データの変換 ---
        if "base_rgb" in data:
            # base カメラ RGB 画像 → observation.images.base.rgb
            rgb = data["base_rgb"]
            rgb = resize_image(rgb, (256, 256))
            frame["observation.images.base.rgb"] = Image.fromarray(rgb)
        if "base_depth" in data and cfg.add_depth:
            depth = data["base_depth"]
            depth = process_depth_image(depth, (256, 256))
            frame["observation.images.base.depth"] = Image.fromarray(depth)
        if "wrist_rgb" in data:
            rgb = data["wrist_rgb"]
            rgb = resize_image(rgb, (256, 256))
            frame["observation.images.wrist.rgb"] = Image.fromarray(rgb)
        if "wrist_depth" in data and cfg.add_depth:
            depth = data["wrist_depth"]
            depth = process_depth_image(depth, (256, 256))
            frame["observation.images.wrist.depth"] = Image.fromarray(depth)

        # --- 行動データ ---
        if "control" in data:
            frame["action"] = torch.tensor(data["control"], dtype=torch.float32)

        # --- 状態データの構築 ---
        # state を joint_positions, gripper_position (+ joint_velocities, ee_pos_quat) の連結とする
        state_parts = []

        state_parts.append(
            torch.tensor(data["joint_positions"][:-1], dtype=torch.float32)
        )  # 最後にgripper_positionがあるため除く
        state_parts.append(
            torch.tensor(data["gripper_position"], dtype=torch.float32).reshape(
                1,
            )
        )
        if cfg.use_joint_vel:
            state_parts.append(
                torch.tensor(data["joint_velocities"][:-1], dtype=torch.float32)
            )  # 最後にgripper_velocityがあるため除く
        if cfg.use_ee_pos_quat:
            state_parts.append(torch.tensor(data["ee_pos_quat"], dtype=torch.float32))

        # 連結
        frame["observation.state"] = torch.cat(state_parts, dim=0)

        # --- 補助情報 ---
        # 最終フレームのみ done=True とする
        is_last = step_idx == len(step_paths) - 1
        frame["next.done"] = torch.tensor(is_last, dtype=torch.bool)
        frame["next.success"] = torch.tensor(is_last, dtype=torch.bool)
        # 報酬が存在しない場合は 0.0 を設定
        frame["next.reward"] = torch.tensor(0.0, dtype=torch.float32)

        frames.append(frame)

    if cfg.use_delta_action:
        # 行動を差分に変換. ただし，gripperは差分を取らない
        for i in range(0, len(frames) - 1):
            frames[i]["action"][:-1] = frames[i + 1]["action"][:-1] - frames[i]["action"][:-1]
        # 最後は何もしない
        frames[-1]["action"][:-1] = torch.zeros_like(frames[-1]["action"][:-1], dtype=torch.float32)

    return frames


@parser.wrap()
def convert_gello_to_lerobot_format(cfg: ConvertConfig):
    """
    gello 形式のデータ（pickle ファイル群）を読み込み，
    LeRobotDatasetV2 形式に変換して out_dir に保存する関数
    """
    raw_dir = Path(cfg.raw_dir)
    out_dir = Path(cfg.out_dir)

    gello_features = define_gello_features(cfg)
    # LeRobotDatasetV2 のインスタンスを作成
    dataset = LeRobotDataset.create(
        repo_id="gello_dataset",
        fps=cfg.fps,
        root=out_dir,
        robot_type="lite6",
        features=gello_features,
        tolerance_s=0.5,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    # raw_dir 内の各エピソードフォルダを処理
    episode_paths = sorted([p for p in raw_dir.glob("*") if p.is_dir()])
    for episode_idx, episode_path in enumerate(episode_paths):
        print(f"[INFO]: Processing episode {episode_path}")
        frames = load_gello_episode(cfg, episode_path)
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode(task=PUSHT_TASK)
        if cfg.debug:
            break

    # 最終的にデータセットを確定（不要な画像ファイルは削除）
    dataset.consolidate(keep_image_files=True)
    print(f"[INFO]: Dataset saved to {out_dir}")
    return dataset


# uv run gello/data_utils/gello_to_lerobot.py --raw_dir ./data/gello_datasets/lite6 --out_dir ./data/lerobot_datasets/lite6_pick_place --fps 30 --video true --use_delta_action true
if __name__ == "__main__":
    convert_gello_to_lerobot_format()
