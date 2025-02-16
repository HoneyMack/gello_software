# CAUTION
This is a fork of the original GELLO repository. The original repository can be found [here](https://github.com/wuphilipp/gello_software). 

This fork is used to support lite6 and [openpi](https://github.com/HoneyMack/openpi.git) policy execution.

The repo has been only tested with Ubuntu 22.04 and CUDA 12.2. (Ubuntu 20.04 maybe works.)
You can align the environment using the docker container provided in this [repo](https://github.com/HoneyMack/docker-deep-cuda-py3/tree/develop)

The following README is also modified to align with the changes made in this fork. 

# GELLO
This is the central repo that holds the all the software for GELLO. See the website for the paper and other resources for GELLO https://wuphilipp.github.io/gello_site/
See the GELLO hardware repo for the STL files and hardware instructions for building your own GELLO https://github.com/wuphilipp/gello_mechanical
```
git clone https://github.com/HoneyMack/gello_software.git -b feature/openpi
cd gello_software
```

<p align="center">
  <img src="imgs/title.png" />
</p>


# Install Gello
```
git submodule init
git submodule update
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install -e .
```

# GELLO configuration setup (PLEASE READ)
Now that you have downloaded the code, there is some additional preparation work to properly configure the Dynamixels and GELLO.
These instructions will guide you on how to update the motor ids of the Dynamixels and then how to extract the joint offsets to configure your GELLO.

## Update motor IDs
Install the [dynamixel_wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/).
By default, each motor has the ID 1. In order for multiple dynamixels to be controlled by the same U2D2 controller board, each dynamixel must have a unique ID.
This process must be done one motor at a time. Connect each motor, starting from the base motor, and assign them in increasing order until you reach the gripper.

Steps:
 * Connect a single motor to the controller and connect the controller to the computer.
 * Open the dynamixel wizard
 * Click scan (found at the top left corner), this should detect the dynamixel. Connect to the motor
 * Look for the ID address and change the ID to the appropriate number.
 * Repeat for each motor

## Create the GELLO configuration and determining joint ID's
After the motor ID's are set, we can now connect to the GELLO controller device. However each motor has its own joint offset, which will result in a joint offset between GELLO and your actual robot arm.
Dynamixels have a symmetric 4 hole pattern which means there the joint offset is a multiple of pi/2.
The `GelloAgent` class  accepts a `DynamixelRobotConfig` (found in `gello/agents/gello_agent.py`). The Dynamixel config specifies the parameters you need to find to operate your GELLO. Look at the documentation for more details.

We have created a simple script to automatically detect the joint offset:
* set GELLO into a known configuration, where you know what the corresponding joint angles should be. For example, we set out GELLO in this configuration, where we know the desired ground truth joints. (0, -90, 90, -90, -90, 0)
<p align="center">
  <img src="imgs/gello_matching_joints.jpg" width="45%"/>
  <img src="imgs/robot_known_configuration.jpg" width="45%"/>
</p>

* run 
```
uv run scripts/gello_get_offset.py \
    --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \ # in radians
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6 \
    --start-id 2 \  # dynamixel servo ID starts start_id
    --baudrate 1000000 # baudrate of dynamixel servos
     # replace values with your own
```
* Use the known starting joints for `start-joints`.
* Use the `joint-signs` for your own robot (see below).
* Use your serial port for `port`. You can find the port id of your U2D2 Dynamixel device by running `ls /dev/serial/by-id` and looking for the path that starts with `usb-FTDI_USB__-__Serial_Converter` (on Ubuntu). On Mac, look in /dev/ and the device that starts with `cu.usbserial`

`joint-signs` for each robot type:
* UR: `1 1 -1 1 1 1`
* Panda: `1 -1 1 1 1 -1 1`
* xArm: `1 1 1 1 1 1 1`
* lite6: `1 1 1 1 1 1`

The script prints out a list of joint offsets. Go to `gello/agents/gello_agent.py` and add a DynamixelRobotConfig to the PORT_CONFIG_MAP. You are now ready to run your GELLO!

# Using GELLO to control a robot!

The code provided here is simple and only relies on python packages. The code does NOT use ROS, but a ROS wrapper can easily be adapted from this code.
For multiprocessing, we leverage [ZMQ](https://zeromq.org/)

## Testing in sim
First test your GELLO with a simulated robot to make sure that the joint angles match as expected.
In one terminal run
```
uv run experiments/launch_nodes.py --robot <sim_ur, sim_panda, sim_xarm or sim_lite6>
```
This launched the robot node. A simulated robot using the mujoco viewer should appear.

Then, launch your GELLO (the controller node).
```
uv run experiments/run_env.py --agent=gello
```
You should be able to use GELLO to control the simulated robot!

## Running on a real robot.
Once you have verified that your GELLO is properly configured, you can test it on a real robot!

Before you run with the real robot, you will have to install a robot specific python package.
The supported robots are in `gello/robots`.
 * UR: [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html)
 * panda: [polymetis](https://facebookresearch.github.io/fairo/polymetis/installation.html). If you use a different framework to control the panda, the code is easy to adpot. See/Modify `gello/robots/panda.py`
 * xArm: [xArm python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
 * Lite6: [xArm python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)

```
# Launch all of the node
uv run experiments/launch_nodes.py --robot=<your robot>
# run the enviroment loop
uv run experiments/run_env.py --agent=gello --hz=50 --gello_port=/dev/ttyUSB0
```

Ideally you can start your GELLO near a known configuration each time. If this is possible, you can set the `--start-joint` flag with GELLO's known starting configuration. This also enables the robot to reset before you begin teleoperation.

## Collect data

We have provided a simple example for collecting data with gello.
To save trajectories with the keyboard, add the following flag `--use-save-interface`

Moreover, if you want to convert raw data with gello to lerobot, please follow instructions bellow.

### Prerequisites
ffmpeg with svt-av1 codec is required to convert gello raw data to lerobot dataset. 
I provide a script to install ffmpeg with svt-av1 codec. 
```bash
./installer/install_ffmpeg.sh # PLEASE RUN ONLY ONCE!
 # reload bash or source ~/.bashrc after installing ffmpeg to reflect bin path.
```
### convertion
Now you can convert dataset!
Data can then be converted to lerobot dataset using the gello_to_lerobot script.

```
uv run gello/data_utils/gello_to_lerobot.py --raw_dir <raw data dir> ----out_dir <out data dir> 
# example
uv run gello/data_utils/gello_to_lerobot.py --raw_dir ./data/gello_datasets/lite6 --out_dir ./data/lerobot_datasets/lite6_pick_place --fps 30 --video true --use_delta_action true
```

## Running a bimanual system with GELLO
GELLO also be used in bimanual configurations.
For an example, see the `bimanual_ur` robot in `launch_nodes.py` and `--bimanual` flag in the `run_env.py` script.

## Notes
Due to the use of multiprocessing, sometimes python process are not killed properly. We have provided the kill_nodes script which will kill the
python processes.
```
./kill_nodes.sh
```

### Using a new robot!
If you want to use a new robot you need a GELLO that is compatible. If the kiniamtics are close enough, you may directly use an existing GELLO. Otherwise you will have to design your own.
To add a new robot, simply implement the `Robot` protocol found in `gello/robots/robot`. See `gello/robots/panda.py`, `gello/robots/ur.py`, `gello/robots/xarm_robot.py` for examples.

### Contributing
Please make a PR if you would like to contribute! The goal of this project is to enable more accessible and higher quality teleoperation devices and we would love your input!

You can optionally install some dev packages.
```
pip install -r requirements_dev.txt
```

The code is organized as follows:
 * `scripts`: contains some helpful python `scripts`
 * `experiments`: contains entrypoints into the gello code
 * `gello`: contains all of the `gello` python package code
    * `agents`: teleoperation agents
    * `cameras`: code to interface with camera hardware
    * `data_utils`: data processing utils. used for imitation learning
    * `dm_control_tasks`: dm_control utils to build a simple dm_control enviroment. used for demos
    * `dynamixel`: code to interface with the dynamixel hardware
    * `robots`: robot specific interfaces
    * `zmq_core`: zmq utilities for enabling a multi node system


This code base uses `isort` and `black` for code formatting.
pre-commits hooks are great. This will automatically do some checking/formatting. To use the pre-commit hooks, run the following:
```
pip install pre-commit
pre-commit install
```

# Citation

```
@misc{wu2023gello,
    title={GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators},
    author={Philipp Wu and Yide Shentu and Zhongke Yi and Xingyu Lin and Pieter Abbeel},
    year={2023},
}
```

# License & Acknowledgements
This source code is licensed under the MIT license found in the LICENSE file. in the root directory of this source tree.

This project builds on top of or utilizes the following third party dependencies.
 * [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie): Prebuilt robot models for mujoco
 * [brentyi/tyro](https://github.com/brentyi/tyro): Argument parsing and configuration
 * [ZMQ](https://zeromq.org/): Enables easy create of node like processes in python.
