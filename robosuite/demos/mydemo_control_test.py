"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import argparse

import imageio
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import make
from robosuite.utils.camera_utils import CameraMover

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Stack")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--camera", type=str, default="agentview", help="Name of camera to render")
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    controller_config = suite.load_controller_config(default_controller="OSC_POSE")

    # initialize an environment with offscreen renderer
    env = make(
        args.environment,
        args.robots,
        controller_configs=controller_config,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        single_object_mode=1,
    )
    # camera_mover.move_camera(np.array([0,1,0]), 1.0)
    # camera_pose = camera_mover.set_camera_pose(np.array([0.5, 2, 1.35]))
    obs = env.reset()
    ndim = env.action_dim

    camera_mover = CameraMover(
        env=env,
        camera=args.camera,
    )

    camera_mover.move_camera(direction=[0.0, 0.0, 1.0], scale=1.0)
    # camera_mover.rotate_camera(point=None, axis=[1.0, 0.0, 0.0], angle=20)
    # create a video writer with imageio
    print(camera_mover.get_camera_pose())
    # camera_mover.set_camera_pose(np.array([1.20806503, 0, 2.0]))
    # print(camera_mover.get_camera_pose())
    writer = imageio.get_writer(args.video_path, fps=20)

    # Keep track of done variable to know when to break loop
    action_xyz_seq = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    # action_xyz_seq = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    steps_per_action = 150
    frames = []
    for i, action_xyz in enumerate(action_xyz_seq):
        action = np.zeros(ndim)
        action[0:3] = action_xyz
        # odict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_proprio-state'])
        print(obs['robot0_eef_pos'])
        # print(obs['robot0_eef_pos'])
        print("\n")
        for j in range(steps_per_action):
            # run a uniformly random agent
            obs, reward, done, info = env.step(action)

            # dump a frame from every K frames
            if j % args.skip_frame == 0:
                frame = obs[args.camera + "_image"]
                writer.append_data(frame)

                # print("Saving frame #{}".format(i * steps_per_action + j))
    writer.close()
