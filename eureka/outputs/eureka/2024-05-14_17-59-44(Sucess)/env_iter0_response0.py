import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi, torch_utils
from .base.vec_task import VecTask

from isaacgymenvs.utils.torch_jit_utils import *

class TwipGPT(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 200

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        num_obs = 5

        num_acts = 2

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        print("dof state", self.dof_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)


        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        self.root_orientations = self.root_states[:, 3:7]
        print("get_euler_xyz(self.root_states[:, 3:7])", get_euler_xyz(self.root_states[:, 3:7]))
        print("self.root_states[:, 3:7]", self.root_states[:, 3:7])
        self.root_pitch = get_euler_xyz(self.root_states[:, 3:7])[1]
        self.root_yaw = get_euler_xyz(self.root_states[:, 3:7])[2]

        self.root_x_d = quat_rotate_inverse(self.root_orientations, self.root_states[:, 7:10])[:, :2]    #
        self.root_angular_vels = self.root_states[:, 10:13] #pitch_d, yaw_d

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)


        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 2] = 0.35

        self.initial_dof_states = self.dof_state.clone()

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]

    def create_sim(self):
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/twip.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        twip_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(twip_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 0.35
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.twip_handles = []
        self.envs = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            twip_handle = self.gym.create_actor(env_ptr, twip_asset, pose, "twip", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, twip_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, twip_handle, dof_props)

            self.envs.append(env_ptr)
            self.twip_handles.append(twip_handle)

    def compute_reward(self):
        self.rew_buf[:], self.rew_dict = compute_reward(self.obs_buf, self.root_states, self.target_pos)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        twip_pitch_d = self.obs_buf[:, 3]
        twip_yaw_d = self.obs_buf[:, 4]
        twip_vel = self.obs_buf[:, :2]

        twip_pitch = self.obs_buf[:, 2]

        self.gt_rew_buf, self.reset_buf[:] = compute_success(
            twip_vel, twip_pitch, twip_pitch_d, twip_yaw_d,
            self.commands, self.reset_buf, self.progress_buf, self.max_episode_length
        )
        self.extras['gt_reward'] = self.gt_rew_buf.mean()

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)



        self.obs_buf[env_ids, 0] = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])[env_ids,0].squeeze() #x_d
        self.obs_buf[env_ids, 1] = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])[env_ids,1].squeeze() #y_d
        self.obs_buf[env_ids, 2] = get_euler_xyz(self.root_states[env_ids, 3:7])[1].squeeze() #pitch
        
        num_resets = len(env_ids)
        treshold = torch.tensor([2], device=self.device)
        pi_tensor = torch.tensor([6.28318], device=self.device)
        self.obs_buf[env_ids, 2] = torch.where(self.obs_buf[env_ids, 2] > treshold, self.obs_buf[env_ids, 2] - pi_tensor, self.obs_buf[env_ids, 2])
        self.obs_buf[env_ids, 2] = torch.where(self.obs_buf[env_ids, 2] < -treshold, self.obs_buf[env_ids, 2] + pi_tensor, self.obs_buf[env_ids, 2])
        self.obs_buf[env_ids, 3] = self.root_states[env_ids, 11].squeeze() #pitch_d
        self.obs_buf[env_ids, 4] = self.root_states[env_ids, 12].squeeze() #yaw_d




        
        return self.obs_buf

    def reset_idx(self, env_ids):

        num_resets = len(env_ids)
        
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)


        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        temp_euler = get_euler_xyz(self.root_states[env_ids, 3:7])
        treshold = torch.tensor([2], device=self.device)
        pi_tensor = torch.tensor([6.28318], device=self.device)

        pitch = torch.tensor(temp_euler[1])
        yaw = torch.tensor(temp_euler[2])
        
        pitch = torch.where(pitch > treshold, pitch - pi_tensor, pitch)
        pitch = torch.where(pitch < -treshold, pitch + pi_tensor, pitch)
        

        
        pitch += pitch + torch_rand_float(-0.4, 0.4, (num_resets, 1), self.device).flatten() #randomized pitch
        yaw = torch.tensor(temp_euler[2]) + torch_rand_float(-3, 3, (num_resets, 1), self.device).flatten() #randomized yaw
        
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(temp_euler[0], pitch, yaw)

   

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.initial_dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))



        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.commands_x[env_ids] = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze()



    def pre_physics_step(self, actions):
        
        actions = actions.to(self.device)
        torques = gymtorch.unwrap_tensor(actions* self.max_push_effort)
        self.gym.set_dof_actuation_force_tensor(self.sim, torques)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()




@torch.jit.script
def compute_success(twip_vel, twip_pitch, twip_pitch_d, twip_yaw_d,
                            commands, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    
    reward = -(torch.abs(twip_pitch) - 0)*1
    reward += -(torch.abs(twip_pitch_d) - 0)*0.3 - (torch.abs(twip_yaw_d) - 0)*0.3


    reward = torch.where(torch.abs(twip_pitch) > np.pi / 3, torch.ones_like(reward) * -10.0, reward)
    
    reset = torch.where(torch.abs(twip_pitch) > np.pi / 3 , torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)




    return reward, reset

from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(
    obs_buf: torch.Tensor, 
    root_states: torch.Tensor, 
    target_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    pos_error_coeff = torch.tensor([0.5], device=obs_buf.device)
    pitch_error_coeff = torch.tensor([1.0], device=obs_buf.device)
    yaw_d_error_coeff = torch.tensor([0.5], device=obs_buf.device)
    pitch_d_error_coeff = torch.tensor([0.5], device=obs_buf.device)
    
    pos_error_temp = torch.tensor([0.2], device=obs_buf.device)
    pitch_error_temp = torch.tensor([0.5], device=obs_buf.device)
    yaw_d_error_temp = torch.tensor([0.2], device=obs_buf.device)
    pitch_d_error_temp = torch.tensor([0.2], device=obs_buf.device)

    x_d = obs_buf[:, 0]
    y_d = obs_buf[:, 1]
    pitch = obs_buf[:, 2]
    pitch_d = obs_buf[:, 3]
    yaw_d = obs_buf[:, 4]
    
    target_x = target_pos[:, 0]
    target_y = target_pos[:, 1]
    
    pos_error = torch.sqrt((x_d - target_x) ** 2 + (y_d - target_y) ** 2)
    pitch_error = torch.abs(pitch)
    pitch_d_error = torch.abs(pitch_d)
    yaw_d_error = torch.abs(yaw_d)
    
    pos_error = torch.exp(-pos_error / pos_error_temp) * pos_error_coeff
    pitch_error = torch.exp(-pitch_error / pitch_error_temp) * pitch_error_coeff
    pitch_d_error = torch.exp(-pitch_d_error / pitch_d_error_temp) * pitch_d_error_coeff
    yaw_d_error = torch.exp(-yaw_d_error / yaw_d_error_temp) * yaw_d_error_coeff
    
    total_reward = pos_error + pitch_error + pitch_d_error + yaw_d_error
    
    reward_components = {
        'pos_error': pos_error,
        'pitch_error': pitch_error,
        'pitch_d_error': pitch_d_error,
        'yaw_d_error': yaw_d_error
    }
    
    return total_reward, reward_components
