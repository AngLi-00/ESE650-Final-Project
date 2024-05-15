@torch.jit.script
def compute_reward(root_states: torch.Tensor, target_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Device and tensor properties
    device = root_states.device

    # Extraction of necessary variables
    x_d = quat_rotate_inverse(root_states[:, 3:7], root_states[:, 7:10])[:, 0]  # Distance in x direction
    y_d = quat_rotate_inverse(root_states[:, 3:7], root_states[:, 7:10])[:, 1]  # Distance in y direction
    pitch = get_euler_xyz(root_states[:, 3:7])[:, 1]  # Pitch angle
    pitch_d = root_states[:, 11]  # Pitch velocity
    yaw_d = root_states[:, 12]  # Yaw velocity

    # Calculating the distance to the target position
    distance_to_target = torch.sqrt((x_d - target_pos[0]) ** 2 + (y_d - target_pos[1]) ** 2)
    
    # Reward Components
    target_pos_reward = -distance_to_target  # Reward for approaching the target position
    pitch_reward = -torch.abs(pitch)  # Reward for minimizing pitch
    yaw_rate_reward = -torch.abs(yaw_d)  # Reward for minimizing yaw rate

    # Combine rewards
    total_reward = target_pos_reward + pitch_reward + yaw_rate_reward

    # Optional: Apply exponential scaling to rewards with temperature parameters
    distance_temp = 1.0
    pitch_temp = 1.0
    yaw_temp = 1.0

    exp_target_pos_reward = torch.exp(target_pos_reward / distance_temp)
    exp_pitch_reward = torch.exp(pitch_reward / pitch_temp)
    exp_yaw_rate_reward = torch.exp(yaw_rate_reward / yaw_temp)

    # Scaled total reward
    total_reward_scaled = exp_target_pos_reward + exp_pitch_reward + exp_yaw_rate_reward

    # Individual components for logging or analysis
    reward_dict = {
        "target_pos_reward": exp_target_pos_reward,
        "pitch_reward": exp_pitch_reward,
        "yaw_rate_reward": exp_yaw_rate_reward
    }

    return total_reward_scaled, reward_dict
