@torch.jit.script
def compute_reward(obs_buf: torch.Tensor, target_position: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = obs_buf.device

    # Extracting useful components from observations
    x_d = obs_buf[:, 0]
    y_d = obs_buf[:, 1]
    pitch = obs_buf[:, 2]
    pitch_d = obs_buf[:, 3]
    yaw_d = obs_buf[:, 4]

    # Target positions (assuming target_position is a 2D tensor with x and y coordinates)
    target_x = target_position[0]
    target_y = target_position[1]

    # Distance to target position
    distance_to_target = torch.sqrt((x_d - target_x) ** 2 + (y_d - target_y) ** 2)

    # Compute components of the reward
    distance_reward = -distance_to_target 
    pitch_penalty = -torch.abs(pitch)
    pitch_d_penalty = -torch.abs(pitch_d)
    yaw_d_penalty = -torch.abs(yaw_d)

    # Combine all components into total reward
    reward = distance_reward + pitch_penalty + pitch_d_penalty + yaw_d_penalty

    # Reward component dictionary
    reward_components = {
        'distance_reward': distance_reward,
        'pitch_penalty': pitch_penalty,
        'pitch_d_penalty': pitch_d_penalty,
        'yaw_d_penalty': yaw_d_penalty
    }

    return reward, reward_components

# Example usage
# obs_buf = torch.tensor([[1.0, 2.0, 0.1, 0.2, 0.1]], device='cuda')
# target_position = torch.tensor([0.0, 0.0], device='cuda')
# compute_reward(obs_buf, target_position)
