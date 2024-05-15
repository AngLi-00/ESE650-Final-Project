@torch.jit.script
def compute_reward(root_states: torch.Tensor, target_position: torch.Tensor, obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack observations
    # Position deltas
    x_d = obs_buf[:, 0]
    y_d = obs_buf[:, 1]
    # Euler angles (pitch)
    pitch = obs_buf[:, 2]
    # Derivatives
    pitch_d = obs_buf[:, 3]
    yaw_d = obs_buf[:, 4]
    
    # Compute distance to target position
    dist_to_target = torch.sqrt(x_d**2 + y_d**2)
    
    # Reward for minimizing distance to target position
    distance_reward = -dist_to_target
    
    # Reward for maintaining balance by minimizing pitch and yaw
    pitch_stability_reward = -torch.abs(pitch)
    
    # Reward for minimizing oscillations in pitch rate and yaw rate
    pitch_d_stability_reward = -torch.abs(pitch_d)
    yaw_d_stability_reward = -torch.abs(yaw_d)
    
    # Combine rewards
    total_reward = distance_reward + pitch_stability_reward + pitch_d_stability_reward + yaw_d_stability_reward
    
    # Scale and normalize rewards using an exponential function to keep within a steady range
    temperature = 0.1
    scaled_total_reward = torch.exp(total_reward / temperature)
    
    reward_components = {
        "distance_reward": distance_reward,
        "pitch_stability_reward": pitch_stability_reward,
        "pitch_d_stability_reward": pitch_d_stability_reward,
        "yaw_d_stability_reward": yaw_d_stability_reward
    }
    
    return scaled_total_reward, reward_components
