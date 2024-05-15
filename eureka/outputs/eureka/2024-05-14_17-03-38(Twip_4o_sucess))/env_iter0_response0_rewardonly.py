@torch.jit.script
def compute_reward(obs_buf: torch.Tensor, goal_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract observation values
    x_d = obs_buf[:, 0]  # x velocity
    y_d = obs_buf[:, 1]  # y velocity
    pitch = obs_buf[:, 2]  # pitch angle
    pitch_d = obs_buf[:, 3]  # pitch angular velocity
    yaw_d = obs_buf[:, 4]  # yaw angular velocity
    
    # Calculate distance to the target position
    distance_to_target = torch.sqrt((goal_pos[0] - x_d)**2 + (goal_pos[1] - y_d)**2)
    
    # Rewards and penalties
    distance_reward = -distance_to_target
    pitch_penalty = -torch.abs(pitch)
    pitch_d_penalty = -torch.abs(pitch_d)
    yaw_d_penalty = -torch.abs(yaw_d)
    
    # Temperature parameters for normalization
    temp_distance = torch.tensor(1.0, device=obs_buf.device)
    temp_pitch = torch.tensor(1.0, device=obs_buf.device)
    temp_pitch_d = torch.tensor(1.0, device=obs_buf.device)
    temp_yaw_d = torch.tensor(1.0, device=obs_buf.device)
    
    # Apply temperature normalization (optional)
    distance_reward = torch.exp(distance_reward / temp_distance)
    pitch_penalty = torch.exp(pitch_penalty / temp_pitch)
    pitch_d_penalty = torch.exp(pitch_d_penalty / temp_pitch_d)
    yaw_d_penalty = torch.exp(yaw_d_penalty / temp_yaw_d)
    
    # Total reward
    total_reward = distance_reward + pitch_penalty + pitch_d_penalty + yaw_d_penalty
    
    # Individual reward components for analysis
    reward_components = {
        "distance_reward": distance_reward,
        "pitch_penalty": pitch_penalty,
        "pitch_d_penalty": pitch_d_penalty,
        "yaw_d_penalty": yaw_d_penalty
    }
    
    return total_reward, reward_components
