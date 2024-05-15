@torch.jit.script
def compute_reward(
    obs_buf: torch.Tensor,
    root_states: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Extract relevant information from observations
    x_d = obs_buf[:, 0]               # Distance in x direction to target
    y_d = obs_buf[:, 1]               # Distance in y direction to target
    pitch = obs_buf[:, 2]             # Pitch angle
    pitch_d = obs_buf[:, 3]           # Pitch rate
    yaw_d = obs_buf[:, 4]             # Yaw rate
    
    # Define positive and negative weights for different objectives
    position_reward_weight = 1.0
    balance_reward_weight = 1.0
    control_effort_weight = 0.1
    
    # Compute distance to target positions reward (Euclidean distance)
    distance_to_target = torch.sqrt(x_d**2 + y_d**2)
    target_position_reward = - distance_to_target * position_reward_weight
    
    # Compute balance-related rewards (keeping pitch and yaw small)
    balance_penalty = (torch.abs(pitch) + torch.abs(pitch_d) + torch.abs(yaw_d)) * balance_reward_weight
    
    # Add small penalty for control effort (to encourage stability)
    control_effort_penalty = (torch.abs(pitch_d) + torch.abs(yaw_d)) * control_effort_weight
    
    # Combine the rewards and penalties
    total_reward = target_position_reward - balance_penalty - control_effort_penalty
    
    # Optionally transform the reward component for better convergence (normalize via exponential)
    temperature = 1.0  # You can tune this parameter
    transformed_total_reward = torch.exp(total_reward / temperature)
    
    reward_components = {
        "target_position_reward": target_position_reward,
        "balance_penalty": balance_penalty,
        "control_effort_penalty": control_effort_penalty,
        "transformed_total_reward": transformed_total_reward
    }
    
    return transformed_total_reward, reward_components
