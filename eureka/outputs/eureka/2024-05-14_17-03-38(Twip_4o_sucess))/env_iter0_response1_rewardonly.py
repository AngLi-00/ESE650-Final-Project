@torch.jit.script
def compute_reward(obs_buf: torch.Tensor, 
                   root_states: torch.Tensor, 
                   device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Rewards the agent for balancing the two-wheeled inverted pendulum and moving it to the target position,
    while minimizing pitch and yaw oscillations.
    """
    # Extracting observations
    x_d = obs_buf[:, 0]  # Velocity along x after rotating to body frame
    y_d = obs_buf[:, 1]  # Velocity along y after rotating to body frame
    pitch = obs_buf[:, 2]  # Pitch angle
    pitch_d = obs_buf[:, 3] # Pitch rate
    yaw_d = obs_buf[:, 4] # Yaw rate

    # Define the target position (we assume it is static and let's say it's origin for simplicity)
    target_pos = torch.tensor([0.0, 0.0], device=device)  # We choose (0,0) as the target for simplicity

    # Current position
    current_pos = root_states[:, :2]  # Assuming root_states holds [x, y, z, ...]

    # Compute distance to target
    distance_to_target = torch.norm(current_pos - target_pos, dim=1)
    
    # Penalties for oscillations
    pitch_penalty = torch.abs(pitch)
    pitch_rate_penalty = torch.abs(pitch_d)
    yaw_rate_penalty = torch.abs(yaw_d)
    
    # Reward component weights
    distance_to_target_weight = 1.0
    pitch_penalty_weight = 0.5
    pitch_rate_penalty_weight = 0.3
    yaw_rate_penalty_weight = 0.2
    
    # Normalizing the rewards
    # Add the temperature parameters to control the exponential transformation
    distance_to_target_temp = 1.0
    pitch_penalty_temp = 1.0
    pitch_rate_penalty_temp = 1.0
    yaw_rate_penalty_temp = 1.0

    distance_to_target_reward = torch.exp(-distance_to_target / distance_to_target_temp)
    pitch_penalty_reward = torch.exp(-pitch_penalty / pitch_penalty_temp)
    pitch_rate_penalty_reward = torch.exp(-pitch_rate_penalty / pitch_rate_penalty_temp)
    yaw_rate_penalty_reward = torch.exp(-yaw_rate_penalty / yaw_rate_penalty_temp)

    # Weighted sum of the rewards
    total_reward = (distance_to_target_weight * distance_to_target_reward +
                    pitch_penalty_weight * pitch_penalty_reward +
                    pitch_rate_penalty_weight * pitch_rate_penalty_reward +
                    yaw_rate_penalty_weight * yaw_rate_penalty_reward)

    # Dictionary of reward components
    reward_components = {
        "distance_to_target_reward": distance_to_target_reward,
        "pitch_penalty_reward": pitch_penalty_reward,
        "pitch_rate_penalty_reward": pitch_rate_penalty_reward,
        "yaw_rate_penalty_reward": yaw_rate_penalty_reward
    }

    return total_reward, reward_components
