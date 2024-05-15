@torch.jit.script
def compute_reward(
    obs_buf: torch.Tensor, 
    target_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Initialize reward components
    x_d = obs_buf[:, 0]
    y_d = obs_buf[:, 1]
    pitch = obs_buf[:, 2]
    pitch_d = obs_buf[:, 3]
    yaw_d = obs_buf[:, 4]
    
    # Distance reward: encourage the agent to move towards the target position
    distance = torch.sqrt((x_d - target_pos[0])**2 + (y_d - target_pos[1])**2)
    distance_reward = -distance
    
    # Pitch penalty: penalize large pitch angles and oscillations
    pitch_penalty = torch.abs(pitch)
    
    # Pitch velocity penalty: penalize high pitch velocities to minimize oscillations
    pitch_velocity_penalty = torch.abs(pitch_d)
    
    # Yaw velocity penalty: penalize high yaw velocities to minimize oscillations
    yaw_velocity_penalty = torch.abs(yaw_d)
    
    # Combine the reward components
    reward = distance_reward - pitch_penalty - pitch_velocity_penalty - yaw_velocity_penalty
    
    reward_components = {
        "distance_reward": distance_reward,
        "pitch_penalty": pitch_penalty,
        "pitch_velocity_penalty": pitch_velocity_penalty,
        "yaw_velocity_penalty": yaw_velocity_penalty
    }

    return reward, reward_components
