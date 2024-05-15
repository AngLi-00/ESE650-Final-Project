@torch.jit.script
def compute_reward(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Unpack observations
    x_d = obs_buf[:, 0]  # x-directional velocity
    y_d = obs_buf[:, 1]  # y-directional velocity
    pitch = obs_buf[:, 2]  # pitch angle
    pitch_d = obs_buf[:, 3]  # pitch rate (d/dt of pitch)
    
    # Define temperature parameters for transformations
    temp_pitch = 0.1
    temp_velocity = 0.1
    temp_pitch_d = 0.1
    
    # Calculate individual reward components
    pitch_reward = torch.exp(-temp_pitch * torch.abs(pitch))
    velocity_penalty = torch.exp(-temp_velocity * (torch.abs(x_d) + torch.abs(y_d)))
    pitch_d_penalty = torch.exp(-temp_pitch_d * torch.abs(pitch_d))
    
    # Total reward is a combination of the above components
    total_reward = pitch_reward + velocity_penalty + pitch_d_penalty
    
    # Construct reward components dictionary
    reward_components = {
        "pitch_reward": pitch_reward,
        "velocity_penalty": velocity_penalty,
        "pitch_d_penalty": pitch_d_penalty
    }
    
    return total_reward, reward_components
