@torch.jit.script
def compute_reward(self, obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the relevant components from the observations
    pitch = obs_buf[:, 2]
    pitch_rate = obs_buf[:, 3]
    
    # Define temperature parameters for reward shaping
    pitch_temp = torch.tensor(0.1, device=pitch.device)
    pitch_rate_temp = torch.tensor(0.1, device=pitch_rate.device)
    
    # Compute individual reward components
    pitch_reward = torch.exp(-pitch_temp * torch.abs(pitch))
    pitch_rate_reward = torch.exp(-pitch_rate_temp * torch.abs(pitch_rate))
    
    # Total reward is the sum of individual rewards
    total_reward = pitch_reward + pitch_rate_reward
    
    # Returning the total reward and the individual reward components
    reward_dict = {
        "pitch_reward": pitch_reward,
        "pitch_rate_reward": pitch_rate_reward
    }
    
    return total_reward, reward_dict
