@torch.jit.script
def compute_reward(pitch: torch.Tensor, pitch_d: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameter for pitch
    pitch_temp = 10.0
    # Temperature parameter for pitch velocity
    pitch_d_temp = 5.0
   
    # Reward components for pitch and pitch velocity deviations
    pitch_reward = torch.exp(-pitch_temp * torch.abs(pitch))
    pitch_d_reward = torch.exp(-pitch_d_temp * torch.abs(pitch_d))
  
    # Summing up the components to get the total reward
    total_reward = pitch_reward + pitch_d_reward

    # Returning the total reward and the individual components as a dictionary
    return total_reward, {
        "pitch_reward": pitch_reward,
        "pitch_d_reward": pitch_d_reward
    }
