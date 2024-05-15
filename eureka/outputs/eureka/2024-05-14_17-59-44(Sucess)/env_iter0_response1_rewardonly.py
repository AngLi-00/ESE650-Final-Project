@torch.jit.script
def compute_reward(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extracting the observations
    x_d = obs_buf[:, 0]
    y_d = obs_buf[:, 1]
    pitch = obs_buf[:, 2]
    pitch_d = obs_buf[:, 3]
    yaw_d = obs_buf[:, 4]

    # Reward for minimizing the distance to the target position
    target_distance = torch.sqrt(x_d ** 2 + y_d ** 2)
    target_distance_reward = -target_distance

    # Reward for minimizing pitch
    pitch_temperature = 10.0
    pitch_reward = torch.exp(-pitch_temperature * torch.abs(pitch))

    # Reward for minimizing pitch angular velocity
    pitch_d_temperature = 5.0
    pitch_d_reward = torch.exp(-pitch_d_temperature * torch.abs(pitch_d))

    # Reward for minimizing yaw angular velocity
    yaw_d_temperature = 5.0
    yaw_d_reward = torch.exp(-yaw_d_temperature * torch.abs(yaw_d))

    # Combine the rewards
    total_reward = target_distance_reward + pitch_reward + pitch_d_reward + yaw_d_reward

    # Creating the reward components dictionary
    reward_components = {
        "target_distance_reward": target_distance_reward,
        "pitch_reward": pitch_reward,
        "pitch_d_reward": pitch_d_reward,
        "yaw_d_reward": yaw_d_reward
    }

    return total_reward, reward_components
