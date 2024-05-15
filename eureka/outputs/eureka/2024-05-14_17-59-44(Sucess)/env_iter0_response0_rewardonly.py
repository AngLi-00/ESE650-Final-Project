@torch.jit.script
def compute_reward(
    obs_buf: torch.Tensor, 
    root_states: torch.Tensor, 
    target_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    pos_error_coeff = torch.tensor([0.5], device=obs_buf.device)
    pitch_error_coeff = torch.tensor([1.0], device=obs_buf.device)
    yaw_d_error_coeff = torch.tensor([0.5], device=obs_buf.device)
    pitch_d_error_coeff = torch.tensor([0.5], device=obs_buf.device)
    
    pos_error_temp = torch.tensor([0.2], device=obs_buf.device)
    pitch_error_temp = torch.tensor([0.5], device=obs_buf.device)
    yaw_d_error_temp = torch.tensor([0.2], device=obs_buf.device)
    pitch_d_error_temp = torch.tensor([0.2], device=obs_buf.device)

    x_d = obs_buf[:, 0]
    y_d = obs_buf[:, 1]
    pitch = obs_buf[:, 2]
    pitch_d = obs_buf[:, 3]
    yaw_d = obs_buf[:, 4]
    
    target_x = target_pos[:, 0]
    target_y = target_pos[:, 1]
    
    pos_error = torch.sqrt((x_d - target_x) ** 2 + (y_d - target_y) ** 2)
    pitch_error = torch.abs(pitch)
    pitch_d_error = torch.abs(pitch_d)
    yaw_d_error = torch.abs(yaw_d)
    
    pos_error = torch.exp(-pos_error / pos_error_temp) * pos_error_coeff
    pitch_error = torch.exp(-pitch_error / pitch_error_temp) * pitch_error_coeff
    pitch_d_error = torch.exp(-pitch_d_error / pitch_d_error_temp) * pitch_d_error_coeff
    yaw_d_error = torch.exp(-yaw_d_error / yaw_d_error_temp) * yaw_d_error_coeff
    
    total_reward = pos_error + pitch_error + pitch_d_error + yaw_d_error
    
    reward_components = {
        'pos_error': pos_error,
        'pitch_error': pitch_error,
        'pitch_d_error': pitch_d_error,
        'yaw_d_error': yaw_d_error
    }
    
    return total_reward, reward_components
