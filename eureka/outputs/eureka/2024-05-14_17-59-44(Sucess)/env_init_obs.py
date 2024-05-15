class Twip(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)



        self.obs_buf[env_ids, 0] = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])[env_ids,0].squeeze() #x_d
        self.obs_buf[env_ids, 1] = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])[env_ids,1].squeeze() #y_d
        self.obs_buf[env_ids, 2] = get_euler_xyz(self.root_states[env_ids, 3:7])[1].squeeze() #pitch
        
        num_resets = len(env_ids)
        treshold = torch.tensor([2], device=self.device)
        pi_tensor = torch.tensor([6.28318], device=self.device)
        self.obs_buf[env_ids, 2] = torch.where(self.obs_buf[env_ids, 2] > treshold, self.obs_buf[env_ids, 2] - pi_tensor, self.obs_buf[env_ids, 2])
        self.obs_buf[env_ids, 2] = torch.where(self.obs_buf[env_ids, 2] < -treshold, self.obs_buf[env_ids, 2] + pi_tensor, self.obs_buf[env_ids, 2])
        self.obs_buf[env_ids, 3] = self.root_states[env_ids, 11].squeeze() #pitch_d
        self.obs_buf[env_ids, 4] = self.root_states[env_ids, 12].squeeze() #yaw_d




        
        return self.obs_buf
