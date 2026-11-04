"""Executable reference contract: reject silent changes to the baseline recipe."""
REFERENCE = {
    'net.backbone':'ataraxos', 'net.depth':8, 'net.embed_dim':384, 'net.n_head':8,
    'net.ff_factor':4, 'net.action_key_dim':384, 'net.pos_emb_std':.1,
    'net.dropout':0., 'net.use_global_summary':False, 'net.identity_encoding':'known_only',
    'net.use_cat_vf':True, 'env.num_envs':1536, 'env.steps_per_env':400,
    'env.max_num_moves':4000, 'env.use_gpu_rollout':True,
    'random_opponent':False, 'reward_shaping':False, 'belief.enabled':False,
    'ppo.advantage_normalization':'raw', 'ppo.return_perspective':'same_seat',
    'ppo.gamma':1., 'ppo.gae_lambda':.5, 'ppo.td_lambda':.8,
    'ppo.clip_range':.2, 'ppo.kl_coef':.1, 'ppo.vf_coef':1., 'ppo.policy_coef':1.,
    'ppo.adv_filt_rate':.25, 'ppo.adv_filt_thresh':.01, 'ppo.adv_filter_scope':'rollout',
    'ppo.value_sample_scope':'policy', 'ppo.minibatch_group':'timestep',
    'ppo.timestep_order':'shuffle', 'ppo.advantage_history_steps':4404,
    'ppo.ema_decay':.999, 'ppo.temperature_coef':.05, 'ppo.temperature_decay':.3,
    'ppo.temperature_floor':.001, 'ppo.temperature_ceil':.1,
    'ppo.magnet_shape':'piece_then_dest', 'ppo.num_epochs_per_rollout':1,
    'ppo.lr_coef':.5, 'ppo.lr_decay':1.1, 'ppo.lr_ceil':1e-4,'ppo.lr_floor':5e-6,
    'ppo.lr_schedule_unit':'rollout', 'ppo.temperature_schedule_unit':'rollout',
    'ppo.max_grad_norm':.267, 'ppo.weight_decay':0., 'ppo.dtype':'float32',
    'arr.enabled':True, 'disable_arr_train':False, 'arr.refresh_every':1,
    'arr.n_arr':4000, 'arr.pool_size':0, 'arr.storage_duration':11,
    'arr.independent_seat_resets':True, 'arr.reset_reward_window':True,
    'arr.net.depth':4, 'arr.net.embed_dim':512, 'arr.net.n_head':8,
    'arr.net.ff_factor':4,'arr.net.pos_emb_std':.1,'arr.net.seat_conditioning':False,
    'arr.net.use_cat_vf':True,'arr.net.dropout':0.,
    'arr.ppo.batch_size':1024,'arr.ppo.ema_decay':.999,'arr.ppo.lr':5e-5,
    'arr.ppo.num_epoch_per_train':5,'arr.ppo.clip_range':.2,'arr.ppo.kl_coef':.1,
    'arr.ppo.vf_coef':.5,'arr.ppo.policy_coef':1.,'arr.ppo.ent_pred_coef':1.,
    'arr.ppo.max_grad_norm':.5,'arr.ppo.weight_decay':0.,'arr.ppo.autocast_dtype':'float32',
    'arr.reg_norm':10.,'arr.reg_temp_init':.1,'arr.reg_temp_decay':.3,'arr.reg_temp_floor':.001,
}


def validate_reference(cfg):
    if cfg.reference_profile not in ('ataraxos_v100', 'ataraxos_v100_distributed'):
        raise ValueError('Unknown reference profile')
    errors=[]
    world = cfg.distributed_world_size if cfg.reference_profile.endswith('_distributed') else 1
    if world not in (1,2,4,8,16):
        raise ValueError('Unsupported reference world size')
    for key, expected in REFERENCE.items():
        if key in ('env.num_envs','arr.n_arr','arr.ppo.batch_size'):
            expected //= world
        value=cfg
        for part in key.split('.'):value=getattr(value,part)
        if value != expected:errors.append(f'{key}: expected {expected!r}, got {value!r}')
    if errors:raise ValueError('Ataraxos reference contract mismatch: '+ '; '.join(errors))
