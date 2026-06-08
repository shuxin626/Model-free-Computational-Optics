"""Parameters for the naive CGH task."""

settings = {
    'env_name': 'sim',
    'dataset_name': 'cifar',
    'slm_type': 'phase',
    'optimizer': 'mfo',
    'pg_type': 'loo',
}

holo_param = {
    'in_size': 128,
    'partition': 128,
    'sample_idx': 1,
    'macro_itr': 10000,
    'show_result': True,
    'show_interval': 100,
    'save_target': True,
}

actor_param = {
    'maskquery_batchsize': 128,
    'dp_std': 0.2,
    'pg_lr': 0.3,
    'use_scheduler': False,
    'optimizer_type': 'sgd',
    'output_normalize_flag': True,
}

optics_param = {
    'input_dx': 4.8,
    'output_dx': 4.8,
    'wave_lengths': 0.633,
    'z': 1e4,
    'response_type': None,
    'pad_scale': 2,
}
