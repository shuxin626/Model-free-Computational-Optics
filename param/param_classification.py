import copy

settings = {
    'sim_or_real': 'sim',
    'input_type': 'phase_only',
    'optimizer': 'mfo',  # {sbt, mfo, hbt}
    'train_or_test': 'train', # {train, test}
    'pg_type': 'loo',
}

train_param = {
    "dataset_name": "mnist", # from {mnist, fmnist}
    "type_idx_list": list(range(10)),
    "shuffle_data": False,
    "num_per_type_train": 100, 
    "num_per_type_val": 100,
    "num_per_type_test": 109, 
    'training_epochs': 150,
    'early_stop_epochs': 10,
    'early_stop_metrics': 'val_loss',
    'batch_size': 32,
    'subbatch_size': 1,  # only take effect for mfo
    'show_result': {
        'show_flag': True,  # either or not show some camera images during training
        'show_epoch_interval': 5,  # interval to show the camera img
        'show_settings': {
            'show_camimg': True,
            'show_mask': True,
            'show_rect': True,
            'num_per_class': 1,  # choose from int and None, if choose None, display the first one
            'save': False,
        },
    },
    'use_pbr_as_optical_weight': True,
    'optical_weight_shift': 70,
    'optical_weight_crop_size': 10,
    'checkpoint': {
        'save_checkpoint': True,
        # automatically generate 'dataset_name + type_idx_list + num_per_type_train + dir_name_suffix' to save checkpoints
        'dir_name_suffix': 'mfo',
        # from ['train_acc', 'train_loss', 'val_acc', 'val_loss']
        'metrics': 'val_loss',
        'clean_prev_ckpt_flag': True,
    }
}

test_param = {
    'dataset_for_test': ['test'],  # list ['train', 'val', 'test']
    'ckpt_dir': 'checkpoint/mnist10-trainnum-10sbt',
    'ckpt_num': None,
}

# mfo training params
actor_param = {
    'maskquery_batchsize': 128,
    'dp_std': 0.05, 
    'pg_lr': 20., 
    'use_scheduler': False,
    'optimizer_type': 'sgd',  # ['sgd', 'adam']
    'output_normalize_flag': False, 
}

# sbt and hbt training params
ideal_optimizer_param = {
    'optimizer_type': 'sgd',  # ['sgd', 'adam']
    'optics_lr': 2, # 2 for sgd, 0.01 for adam
    'momentum': 0.9,  # for sgd
}

exp_param = {}


# optical computing system
optics_param = {
    'general': {
        'pad_scale': 2,  # pad scale of the propogator; >2 for linear conv
        'response_type': None,
        'interp_sign': True,
    },
    'light_source': {
        'wave_length': 633.*1e-9,  # m Red laser
    },
    'input_layer': {
        'pixel_size': 8.0e-6,  # 8.0e-6 for pluto slm
        'effective_shape': [512, 512],
        'full_size': [1080, 1920],  # only take effect in real mode
        'type': 'slm',
        'tilt': 0.0, # degree
    },

    'optical_computing_layer': {
        'mask_representation': 'pixelwise',
        'mask_init_type': 'rand_init',  # {zero_init, rand_init}
        'mask_num_partitions': 128,  # if do not limit slm partitions, set None instead
        'effective_shape': [1280, 1280],
        'pixel_size': 3.74e-6,  # 3.74e-6, GAEA slm, if None, not take effect, equal to input field's pixel size
        'full_size': [2160, 3840],  # only take effect in real mode
        'misalignment': [0, 0],
    },
    'propogator.IC': {
        'length': 214.1e-3,
    },
    'propogator.CO': {
        'length': 202.5e-3,
    },
    'propogator.IO': {
        'length': 416.6e-3,
        },
    'camera': {
        'camera_mode': 'BaseAreaSensor',
        'effective_shape': [512, 512],  # if none, not take effect, equal to slm's effective size
        'full_size': [1024, 1280],  # only take effect in real mode
        "pixel_size": 4.8e-6,  # 4.8e-6, if none, not take effect, equal to slm's pixel size
        "gain": 0,
        "misalignment": [0, 0],
    }
}


# the dummy optical computing system, i.e., the system paramters we imagined.
optics_param_dummy = copy.deepcopy(optics_param)
optics_param['optical_computing_layer']['misalignment'] = [10, 0] # add misalignment in optics_param