from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_io import load_data
from env.classification_env_sim import OpticalClassifier
from agent.actor.mfo_optimizer import MFOOptimizer
from agent.actor.pg import PG
from trainer.ideal_classification_trainer import IdealClassificationTrainer
from trainer.mfo_classification_trainer import MFOClassificationTrainer
from trainer.classification_tester import ClassificationTester
from tasks.param.param_onn import settings, train_param, test_param, actor_param, ideal_optimizer_param, optics_param, optics_param_dummy, exp_param
from utils.gpu_device_config import device


def print_run_config(optics_param, actor_param, train_param, ideal_optimizer_param, exp_param):
    for name, value in [
        ('optics_param', optics_param),
        ('actor_param', actor_param),
        ('train_param', train_param),
        ('ideal_optimizer_param', ideal_optimizer_param),
        ('exp_param', exp_param),
    ]:
        print('-' * 20 + name + '-' * 20 + '\n', value)


def load_classification_data(settings, train_param, optics_param):
    return load_data(
        train_param['dataset_name'],
        settings['input_type'],
        optics_param['input_layer']['effective_shape'][0],
        optics_param['input_layer']['effective_shape'][1],
        train_param['batch_size'],
        train_param['type_idx_list'],
        num_per_type_train=train_param['num_per_type_train'],
        num_per_type_val=train_param['num_per_type_val'],
        num_per_type_test=train_param['num_per_type_test'],
        shuffle_data=train_param['shuffle_data'],
    )


def build_actor(settings, actor_param, optics_param):
    if settings['optimizer'] != 'mfo':
        return None
    if settings['pg_type'] != 'loo':
        raise NotImplementedError("Only the naive 'loo' policy-gradient actor is supported.")

    num_optical_computing_layers = settings.get('num_optical_computing_layers', 1)
    mask_shape = [
        optics_param['optical_computing_layer']['mask_num_partitions'],
        optics_param['optical_computing_layer']['mask_num_partitions'],
    ]
    if num_optical_computing_layers > 1:
        mask_shape.append(num_optical_computing_layers)

    return PG(
        mask_shape=mask_shape,
        query_batchsize=actor_param['maskquery_batchsize'],
        pg_lr=actor_param['pg_lr'],
        dp_std=actor_param['dp_std'],
        use_scheduler=actor_param['use_scheduler'],
        optimizer_type=actor_param['optimizer_type'],
        output_normalize_flag=actor_param['output_normalize_flag'],
    )


def build_model(settings, optics_param, train_param, actor_param, exp_param, optics_param_dummy, actor, number_of_type):
    model = OpticalClassifier(
        settings['input_type'],
        optics_param,
        number_of_type,
        actor_param['maskquery_batchsize'],
        optics_param['optical_computing_layer']['mask_num_partitions'],
        use_pbr_as_optical_weight=train_param['use_pbr_as_optical_weight'],
        exp_param=exp_param,
        actor=actor,
        shift=train_param['optical_weight_shift'],
        crop_size=train_param['optical_weight_crop_size'],
        optimizer_type=settings['optimizer'],
        pg_type=settings['pg_type'],
        optics_param_dummy=optics_param_dummy,
        num_optical_computing_layers=settings.get('num_optical_computing_layers', 1),
    )
    return model.to(device)


def build_trainer(model, settings, optimizer, actor_param, train_param, ideal_optimizer_param, optics_param):
    if settings['train_or_test'] == 'train':
        if settings['optimizer'] == 'mfo':
            return MFOClassificationTrainer(
                model, settings, optimizer, actor_param, train_param, pg_type=settings['pg_type'])
        if settings['optimizer'] in ['sbt', 'hbt']:
            return IdealClassificationTrainer(
                model, settings, ideal_optimizer_param, train_param, optics_param)
        raise NotImplementedError("Unsupported optimizer: {}".format(settings['optimizer']))
    return None


def main_classification(settings, test_param, optics_param, train_param, ideal_optimizer_param, actor_param, exp_param, optics_param_dummy):
    print_run_config(optics_param, actor_param, train_param, ideal_optimizer_param, exp_param)
    train_loader, val_loader, test_loader, in_ch, number_of_type = load_classification_data(
        settings,
        train_param,
        optics_param,
    )

    actor = build_actor(settings, actor_param, optics_param)
    model = build_model(
        settings,
        optics_param,
        train_param,
        actor_param,
        exp_param,
        optics_param_dummy,
        actor,
        number_of_type,
    )
    optimizer = None
    if settings['optimizer'] == 'mfo':
        optimizer = MFOOptimizer([model.phase_mask, model.phase_mask_clone], actor=actor, actor_param=actor_param)

    trainer = build_trainer(
        model,
        settings,
        optimizer,
        actor_param,
        train_param,
        ideal_optimizer_param,
        optics_param,
    )

    if trainer is not None:
        train_acc_lst, val_acc_lst = trainer.fit(number_of_type, in_ch, train_loader, val_loader)
        return train_acc_lst, val_acc_lst

    tester = ClassificationTester(
        model,
        test_param['ckpt_dir'],
        test_param['ckpt_num'],
        test_param['dataset_for_test'],
        train_param,
        settings=settings,
    )
    return tester.fit(number_of_type, in_ch, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main_classification(settings, test_param, optics_param, train_param, ideal_optimizer_param, actor_param, exp_param, optics_param_dummy)
