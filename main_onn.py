#%%
from data.dataio import dataio
from model.opticalclassifier import OpticalClassifier
from actor.mfooptimizer import MFOOptimizer
from actor.pg import PG
from trainer.ideal_classification_trainer import IdealClassificationTrainer
from trainer.mfo_classification_trainer import MFOClassificationTrainer
from trainer.classification_tester import ClassificationTester
from param.param_onn import settings, train_param, test_param, actor_param, ideal_optimizer_param, optics_param, optics_param_dummy, exp_param
from config import *

def main_classification(settings, test_param, optics_param, train_param, ideal_optimizer_param, actor_param, exp_param, optics_param_dummy):
    print('-'*20 + 'optics_param' + '-'*20+'\n', optics_param)
    print('-'*20 + 'actor_param' + '-'*20+'\n', actor_param)
    print('-'*20 + 'train_param' + '-'*20+'\n', train_param)
    print('-'*20 + 'ideal_optimizer_param' + '-'*20+'\n', ideal_optimizer_param)
    print('-'*20 + 'exp_param' + '-'*20+'\n', exp_param)

    # load data for training
    train_loader, val_loader, test_loader, in_ch, number_of_type =\
        dataio(train_param['dataset_name'], settings['input_type'], optics_param['input_layer']['effective_shape'][0], optics_param['input_layer']['effective_shape'][1], train_param['batch_size'],
            train_param['type_idx_list'], num_per_type_train=train_param['num_per_type_train'], num_per_type_val=train_param['num_per_type_val'], num_per_type_test=train_param['num_per_type_test'],
            shuffle_data=train_param['shuffle_data'],)

    if settings['optimizer'] == 'mfo':
        if settings['pg_type'] == 'loo':
            actor = PG(
                    mask_shape=[optics_param['optical_computing_layer']['mask_num_partitions'], optics_param['optical_computing_layer']['mask_num_partitions']],
                    query_batchsize=actor_param['maskquery_batchsize'],
                    pg_lr=actor_param['pg_lr'],
                    dp_std=actor_param['dp_std'],
                    use_scheduler=actor_param['use_scheduler'],
                    optimizer_type=actor_param['optimizer_type'],
                    output_normalize_flag=actor_param['output_normalize_flag'],
                    )
    else:
        actor = None


    model = OpticalClassifier(settings['input_type'], optics_param, number_of_type, actor_param['maskquery_batchsize'],
                                optics_param['optical_computing_layer']['mask_num_partitions'], use_pbr_as_optical_weight=train_param[
                                    'use_pbr_as_optical_weight'], exp_param=exp_param, actor=actor,
                                shift=train_param['optical_weight_shift'], crop_size=train_param[
                                    'optical_weight_crop_size'], optimizer_type=settings['optimizer'],
                                pg_type=settings['pg_type'],
                                optics_param_dummy=optics_param_dummy)

    model.to(device)

    if settings['optimizer'] == 'mfo':
        optimizer = MFOOptimizer([model.phase_mask, model.phase_mask_clone], actor=actor, actor_param=actor_param)

    if settings['train_or_test'] == 'train':
        if settings['optimizer'] == 'mfo':
            trainer = MFOClassificationTrainer(
                model, settings, optimizer, actor_param, train_param, pg_type=settings['pg_type'])
        elif settings['optimizer'] == 'sbt' or 'hbt':
            trainer = IdealClassificationTrainer(
                model, settings, ideal_optimizer_param, train_param, optics_param)
        train_acc_lst, val_acc_lst = trainer.fit(number_of_type, in_ch, train_loader, val_loader)
        return train_acc_lst, val_acc_lst
    else:
        tester = ClassificationTester(
            model, test_param['ckpt_dir'], test_param['ckpt_num'], test_param['dataset_for_test'], train_param, settings=settings)
        test_result = tester.fit(number_of_type, in_ch, train_loader,
                val_loader, test_loader)
        return test_result
    

if __name__ == "__main__":
    main_classification(settings, test_param, optics_param, train_param, ideal_optimizer_param, actor_param, exp_param, optics_param_dummy)