from utils.visualize_utils import multi_show, plot_loss 
from utils.checkpoint import CkptController
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from trainer.base_trainer import BaseTrainer, create_summary_writer
from trainer.classification_utils import ClassificationStats, evaluate_classification_loader, prepare_classification_batch

class MFOClassificationTrainer(BaseTrainer):
    def __init__(self, model, settings, optimizer, actor_param, train_param, pg_type, optimizer_type='mfo', load_pretrained=False):
        self.tb_writer = create_summary_writer()
        super(MFOClassificationTrainer, self).__init__(self.tb_writer)
        self.optimizer = optimizer
        self.model = model
        self.settings = settings
        self.actor_param = actor_param
        self.train_param = train_param
        self.criterion_train = nn.CrossEntropyLoss(reduction='none')
        self.criterion_val = nn.CrossEntropyLoss()

        self.pg_type = pg_type
        self.optimizer_type = optimizer_type

        if self.train_param['checkpoint']['save_checkpoint']:
            self.ckpt_controller = CkptController(
                train_param, self.train_param['checkpoint']['clean_prev_ckpt_flag'],
                dir_name_suffix=self.train_param['checkpoint']['dir_name_suffix'])

    def perform_batch_eval(self, inputs, mask_for_prediction=None):
        """Run the optical model in subbatches and keep the first camera image for visualization."""
        outputs = []
        cam_img_first_subbatch = None
        subbatch_size = max(1, int(self.train_param['subbatch_size']))

        for start in range(0, inputs.shape[0], subbatch_size):
            outputs_subbatch, cam_img_subbatch = self.model(
                inputs[start:start + subbatch_size],
                exogenous_phase_mask=mask_for_prediction,
                if_test=False,
            )
            if cam_img_first_subbatch is None:
                cam_img_first_subbatch = cam_img_subbatch.clone()
            outputs.append(outputs_subbatch)
            del cam_img_subbatch

        outputs_accum = torch.cat(outputs, dim=0)
        return outputs_accum, cam_img_first_subbatch

    def cal_batch_train_loss(self, targets, outputs, number_of_type):
        targets_in_maskquerybatch = targets.unsqueeze(-1).repeat(
            1, self.actor_param['maskquery_batchsize'])

        targets_in_maskquerybatch_one_hot = F.one_hot(
            targets_in_maskquerybatch, num_classes=number_of_type)
        rewards = torch.sum(
            targets_in_maskquerybatch_one_hot * outputs, dim=-1)
        rewards = torch.mean(rewards, dim=0)
        loss = self.criterion_train(outputs.clone().permute(
            0, 2, 1), targets_in_maskquerybatch)  # loss shape [b, qb]
        # take mean along the batch dim, loss shape [qb]
        loss = torch.mean(loss, dim=0)
        return loss

    def train(self, epoch, number_of_type, in_ch, train_loader):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        stats = ClassificationStats()
        loss_cache = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            with torch.no_grad():
                inputs, targets = prepare_classification_batch(inputs, targets, in_ch)

                # inputs shape [b, ch, h, w]
                # outputs shape [b, qb, c]
                outputs, cam_img = self.perform_batch_eval(inputs)
                loss = self.cal_batch_train_loss(targets, outputs, number_of_type)

            # topest_ind is the ind of the best mask
            topest_mask_ind, _ = self.optimizer.step(-loss)

            mask_ind_for_prediction = topest_mask_ind
            mask_for_prediction = self.model.interp_phase_mask[0, mask_ind_for_prediction].clone().detach()
            loss_cache.append(loss[mask_ind_for_prediction].clone().cpu().detach().numpy())
            # calc correct
            stats.update(outputs[:, mask_ind_for_prediction, :], targets)

            if (batch_idx + 1) % 25 == 0 or batch_idx == (len(train_loader) - 1):
                print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    (batch_idx + 1) * len(inputs),
                    len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader),
                    loss[mask_ind_for_prediction].item())
                )


        if self.train_param['show_result']['show_flag'] == True and (epoch+1) % self.train_param['show_result']['show_epoch_interval'] == 0:
            multi_show(epoch, inputs[:, 0], mask_for_prediction,cam_img[:, mask_ind_for_prediction].detach(),
                       targets[:self.train_param['subbatch_size']], number_of_type, self.train_param['show_result']['show_settings'],
                       self.train_param['optical_weight_shift'], self.train_param['optical_weight_crop_size'],
                       cam_img.shape)

        acc = stats.accuracy
        print("avg train acc of epoch: %3d is : %3.4f" % (epoch, acc))

        return np.mean(loss_cache), mask_for_prediction, acc

    def test(self, epoch, mask_for_prediction, in_ch, test_loader, number_of_type, dataset='val'):
        # topest_mask_ind is the ind of the best mask in the batch of maskquery
        self.model.eval()
        stats = evaluate_classification_loader(
            test_loader,
            in_ch,
            self.criterion_val,
            forward_batch=lambda batch_inputs: self.perform_batch_eval(
                batch_inputs,
                mask_for_prediction,
            )[0][:, 0, :],
        )
        acc = stats.accuracy
        if dataset == 'train':
            print("train set acc of epoch : %3d is : %3.4f" % (epoch, acc))
        elif dataset == 'val':
            print("val set acc of epoch : %3d is : %3.4f" % (epoch, acc))

        return stats.loss_sum, acc

    def fit(self, number_of_type, in_ch, train_loader, val_loader):
        # Run training
        result_lst = {'train_loss': [], 'train_acc': [],
                      'val_loss': [], 'val_acc': []}
        best_result = {'train_loss': 1000, 'train_acc': 0,
                       'val_loss': 1000, 'val_acc': 0}
        result_epoch = {}
        epoch_lst = []
        early_stop_counter = 0

        for epoch in range(self.train_param['training_epochs']):
            epoch_lst.append(epoch)
            train_loss_avg, mask_for_prediction, train_acc_avg = self.train(
                epoch, number_of_type, in_ch, train_loader)

            result_epoch['val_loss'], result_epoch['val_acc'] = self.test(
                epoch, mask_for_prediction, in_ch, val_loader, number_of_type, 'val')
            result_epoch['train_loss'], result_epoch['train_acc'] = self.test(
                epoch, mask_for_prediction, in_ch, train_loader, number_of_type, 'train')

            result_lst, best_result, early_stop_counter = self.update_result(
                epoch, result_epoch, result_lst, best_result, early_stop_counter,
                self.train_param['early_stop_metrics'], self.train_param['checkpoint'][
                'save_checkpoint'], self.train_param['checkpoint']['metrics'],
                mask_for_prediction=mask_for_prediction)

            if early_stop_counter == self.train_param['early_stop_epochs']:
                break

            if (epoch % 10 == 0) and epoch > 1 or (epoch == self.train_param['training_epochs']-1):
                dir=self.ckpt_controller.ckpt_dir if self.train_param['checkpoint']['save_checkpoint'] else ''
                plot_loss(epoch_lst, result_lst['train_acc'], 'train_acc', newfig=True, dir=dir)
                plot_loss(
                    epoch_lst, result_lst['train_loss'], 'train_loss', newfig=True, dir=dir)
                plot_loss(epoch_lst, result_lst['val_acc'], 'val_acc', newfig=True, dir=dir)
                plot_loss(epoch_lst, result_lst['val_loss'], 'val_loss', newfig=True, dir=dir)

                print("best train acc is : %3.4f" % (best_result['train_acc']))
                print("best train_loss" + " is : %3.5f" % (best_result['train_loss']))
                print("best val acc is : %3.4f" % (best_result['val_acc']))
                print("best val_loss" + " is : %3.5f" % (best_result['val_loss']))
            result_epoch = {}
        return result_lst['train_acc'], result_lst['val_acc']
