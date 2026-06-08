
from utils.visualize_utils import sgd_vis, plot_loss
from utils.checkpoint import CkptController
import torch
from trainer.base_trainer import BaseTrainer, create_summary_writer
from trainer.classification_utils import ClassificationStats, evaluate_classification_loader, prepare_classification_batch
import torch.nn as nn

class IdealClassificationTrainer(BaseTrainer):
    def __init__(self, model, settings, ideal_optimizer_param, train_param, optics_param):
        self.tb_writer = create_summary_writer()
        super(IdealClassificationTrainer, self).__init__(self.tb_writer)
        self.model = model

        optimizer_parameters = [{'params': self.model.phase_mask,
                                  'lr': ideal_optimizer_param['optics_lr'],
                                  'momentum': ideal_optimizer_param['momentum']}]

        if ideal_optimizer_param['optimizer_type'] == 'sgd':
            self.optimizer = torch.optim.SGD(optimizer_parameters)
        elif ideal_optimizer_param['optimizer_type'] == 'adam':
            self.optimizer = torch.optim.Adam(optimizer_parameters)
        else:
            raise Exception('Optimizer not found')

        self.criterion = nn.CrossEntropyLoss()
        self.train_param = train_param
        self.optics_param = optics_param
        self.settings = settings
        if self.train_param['checkpoint']['save_checkpoint']:
            self.ckpt_controller = CkptController(
                train_param, self.train_param['checkpoint']['clean_prev_ckpt_flag'],
                dir_name_suffix=self.train_param['checkpoint']['dir_name_suffix'])


    def train(self, epoch, train_loader, in_ch, number_of_type):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        stats = ClassificationStats()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = prepare_classification_batch(inputs, targets, in_ch)
            self.optimizer.zero_grad()

            outputs, cam_img = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
            self.optimizer.step()

            predicted = stats.update(outputs, targets, loss)

        sgd_vis(self.train_param, epoch, self.model, cam_img,
                targets, inputs, number_of_type, predicted)
        acc = stats.accuracy
        print("train acc of epoch at training stage: %3d is : %3.4f" %
              (epoch, acc))
        return acc

    def test(self, epoch, val_loader, in_ch, number_of_type, dataset='val'):
        self.model.eval()
        stats = evaluate_classification_loader(
            val_loader,
            in_ch,
            self.criterion,
            forward_batch=lambda batch_inputs: self.model(batch_inputs, if_test=True)[0],
        )
        acc = stats.accuracy
        avg_loss = stats.mean_loss_per_sample
        if dataset == 'train':
            print(
                "train set acc of epoch after training stage: %3d is : %3.4f" % (epoch, acc))
        elif dataset == 'val':
            print("val acc of epoch after training stage: %3d is : %3.4f" %
                  (epoch, acc))
        return avg_loss, acc

    def fit(self, number_of_type, in_ch, train_loader, val_loader):
        result_epoch = {}
        result_lst = {'train_loss': [], 'train_acc': [],
                      'val_loss': [], 'val_acc': []}
        best_result = {'train_loss': 1000, 'train_acc': 0,
                       'val_loss': 1000, 'val_acc': 0}
        epoch_lst = []
        early_stop_counter = 0

        for epoch in range(self.train_param['training_epochs']):
            epoch_lst.append(epoch)

            train_acc_avg = self.train(
                epoch, train_loader, in_ch, number_of_type)

            result_epoch['val_loss'], result_epoch['val_acc'] = self.test(
                epoch, val_loader, in_ch, number_of_type, 'val')

            result_epoch['train_loss'], result_epoch['train_acc'] = self.test(
                epoch, train_loader, in_ch, number_of_type, 'train')

            result_lst, best_result, early_stop_counter = self.update_result(epoch, result_epoch, result_lst, best_result,
                              early_stop_counter, self.train_param['early_stop_metrics'],
                              self.train_param['checkpoint']['save_checkpoint'], self.train_param['checkpoint']['metrics'])

            if early_stop_counter == self.train_param['early_stop_epochs']:
                break

            if (epoch % 10 == 0) and epoch > 1 or (epoch == self.train_param['training_epochs']-1):
                plot_loss(epoch_lst, result_lst['train_acc'], 'train_acc')
                plot_loss(epoch_lst, result_lst['train_loss'], 'train_loss')
                plot_loss(epoch_lst, result_lst['val_acc'], 'val_acc')
                plot_loss(epoch_lst, result_lst['val_loss'], 'val_loss')


                print("best train_acc" + " is : %3.5f" %(best_result['train_acc']))
                print("best train_loss" + " is : %3.5f" %(best_result['train_loss']))
                print("best val_acc" + " is : %3.5f" % (best_result['val_acc']))
                print("best val_loss" + " is : %3.5f" % (best_result['val_loss']))
            result_epoch = {}
        return result_lst['train_acc'], result_lst['val_acc']
