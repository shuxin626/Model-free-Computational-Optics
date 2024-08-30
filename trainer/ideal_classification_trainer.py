
from utils.visualize_utils import sgd_vis, plot_loss
from utils.general_utils import CkptController
import torch
import torch.nn
from trainer.basetrainer import BaseTrainer
from config import *
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class IdealClassificationTrainer(BaseTrainer):
    def __init__(self, model, settings, ideal_optimizer_param, train_param, optics_param):
        self.tb_writer = SummaryWriter()
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
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device)
            self.optimizer.zero_grad()

            outputs, cam_img = self.model(inputs[:, in_ch, (...)])

            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        sgd_vis(self.train_param, epoch, self.model, cam_img,
                targets, inputs, number_of_type, predicted)
        acc = 100.*correct/total
        print("train acc of epoch at training stage: %3d is : %3.4f" %
              (epoch, acc))
        return acc

    def test(self, epoch, val_loader, in_ch, number_of_type, dataset='val'):
        self.model.eval()
        acc_loss = 0
        acc_correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.float().to(device)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device)

                outputs, cam_image = self.model(inputs[:, in_ch, (...)], if_test=True)

                loss = self.criterion(outputs, targets)
                acc_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                acc_correct += predicted.eq(targets).sum().item()

                if batch_idx == 0:
                    predicted_all = predicted
                    targets_all = targets
                else:
                    predicted_all = torch.cat((predicted_all, predicted))
                    targets_all = torch.cat((targets_all, targets))

        acc = 100.*acc_correct/total
        avg_loss = acc_loss/total
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