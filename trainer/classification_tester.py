import torch
import torch.nn as nn

from trainer.classification_utils import evaluate_classification_loader
from utils.checkpoint import CkptController
from utils.gpu_device_config import device


class ClassificationTester(object):

    def __init__(self, model, ckpt_dir, ckpt_num, dataset_for_test, train_param, settings):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        ckpt_controller = CkptController(None, clean_prev_ckpt_flag=False, ckpt_dir=ckpt_dir)
        self.ckpt_state = ckpt_controller.load_ckpt(ckpt_num)


        self.model.phase_mask.data = torch.as_tensor(self.ckpt_state['net'], device=device)
        modulator_phasemask = self.model.phase_mask.data
        self.modulator_phasemask = modulator_phasemask[0, 0]
        self.dataset_for_test = dataset_for_test
        self.train_param = train_param
        
        print('loaded model has train accuracy {}'.format(self.ckpt_state['train_acc']))
        print('loaded model has val accuracy {}'.format(self.ckpt_state['val_acc']))

    def test(self, in_ch, dataloader, number_of_type):
        # topest_mask_ind is the ind of the best mask in the batch of maskquery
        self.model.eval()
        stats = evaluate_classification_loader(
            dataloader,
            in_ch,
            self.criterion,
            forward_batch=lambda batch_inputs: self.model(
                batch_inputs,
                self.modulator_phasemask,
                if_test=True,
            )[0],
            log_interval=5,
        )
        return stats.accuracy, stats.loss_sum

    def fit(self, number_of_type, in_ch, train_loader, val_loader, test_loader):
        result = {}
        
        if 'train' in self.dataset_for_test: 
            print('test train dataset')
            train_acc, train_loss = self.test(in_ch, train_loader, number_of_type)
            print("exp train acc is : %3.4f, train loss is : %3.4f" % (train_acc, train_loss)) 
            result['train_acc'] = train_acc
            result['train_loss'] = train_loss
        if 'val' in self.dataset_for_test: 
            print('test val dataset')
            val_acc, val_loss = self.test(in_ch, val_loader, number_of_type)
            print("exp val acc is : %3.4f, val loss is : %3.4f" % (val_acc, val_loss))
            result['val_acc'] = val_acc
            result['val_loss'] = val_loss
        if 'test' in self.dataset_for_test:
            print('test test dataset')
            test_acc, test_loss = self.test(in_ch, test_loader,number_of_type)
            print("exp test acc is : %3.4f, test loss is : %3.4f" % (test_acc, test_loss))
            result['test_acc'] = test_acc
            result['test_loss'] = test_loss
        
        return result
