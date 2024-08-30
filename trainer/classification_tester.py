from config import *
from utils.general_utils import CkptController
import torch.nn as nn



class ClassificationTester(object):

    def __init__(self, model, ckpt_dir, ckpt_num, dataset_for_test, train_param, settings):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        ckpt_controller = CkptController(None, clean_prev_ckpt_flag=False, ckpt_dir=ckpt_dir)
        self.ckpt_state = ckpt_controller.load_ckpt(ckpt_num)


        self.model.phase_mask.data = torch.tensor(self.ckpt_state['net']).to(device)
        modulator_phasemask = self.model.phase_mask.data
        self.modulator_phasemask = modulator_phasemask[0, 0]
        self.dataset_for_test = dataset_for_test
        self.train_param = train_param
        
        print('loaded model has train accuracy {}'.format(self.ckpt_state['train_acc']))
        print('loaded model has val accuracy {}'.format(self.ckpt_state['val_acc']))

    def test(self, in_ch, dataloader, number_of_type):
        # topest_mask_ind is the ind of the best mask in the batch of maskquery
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        
        with torch.no_grad():
            for iter_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.float().to(device)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device)
                
                outputs, _ = self.model(inputs[:, in_ch,(...)], self.modulator_phasemask, if_test=True)

                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if (iter_idx + 1) % 5 == 0:
                    print('[{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                        (iter_idx + 1)* len(inputs),
                        len(dataloader.dataset),
                        100. * iter_idx / len(dataloader),
                        loss.item())
                    )

        acc = 100.*correct/total
        return acc, test_loss

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