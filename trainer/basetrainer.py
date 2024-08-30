
class BaseTrainer(object):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    def update_result(self, epoch, result_epoch, result_lst, best_result, early_stop_counter,
                      early_stop_metrics, save_ckpt, ckpt_metrics, mask_for_prediction=None, 
                      key_list=['train_loss', 'train_acc', 'val_loss', 'val_acc']):

        for key in key_list:
            result_lst[key].append(result_epoch[key])

            if 'loss' in key and result_epoch[key] < best_result[key]:
                best_flag = True
            elif 'loss' not in key and result_epoch[key] > best_result[key]:
                best_flag = True
            else:
                best_flag = False

            if best_flag:
                best_result[key] = result_epoch[key]

                if save_ckpt and key == ckpt_metrics:
                    self.ckpt_controller.save_ckpt(self.model, result_epoch['train_acc'], result_epoch['val_acc'],
                                                   epoch, num_slm_layer=1, mask_for_prediction=mask_for_prediction)
                    print('saving ckpt at epoch {} according to the {}'.format(epoch, self.train_param['checkpoint']['metrics']))

                if key == early_stop_metrics:
                    early_stop_counter = 0
            elif key == early_stop_metrics and not best_flag:
                early_stop_counter += 1

        print('{} is {}, early_stop_counter {}'.format(early_stop_metrics, result_epoch[early_stop_metrics], early_stop_counter))

        for key in key_list:
            self.tb_writer.add_scalar(key, result_epoch[key], epoch)
        self.tb_writer.flush()
          
        return result_lst, best_result, early_stop_counter

