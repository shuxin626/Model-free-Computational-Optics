import glob
import os

import torch

from utils.file_utils import clean_pt_files_in_dir, cond_mkdir, num_list_to_str, sort_file_by_digit_in_name
from utils.gpu_device_config import device


class CkptController(object):
    def __init__(self, train_param, clean_prev_ckpt_flag=True, ckpt_dir=None, dir_name_suffix="") -> None:
        self.dir_name_suffix = dir_name_suffix
        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
        else:
            self.ckpt_dir = self.create_ckpt_dir_handle(train_param)
        if clean_prev_ckpt_flag:
            clean_pt_files_in_dir(self.ckpt_dir)
        print("ckpt dir is {}".format(self.ckpt_dir))

    def create_ckpt_dir_handle(self, train_param):
        ckpt_dir = "checkpoint/{}{}-trainnum-{}".format(
            train_param["dataset_name"],
            num_list_to_str(train_param["type_idx_list"])
            if train_param["type_idx_list"] != list(range(10))
            else 10,
            str(train_param["num_per_type_train"]),
        )
        ckpt_dir = ckpt_dir + self.dir_name_suffix
        cond_mkdir(ckpt_dir)
        return ckpt_dir

    def save_ckpt(self, model, train_acc, val_acc, epoch, num_slm_layer, mask_for_prediction=None, test_acc=None, test_loss=None):
        if mask_for_prediction is not None:
            if len(mask_for_prediction.size()) in [2, 3]:
                mask_for_prediction = mask_for_prediction[None, None, ...]
            net = mask_for_prediction
        else:
            net = model.phase_mask

        state = {
            "net": net,
            "val_acc": val_acc,
            "train_acc": train_acc,
            "epoch": epoch,
            "test_acc": test_acc,
            "test_loss": test_loss,
        }
        torch.save(state, os.path.join(self.ckpt_dir, "{}.pth".format(epoch)))

    def load_ckpt(self, ckpt_num=None):
        if ckpt_num is None:
            filelist = glob.glob(os.path.join(self.ckpt_dir, "*.pth"))
            assert filelist != [], "dir is empty and we need to create one"
            return torch.load(sort_file_by_digit_in_name(filelist)[-1], map_location=device)
        return torch.load(os.path.join(self.ckpt_dir, "{}.pth".format(ckpt_num)), map_location=device)
