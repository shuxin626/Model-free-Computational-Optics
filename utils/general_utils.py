import numpy as np
import os
import glob
import torch.nn as nn
import datetime
import torch.nn.functional as F
import math
from datetime import timedelta
from pynvml import * 
import torch
from config import *


class CkptController(object):
    def __init__(self, train_param, clean_prev_ckpt_flag=True, ckpt_dir=None, dir_name_suffix='') -> None:
        self.dir_name_suffix = dir_name_suffix
        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
        else:
            self.ckpt_dir = self.create_ckpt_dir_handle(train_param)
        if clean_prev_ckpt_flag:
            clean_pt_files_in_dir(self.ckpt_dir)
        print('ckpt dir is {}'.format(self.ckpt_dir))

    def create_ckpt_dir_handle(self, train_param):
        ckpt_dir = 'checkpoint/{}{}-trainnum-{}'.format(train_param['dataset_name'],
                                                        num_list_to_str(train_param['type_idx_list']) if train_param['type_idx_list'] != list(
                                                            range(10)) else 10,
                                                        str(train_param['num_per_type_train']))
        ckpt_dir = ckpt_dir + self.dir_name_suffix
        cond_mkdir(ckpt_dir)
        return ckpt_dir

    def save_ckpt(self, model, train_acc, val_acc, epoch, num_slm_layer, mask_for_prediction=None, test_acc=None, test_loss=None):
        if num_slm_layer > 1:
            state = {
                'net': [getattr(model.optics_sim, 'SLM{}'.format(i)) for i in range(num_slm_layer)],
                'val_acc': val_acc,
                'train_acc': train_acc,
                'epoch': epoch,
                'test_acc': test_acc,
                'test_loss': test_loss,
            }
        else:
            if mask_for_prediction is not None: # mfo
                if len(mask_for_prediction.size()) == 2:
                    mask_for_prediction = mask_for_prediction[None, None, ...]
                state = {
                        'net': mask_for_prediction,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'epoch': epoch,
                        'test_acc': test_acc,
                        'test_loss': test_loss,
                } 
            else: # ideal
                state = {
                    'net': model.phase_mask,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                }
        torch.save(state, os.path.join(self.ckpt_dir, '{}.pth'.format(epoch)))

    def load_ckpt(self, ckpt_num=None):
        if ckpt_num is None:
            filelist = glob.glob(os.path.join(self.ckpt_dir, "*.pth"))
            assert filelist != [], 'dir is empty and we need to create one'
            ckpt_dict = (torch.load(sort_file_by_digit_in_name(filelist)[-1], map_location=device))
        else:
            ckpt_dict = (torch.load(os.path.join(self.ckpt_dir, '{}.pth'.format(ckpt_num)), map_location=device))
        return ckpt_dict
        # raise NotImplementedError

def sort_file_by_digit_in_name(filelist, suffix='.pth'):
    end_index = len(suffix)
    file_name_list_int = [int(os.path.basename(file)[:-end_index]) for file in filelist]
    folder_name = os.path.dirname(filelist[0])
    sortted_file_name_list_int = sorted(file_name_list_int)
    sortted_file_name_list_str = [os.path.join(folder_name, '{}{}'.format(num ,suffix)) for num in sortted_file_name_list_int]
    return sortted_file_name_list_str

def num_list_to_str(num_list):
    str_list = [str(i) for i in num_list]
    return ''.join(str_list)

def clean_pt_files_in_dir(path_to_dir):
    # https://www.techiedelight.com/delete-all-files-directory-python/
    filelist = glob.glob(os.path.join(path_to_dir, "*.pth"))
    for f in filelist:
        os.remove(f)


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def shift_and_crop(img, two_d_shift, centersize):
    x_center = int(img.shape[-2] / 2)
    y_center = int(img.shape[-1] / 2)
    half_centersize = int(centersize/2)
    img_cropped = img[..., (x_center - half_centersize)+two_d_shift[-2]: (x_center + half_centersize)+two_d_shift[-2],
                      y_center - half_centersize+two_d_shift[-1]: y_center + half_centersize+two_d_shift[-1]]
    return img_cropped        


def central_crop(variable, tw=None, th=None, dim=2):
    
    if dim == 2:
        w = variable.shape[-2]
        h = variable.shape[-1]
        if w>= tw and h>= th:
            x1 = int(round((w - tw) / 2.0))
            y1 = int(math.ceil((h - th) / 2.0))
            cropped = variable[..., x1: x1 + tw, y1: y1 + th]
        else:
            raise NotImplementedError
    elif dim == 1:
        if h>=th:
            h = variable.shape[-1]
            y1 = int(round((h - th) / 2.0))
            cropped = variable[..., y1: y1 + th]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return cropped


class InterpolateComplex2d(nn.Module):
    def __init__(self, input_dx, input_field_shape, output_dx, output_field_shape=None, mode='bicubic', del_intermediate_var=False) -> None:
        super().__init__()
        self.mode = mode
        if output_dx == input_dx:
            pass
        self.input_pad_scale = self.get_input_pad_scale(
            input_dx, input_field_shape, output_dx, output_field_shape)
                
        self.interpolated_input_field_shape = [
            int(input_dx*side_length*self.input_pad_scale/output_dx) for side_length in input_field_shape[-2:]]

        self.output_field_shape = output_field_shape if output_field_shape is not None else self.interpolated_input_field_shape

        self.del_intermediate_var = del_intermediate_var

        self.scale_factor = input_dx/output_dx


    def get_input_pad_scale(self, input_dx, input_field_shape, output_dx, output_field_shape):
        if output_field_shape is None:
            input_pad_scale = 1
        else:
            if input_dx * input_field_shape[-2] <= output_dx * output_field_shape[-2]:
                input_pad_scale_x = (
                    output_dx * output_field_shape[-2]) / (input_dx * input_field_shape[-2])
            else:
                input_pad_scale_x = 1

            if input_dx * input_field_shape[-1] <= output_dx * output_field_shape[-1]:
                input_pad_scale_y = (
                    output_dx * output_field_shape[-1]) / (input_dx * input_field_shape[-1])
            else:
                input_pad_scale_y = 1
            input_pad_scale = max(input_pad_scale_y, input_pad_scale_x)
            
        return input_pad_scale
        

    def interp_complex(self, x):
        x_in_real_imag = torch.view_as_real(x)  # shape [..., w, h, 2]
        x_real_interpolated = F.interpolate(x_in_real_imag[..., 0], (self.interpolated_input_field_shape[-2], self.interpolated_input_field_shape[-1]), mode=self.mode, align_corners=False)
        
        x_imag_interpolated = F.interpolate(x_in_real_imag[..., 1], (
            self.interpolated_input_field_shape[-2], self.interpolated_input_field_shape[-1]), mode=self.mode, align_corners=False)
        
        x_interpolated = torch.stack(
            [x_real_interpolated, x_imag_interpolated], dim=-1)
        
        if self.del_intermediate_var:
            del x_real_interpolated
        if self.del_intermediate_var:
            del x_imag_interpolated
        x_interpolated = torch.view_as_complex(x_interpolated)
        return x_interpolated

    
    def circular_pad_or_crop(self, x):
        binary_ouputs = torch.tensor(x.shape[-2:]) < torch.tensor(self.output_field_shape)
        
        intermediate_size = binary_ouputs * (torch.tensor(self.output_field_shape) - torch.tensor(x.shape[-2:])) + torch.tensor(x.shape[-2:])

        
        x = circular_pad(x, w_padded=intermediate_size[-2].item(), h_padded=intermediate_size[-1].item())

        x = central_crop(x, tw=self.output_field_shape[-2], th=self.output_field_shape[-1])
        
        return x  
    
    def forward(self, x):
        x = circular_pad(x, pad_scale=self.input_pad_scale)
        
        x_interpolated = self.interp_complex(x)
        
        x = x_interpolated/(self.scale_factor)

              
        if self.del_intermediate_var:
            del x_interpolated
        # central crop to get the desired ouput shape
        if self.del_intermediate_var:
            pass 

        if torch.prod(torch.tensor(x.shape[-2:]) >= torch.tensor(self.output_field_shape)):
            output = central_crop(x,
                                tw=self.output_field_shape[-2], th=self.output_field_shape[-1])
        elif torch.prod(torch.tensor(x.shape[-2:]) < torch.tensor(self.output_field_shape)):
            output = circular_pad(x,
                                w_padded=self.output_field_shape[-2], h_padded=self.output_field_shape[-1])
        else:
            output = self.circular_pad_or_crop(x)
        if self.del_intermediate_var:
            del x
        return output

def pad_image(
    field, target_shape, pytorch=True, stacked_complex=False, padval=0, mode="constant"
):
    r"""Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [
                int(p)  # convert from np.int64
                for tple in zip(pad_front[::-1], pad_end[::-1])
                for p in tple
            ]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
            # raise NotImplementedError
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(
                field, tuple(zip(pad_front, pad_end)), mode, constant_values=padval
            )
    else:
        return field


def circular_pad(u,  w_padded=None, h_padded=None, pad_scale=None):
    """circular padding last two dimension of a tensor"""
    w, h = u.shape[-2], u.shape[-1]
    if pad_scale != None:
        w_padded, h_padded = w*pad_scale, h*pad_scale
    ww = int(round((w_padded - w) / 2.0))
    hh = int(math.ceil((h_padded - h) / 2.0))
    p2d = (hh, hh, ww, ww)
    u_padded = F.pad(u, p2d, mode="constant", value=0)
    return u_padded



def shift_list(num_class, shift):
    if num_class == 10:
        shift_list = [
            [-shift, -shift],
            [-shift, 0],
            [-shift, shift],
            [0, int(-1.5*shift)],
            [0, int(-0.5*shift)],
            [0, int(0.5*shift)],
            [0, int(1.5*shift)],
            [shift, -shift],
            [shift, 0],
            [shift, shift], ]
    elif num_class == 4:
        shift_list = [
            [-shift, -shift],
            [shift, shift],
            [-shift, shift],
            [shift, -shift],
        ]
    elif num_class == 2:
        shift_list = [
            [-shift, -shift],
            [shift, shift],
        ]
    return shift_list


def convert_tensor_to_cpu(input):
    if torch.is_tensor(input):
        if input.device.type != 'cpu':
            input = input.cpu().numpy()
    return input