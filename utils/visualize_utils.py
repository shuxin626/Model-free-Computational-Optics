

"""_utils functions for visualization purposes_
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.general_utils import convert_tensor_to_cpu, shift_list
from sklearn.metrics import confusion_matrix
import os

def calculate_rect_params(img_shape, centersize, two_d_shift):
    img_shape = list(img_shape)
    rect_left_bottom_pos = [img_shape[-1] / 2 + two_d_shift[-1] - centersize / 2,
                            img_shape[-2] / 2 + two_d_shift[-2] - centersize / 2]
    return [tuple(rect_left_bottom_pos), centersize, centersize]

def plot_loss(iter, loss, filename="loss", label=None, newfig=False, color="b", dir=''):
    plt.figure(1)
    plt.clf()
    plt.title(filename)
    plt.xlabel("epoch")
    _ = plt.plot(iter, loss)
    if newfig:
        if filename is not None:
            plt.savefig(os.path.join(dir, filename) + ".png",
                        dpi=200, bbox_inches="tight")
    plt.draw()
    plt.show()

def draw_subplot_with_rect(ax, img, rect_list=None, cmap='gray', clim=None, target=None, show_colorbar=True):
    gca = ax.imshow(img, cmap=cmap, clim=clim, aspect='auto')
    if show_colorbar: plt.colorbar(gca, ax=ax)
    if rect_list is not None:
        import matplotlib.patches as patches
        # rect params contain [(x, y), width, height]
        for cls, rect in enumerate(rect_list):
            if cls == target:
                rect = patches.Rectangle(
                    rect[0], rect[1], rect[2],
                    linewidth=0.8, edgecolor='w', facecolor='none')
            else:
                rect = patches.Rectangle(
                    rect[0], rect[1], rect[2],
                    linewidth=0.8, edgecolor='g', ls='--', facecolor='none')

            ax.add_patch(rect)
    
    return gca

def multi_show(epoch, inputs, phase_mask, cam_img, targets, num_classes, display_settings, optical_weight_shift=None, optical_weight_crop_size=None, cam_img_shape=None, pred=None):
    '''
    phase_mask shape [height, width]
    target shape [batchsize]
    inputs shape [batchsize, height, width]
    cam_img shape [batchsize, height, width]
    '''
    # convert tensor to cpu
    inputs = convert_tensor_to_cpu(inputs)
    if phase_mask is not None:
        phase_mask = convert_tensor_to_cpu(phase_mask)
    cam_img = convert_tensor_to_cpu(cam_img)
    targets = convert_tensor_to_cpu(targets)

    # show phase mask
    if display_settings['show_mask'] and phase_mask is not None:
        show(phase_mask, 'modulate masks {}'.format(epoch), save=False)
        
    # select candidate images to show for each type
    list_of_ind_list_of_class = []  # [[type1_ind1,type1_ind2],[type2_ind1, type2_ind2]]
    row_count = 0  # in case batch contains no objects belong to certain type
    label_in_batch = []
    for cls in range(num_classes):
        ind_list_of_class = np.argwhere(targets == cls)
        obj_num_of_class = len(ind_list_of_class)
        num_to_select = display_settings['num_per_class'] if display_settings[
            'num_per_class'] < obj_num_of_class else obj_num_of_class
        if num_to_select != 0:
            list_of_ind_list_of_class.append(
                ind_list_of_class[:num_to_select, 0])
            row_count = row_count + 1
            label_in_batch.append(cls)

    # Draw subplots
    if display_settings['show_camimg']:
        ''' display order
                  col1           col2          col3          col4
            row1  class1_input1 class1_camimg1 class1_input2 class1_camimg2
            row2  class2_input1 class2_camimg1 class2_input2 class2_camimg2
        '''
        if display_settings['show_rect']:
            assert (optical_weight_crop_size is not None) and (
                optical_weight_shift is not None) and (cam_img_shape is not None)
            rect_list_all_type = [calculate_rect_params(
                cam_img_shape, optical_weight_crop_size, shift) for shift in shift_list(num_classes, optical_weight_shift)]
        else:
            rect_list_all_type = [None, None, None, None]
        col_count = display_settings['num_per_class'] * 2
        fig = plt.figure(figsize=(col_count*5, row_count*5))
        plt.suptitle('cam_img at epoch {}'.format(epoch))
        for row, ind_list in enumerate(list_of_ind_list_of_class):
            for col, ind in enumerate(ind_list):
                ax_input = fig.add_subplot(
                    row_count, col_count, row*col_count+col*2+1)
                draw_subplot_with_rect(ax_input, inputs[ind])
                ax_camimg = fig.add_subplot(
                    row_count, col_count, row*col_count+col*2+2)
                draw_subplot_with_rect(
                    ax_camimg, cam_img[ind], rect_list=rect_list_all_type[:num_classes], target=label_in_batch[row])
                # print('row {}, col {} is precited as {}'.format(row, col, pred[ind]))
        if display_settings['save'] is True:
            plt.savefig('imgs/cam_img.png')
        plt.show()




def show(input, title="image", cut=False, cmap='gray',
         clim=None, rect_list=None, hist=False, save=False,
         save_name='picture', log_scale=False, v_max=1.):

    if log_scale:
        if torch.is_tensor(input):
            input = torch.log(input)
        else:
            input = np.log(input)

    if torch.is_tensor(input):
        if input.device.type != 'cpu':
            print('detect the cuda')
            input = input.cpu()

    if hist:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    ax.title.set_text(title)
    if cut:
        img = ax.imshow(input, cmap=cmap, vmin=0, vmax=v_max)
    else:
        img = ax.imshow(input, cmap=cmap)
    plt.colorbar(img, ax=ax)

    if rect_list is not None:
        # rect_params contain [(x, y), width, height]
        rect = patches.Rectangle(
            rect_list[0], rect_list[1], rect_list[2],
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if hist:
        ax_ = fig.add_subplot(122)
        n, bins, patches = ax_.hist(input.flatten(), 100)
        ax_.title.set_text(title)

    if save is True:
        plt.savefig(save_name + '.png')
    plt.xlabel('x')
    plt.show()


def sgd_vis(train_param, epoch, model, cam_img, targets, inputs, number_of_type, pred):
    if train_param['show_result']['show_flag'] == True and (epoch+1)%train_param['show_result']['show_epoch_interval'] == 0:
        if hasattr(model, 'interp_phase_mask'):
            multi_show(epoch, inputs[:,0], model.interp_phase_mask[0,0].clone().detach(),
                    cam_img[:,0].detach(), targets, number_of_type, train_param['show_result']['show_settings'],
                    train_param['optical_weight_shift'], train_param['optical_weight_crop_size'],
                    cam_img.shape, pred) # TODO input only show phase channel right now
        else:
            multi_show(epoch, inputs[:,0], None, 
                    cam_img[:,0].detach(), targets, number_of_type, train_param['show_result']['show_settings'],
                    train_param['optical_weight_shift'], train_param['optical_weight_crop_size'],
                    cam_img.shape, pred) # TODO input only show phase channel right now




        
    
