import torch
import math
import numpy as np
from torch.utils.data.dataset import Dataset

class TurnToPhaseamp(object):

    def __call__(self, image):
        phase = image * math.pi
        phaseamp = torch.concat((phase, image), dim=0)
        return phaseamp


def split_train_and_val(index_for_each_type, val_num_per_type, shuffle_data=True):
    train_indices = []
    train_sampling_weight = []
    val_indices = []
    for i in range(len(index_for_each_type)):
        index_type_i = index_for_each_type[str(i)]
        if shuffle_data:
            np.random.shuffle(index_type_i)
        index_type_i_for_train = index_type_i[val_num_per_type:, 0]
        train_indices = np.concatenate((train_indices, index_type_i_for_train))
        train_sampling_weight = np.concatenate((train_sampling_weight, [1 / len(index_type_i_for_train)]*len(index_type_i_for_train)))
        val_indices = np.concatenate((val_indices, index_type_i[:val_num_per_type, 0]))
    train_indices = train_indices.astype(int)
    val_indices = val_indices.astype(int)
    return train_indices, train_sampling_weight, val_indices

def dataset_selector(image, target, type_idx_list, num_per_type, begin_sampling=None):
    chooser = np.array([0] * len(target))
    label = np.array([0] * len(target))
    if type_idx_list is not None:
        for i, type_number in enumerate(type_idx_list):
            idx_selector = np.argwhere(np.array(target) == type_number)
            if num_per_type is not None:
                if begin_sampling is None:
                    idx_selector = idx_selector[:num_per_type]
                else:
                    idx_selector = idx_selector[begin_sampling:int(begin_sampling + num_per_type)]
            chooser[idx_selector] = 1
            label[idx_selector] = i
        image = image[chooser > 0] 
        target = np.array(label)[chooser > 0]
    return image, target

    
def get_index_for_each_type(label_set):
    type_count = np.max(label_set) + 1
    index_for_each_type = {}
    for i in range(type_count):
        index_for_each_type[str(i)] = np.argwhere(label_set == i)
    return index_for_each_type


class CustomDataset(Dataset):
    def __init__(self, data, target, transforms=None):
        self.data = data
        self.target = target
        self.transforms = transforms

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        image = self.data[index, ...]
        target = self.target[index]
        if self.transforms is not None: image = self.transforms(image.numpy())
        return image, target

