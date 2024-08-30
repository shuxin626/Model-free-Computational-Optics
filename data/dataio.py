import torch
from torchvision import transforms
from data.data_utils import TurnToPhaseamp, split_train_and_val, CustomDataset
from data.public_dataset import Phaseamp_CIFAR10, Phaseamp_FMNIST, Phaseamp_MNIST
import math
import numpy as np
np.random.seed(12345)
torch.manual_seed(12345)


def dataio(dataset_name, input_type, obj_height, obj_width=300, batch_size=16, type_idx_list=None, num_per_type_train=None,
           num_per_type_test=None, num_per_type_val=None, shuffle_data=True,):

    transform_list = [transforms.ToTensor(), transforms.Resize([obj_height, obj_width]),
                      TurnToPhaseamp()]


    if dataset_name == "fmnist":
        dataset_train_val = Phaseamp_FMNIST(train=True, type_idx_list=type_idx_list, num_per_type=int(num_per_type_train + num_per_type_val))
        dataset_test = Phaseamp_FMNIST(train=False, type_idx_list=type_idx_list, num_per_type=num_per_type_test)        
        number_of_type = dataset_test.number_of_type

    elif dataset_name == "mnist":
        dataset_train_val = Phaseamp_MNIST(train=True, type_idx_list=type_idx_list, num_per_type=int(num_per_type_train + num_per_type_val))
        dataset_test = Phaseamp_MNIST(train=False, type_idx_list=type_idx_list, num_per_type=num_per_type_test)
        number_of_type = dataset_test.number_of_type

    elif dataset_name == 'cifar10':
        transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
        dataset_train_val = Phaseamp_CIFAR10(train=True, type_idx_list=type_idx_list, num_per_type=int(num_per_type_train + num_per_type_val))
        dataset_test = Phaseamp_CIFAR10(train=False, type_idx_list=type_idx_list, num_per_type=num_per_type_test)
        number_of_type = dataset_test.number_of_type

    else:
        raise Exception

    train_indices, train_sampling_weight, val_indices = split_train_and_val(
        dataset_train_val.index_for_each_type, num_per_type_val, shuffle_data=shuffle_data)

    dataset_train = CustomDataset(
        dataset_train_val.data[train_indices, ...], dataset_train_val.targets[train_indices], transforms=transforms.Compose(transform_list))
    dataset_val = CustomDataset(
        dataset_train_val.data[val_indices, ...], dataset_train_val.targets[val_indices], transforms=transforms.Compose(transform_list))
    dataset_test = CustomDataset(
        dataset_test.data, dataset_test.targets, transforms=transforms.Compose(transform_list))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False)

    print('train length is {}'.format(len(dataset_train)))
    print('val length is {}'.format(len(dataset_val)))
    print('test length is {}'.format(len(dataset_test)))

    if input_type == 'phase_and_amp':
        ch = [0, 1]
    elif input_type == 'phase_only' or 'fourier_transform':
        ch = [0]
    elif input_type == 'intensity_only':
        ch = [1]

    return train_loader, val_loader, test_loader, ch, number_of_type




