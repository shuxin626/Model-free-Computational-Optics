from torchvision import datasets, transforms
from data.data_utils import dataset_selector, get_index_for_each_type

class Phaseamp_FMNIST(datasets.FashionMNIST):
    def __init__(self, train=True, transform_list=[], type_idx_list=None, num_per_type=None, begin_sampling=None):
        # output shape [phase/amp, height, width], phase in channel 0, amp in channel 1
        transform_origin_dataset = transforms.Compose(transform_list)
        super().__init__(root='data/F_MNIST_data/', train=train, transform=transform_origin_dataset, download=True)
        self.data, self.targets = dataset_selector(self.data, self.targets, type_idx_list, num_per_type, begin_sampling)
        self.number_of_type = max(self.targets) + 1
        self.index_for_each_type = get_index_for_each_type(self.targets)

class Phaseamp_CIFAR10(datasets.CIFAR10):
    def __init__(self, train=True, transform_list=[], type_idx_list=None, num_per_type=None, begin_sampling=None):
        transform_origin_dataset = transforms.Compose(transform_list)
        super().__init__(root='data/CIFAR_10/', train=train, transform=transform_origin_dataset, download=True)
        self.data, self.targets = dataset_selector(self.data, self.targets, type_idx_list, num_per_type, begin_sampling)
        self.number_of_type = max(self.targets) + 1
        self.index_for_each_type = get_index_for_each_type(self.targets)

 
class Phaseamp_MNIST(datasets.MNIST):
    def __init__(self, train=True, transform_list=[], type_idx_list=None, num_per_type=None, begin_sampling=None):
        transform_origin_dataset = transforms.Compose(transform_list)
        super().__init__(root='data/MNIST/', train=train, transform=transform_origin_dataset, download=True)
        self.data, self.targets = dataset_selector(self.data, self.targets, type_idx_list, num_per_type, begin_sampling)
        self.number_of_type = max(self.targets) + 1
        self.index_for_each_type = get_index_for_each_type(self.targets)

    
