import sys, random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset


class AugComp(Dataset):

    def __init__(self, data, cl, tl, id, transform=None):
        self.data = data
        self.cl = cl
        self.tl = tl
        self.id = id
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, cl_target, tl_target, idx = self.data[index], self.cl[index], self.tl[index], self.id[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)  # for cifar10
        # img = Image.fromarray(img.numpy(), mode='L')  # for [mnist, fashion, kuzushiji]

        if self.transform is not None:
            img = self.transform(img)

        return img, cl_target, tl_target, idx

    def __len__(self):
        return len(self.data)


def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)  # (len(labels), K)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class: (len(labels), K-1)
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels


def class_prior(complementary_labels):
    # p(\bar{y})
    return np.bincount(complementary_labels) / len(complementary_labels)


def prepare_data(dataset, batch_size):
    if dataset == "mnist":
        ordinary_train_dataset = dsets.MNIST(root='./data/MNIST', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./data/MNIST', train=False, transform=transforms.ToTensor())
        input_dim, input_channel = 28 * 28, 1
    elif dataset == "kuzushiji":
        ordinary_train_dataset = dsets.KMNIST(root='./data/KMNIST', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.KMNIST(root='./data/KMNIST', train=False, transform=transforms.ToTensor())
        input_dim, input_channel = 28 * 28, 1
    elif dataset == "fashion":
        ordinary_train_dataset = dsets.FashionMNIST(root='./data/FashionMNIST', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.FashionMNIST(root='./data/FashionMNIST', train=False, transform=transforms.ToTensor())
        input_dim, input_channel = 28 * 28, 1
    elif dataset == "cifar10":
        ordinary_train_dataset = dsets.CIFAR10(root='./data/CIFAR10', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.CIFAR10(root='./data/CIFAR10', train=False, transform=transforms.ToTensor())
        input_dim, input_channel = 3 * 32 * 32, 3
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    num_classes = len(ordinary_train_dataset.classes)
    return train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes, input_dim, input_channel


def prepare_train_loaders(batch_size, ordinary_train_dataset, cl_num, data_aug=None):
    """
        ccp is only used for "free", "ga", "nn"
        partialY is used for utils_mcl_loss, in this case, complementary_labels(var) is useless
    """
    # load raw_data if data_aug is not None
    if data_aug is not None:
        data, labels = ordinary_train_dataset.data, ordinary_train_dataset.targets
        bs, K = len(ordinary_train_dataset.data), len(ordinary_train_dataset.classes)
        labels = torch.LongTensor(labels)
    else:
        full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset.data), shuffle=False)
        for i, (data, labels) in enumerate(full_train_loader):
            K = torch.max(labels) + 1
            bs = labels.size(0)

    complementary_labels = generate_compl_labels(labels)
    x_to_tls = {i: -1 for i in range(bs)}
    x_to_mcls = {i: set() for i in range(bs)}
    partialY = torch.ones(bs, K).scatter_(1, torch.LongTensor(complementary_labels).unsqueeze(1), 0)
    for idx, cl in enumerate(complementary_labels.tolist()):
        x_to_mcls[idx] = {cl}
    for idx, tl in enumerate(labels.tolist()):
        x_to_tls[idx] = tl
        if cl_num != 1:
            all = set(i for i in range(K))
            cl = complementary_labels[idx].item()
            mcls = random.sample(all - {tl, cl}, cl_num-1)
            mcls.append(cl)
            x_to_mcls[idx] = set(mcls)
            partialY[idx] = torch.ones(K).scatter_(0, torch.LongTensor(mcls), 0)
    ccp = class_prior(complementary_labels)
    id = torch.arange(bs)

    if data_aug is not None:
        complementary_dataset = AugComp(data=data, cl=torch.from_numpy(complementary_labels).long(), tl=labels, id=id, transform=data_aug)
    else:
        complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).long(), labels, id)
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=batch_size, shuffle=True)

    return complementary_train_loader, ccp, x_to_mcls, x_to_tls, partialY
