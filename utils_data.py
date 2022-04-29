import sys, random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset
from scipy.special import comb


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
    """
        Generating single complementary label with uniform assumption
    """
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


def generate_uniform_mul_comp_labels(labels):
    """
        Generating multiple complementary labels following the distribution described in Section 5 of
        "Learning with Multiple Complementary Labels" by Lei Feng et al.
    """
    if torch.min(labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(labels) == 1:
        labels = labels - 1

    K = torch.max(labels) - torch.min(labels) + 1
    n = labels.shape[0]
    cardinality = 2 ** K - 2
    number = torch.tensor([comb(K, i + 1) for i in range(K - 1)])  # 0 to K-2, convert list to tensor
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K - 1)  # tensor of K-1
    for i in range(K - 1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i] + prob_dis[i - 1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()  # tensor: n
    mask_n = torch.ones(n)  # n is the number of train_data
    partialY = torch.ones(n, K)
    temp_num_comp_train_labels = 0  # save temp number of comp train_labels

    for j in range(n):  # for each instance
        # if j % 1000 == 0:
        #     print("current index:", j)
        for jj in range(K - 1):  # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_comp_train_labels = jj + 1  # decide the number of complementary train_labels
                mask_n[j] = 0

        candidates = torch.from_numpy(np.random.permutation(K.item()))  # because K is tensor type
        candidates = candidates[candidates != labels[j]]
        temp_comp_train_labels = candidates[:temp_num_comp_train_labels]

        for kk in range(len(temp_comp_train_labels)):
            partialY[j, temp_comp_train_labels[kk]] = 0  # fulfill the partial label matrix
    return partialY


def generate_mul_comp_labels(data, labels, s):
    """
        Generating multiple complementary labels given the fixed size s of complementary label set \bar{Y} of each instance
        by "Learning with Multiple Complementary Labels" by Lei Feng et al.
    """
    k = torch.max(labels) + 1
    n = labels.shape[0]
    index_ins = torch.arange(n)  # torch type
    realY = torch.zeros(n, k)
    realY[index_ins, labels] = 1
    partialY = torch.ones(n, k)

    labels_hat = labels.clone().numpy()
    candidates = np.repeat(np.arange(k).reshape(1, k), len(labels_hat), 0)  # candidate labels without true class
    mask = np.ones((len(labels_hat), k), dtype=bool)
    for i in range(s):
        mask[np.arange(n), labels_hat] = False
        candidates_ = candidates[mask].reshape(n, k - 1 - i)
        idx = np.random.randint(0, k - 1 - i, n)
        comp_labels = candidates_[np.arange(n), np.array(idx)]
        partialY[index_ins, torch.from_numpy(comp_labels)] = 0
        if i == 0:
            complementary_labels = torch.from_numpy(comp_labels)
            multiple_data = data
        else:
            complementary_labels = torch.cat((complementary_labels, torch.from_numpy(comp_labels)), dim=0)
            multiple_data = torch.cat((multiple_data, data), dim=0)
        labels_hat = comp_labels
    return partialY


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
    elif dataset == "cifar100":
        ordinary_train_dataset = dsets.CIFAR100(root='./data/CIFAR100', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.CIFAR100(root='./data/CIFAR100', train=False, transform=transforms.ToTensor())
        input_dim, input_channel = 3 * 32 * 32, 3
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    num_classes = len(ordinary_train_dataset.classes)
    return train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes, input_dim, input_channel


def prepare_train_loaders(batch_size, ordinary_train_dataset, cl_num, data_aug=None):
    """
        ccp is only used for "free", "ga", "nn"
        partialY is used for multiple complementary labels
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
    ccp = class_prior(complementary_labels)
    if cl_num == 0:
        partialY = generate_uniform_mul_comp_labels(labels)
    elif cl_num == 1:
        partialY = torch.ones(bs, K).scatter_(1, torch.LongTensor(complementary_labels).unsqueeze(1), 0)
        # ema = (torch.ones(bs, K).scatter_(1, torch.LongTensor(complementary_labels).unsqueeze(1), 0)) / (K - 1)
    else:  # 2-9
        partialY = generate_mul_comp_labels(data, labels, cl_num)
    ema = partialY / partialY.sum(1).unsqueeze(1)
    id = torch.arange(bs)
    if data_aug is not None:
        complementary_dataset = AugComp(data=data, cl=torch.from_numpy(complementary_labels).long(), tl=labels, id=id, transform=data_aug)
    else:
        complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).long(), labels, id)
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=batch_size, shuffle=True)

    return complementary_train_loader, ccp, partialY, ema
