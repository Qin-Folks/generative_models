import socket
import numpy as np
from torchvision import datasets, transforms
import torch
from torchvision.datasets import VisionDataset
import random
from collections import defaultdict


class MultiMNIST(VisionDataset):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, num_digits=2, image_size=64, channels=3, nxt_dig_prob=1, to_sort_label=False,
                 dig_to_use=None, rand_dig_combine=True, split_dig_set=False):
        path = data_root
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.seed_is_set = False  # multi threaded loading
        self.channels = channels
        self.to_sort_label = to_sort_label
        self.rand_dig_combine = rand_dig_combine
        self.split_dig_set = split_dig_set

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)
        self.nxt_dig_prob = nxt_dig_prob
        self.dig_to_use = dig_to_use
        self.idx_to_use = None

        if self.dig_to_use is not None:
            self.idx_to_use = []
            an_idx = 0
            for x in self.data:
                if x[1] in self.dig_to_use:
                    self.idx_to_use.append(an_idx)
                an_idx += 1
        else:
            self.dig_to_use = list(range(10))

        self.split_labels = []
        self.split_label_sets = []
        if self.split_dig_set:
            assert len(self.dig_to_use) >= self.num_digits
            split_unit = int(len(self.dig_to_use) / self.num_digits)
            for i in range(self.num_digits):
                if i == self.num_digits-1:
                    a_label_set = list([self.dig_to_use[x] for x in list(range(i * split_unit, len(self.dig_to_use)))])
                else:
                    a_label_set = list([self.dig_to_use[x] for x in list(range(i * split_unit, (i+1)*split_unit))])
                a_contained_label_set = []
                for a_idx, (_, a_label) in enumerate(self.data):
                    if a_label in a_label_set:
                        a_contained_label_set.append(a_idx)
                self.split_label_sets.append(a_label_set)
                self.split_labels.append(a_contained_label_set)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        # return int(self.N * 2 / (10 / self.num_digits))
        return self.N * 2
        # return 512

    def __getitem__(self, index):
        labels = []
        self.set_seed(index)
        image_size = self.image_size
        x = np.zeros((self.channels,
                      image_size,
                      image_size),
                     dtype=np.float32)

        has_digit = False
        rand_idxes = []
        existing_labels = []
        dig_idx = 0
        while dig_idx < self.num_digits:
            if random.uniform(0, 1) < self.nxt_dig_prob or (not has_digit and dig_idx == (self.num_digits-1)):
                has_digit = True
                if len(self.dig_to_use) == 10:
                    if self.split_dig_set:
                        idx = np.random.randint(len(self.split_labels[dig_idx]))
                        rand_idxes.append(idx)
                    else:
                        idx = np.random.randint(self.N)
                        rand_idxes.append(idx)
                else:
                    if self.split_dig_set:
                        idx = np.random.randint(len(self.split_labels[dig_idx]))
                        to_append = self.split_labels[dig_idx][idx]
                    else:
                        idx = np.random.randint(len(self.idx_to_use))
                        to_append = self.idx_to_use[idx]

                    if self.rand_dig_combine:
                        rand_idxes.append(to_append)
                    else:
                        if self.nxt_dig_prob > 0.99:
                            assert len(self.dig_to_use) > 1
                        if self.data[to_append][1] not in existing_labels:
                            rand_idxes.append(to_append)
                            existing_labels.append(self.data[to_append][1])
                        else:
                            continue
            else:
                rand_idxes.append('empty')
            dig_idx += 1

        if self.to_sort_label:
            rand_idx_label = []
            for rand_idxes_idx in range(len(rand_idxes)):
                a_rand_idx = rand_idxes[rand_idxes_idx]
                if self.split_dig_set:
                    rand_idx_label.append(self.data[a_rand_idx][1] if a_rand_idx != 'empty' else random.choice(self.split_label_sets[rand_idxes_idx]))
                else:
                    rand_idx_label.append(self.data[a_rand_idx][1] if a_rand_idx != 'empty' else random.choice(self.dig_to_use))

            arg_sort_rand_idx_label = np.argsort(rand_idx_label)
            rand_idxes = list([rand_idxes[i] for i in arg_sort_rand_idx_label])

        for n in range(len(rand_idxes)):
            cur_digit_idx = rand_idxes[n]
            if cur_digit_idx == 'empty':
                labels.append(-1)
            else:
                digit, dig_label = self.data[cur_digit_idx]
                labels.append(dig_label)
                img_qtr_sz = int(self.image_size/4)

                if self.channels > 1:
                    cc = n+1
                else:
                    cc = 0

                if self.num_digits == 2:
                    x[cc, img_qtr_sz:img_qtr_sz+self.digit_size, n*self.digit_size:(n+1)*self.digit_size] = \
                        np.copy(digit.numpy())

                elif self.num_digits == 3 or self.num_digits == 4:
                    idx_i = int(n / 2)
                    idx_j = n % 2
                    x[cc, idx_i*self.digit_size:(idx_i+1)*self.digit_size,
                        idx_j*self.digit_size:(idx_j+1)*self.digit_size] = np.copy(digit.numpy())

        # pick on digit to be in front
        if self.channels > 1:
            front = np.random.randint(self.num_digits)
            for cc in range(self.num_digits):
                if cc != front:
                    x[cc, :, :][x[front, :, :] > 0] = 0
        return x, torch.tensor(labels)
