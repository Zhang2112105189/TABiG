# PhysioNet2012数据集的数据加载

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 训练集每个样本的人工掩蔽比例
train_artificial_missing_rate = 0.2


# 计算时间间隔矩阵
def parse_delta(masks, seq_len, feature_num):
    deltas = []
    for h in range(seq_len):
        if h == 0:
            deltas.append(np.zeros(feature_num))
        else:
            deltas.append(np.ones(feature_num) + (1 - masks[h]) * deltas[-1])
    return np.stack(deltas, axis=0)


# 加载验证集、测试集
class LoadValTestDataset(Dataset):
    def __init__(self, file_path, set_name, seq_len, feature_num):
        super(LoadValTestDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        with h5py.File(self.file_path, 'r') as hf:
            self.X = hf[set_name]['feature_vectors'][:]
            self.X_hat = hf[set_name]['feature_vectors_hat'][:]
            self.missing_mask = hf[set_name]['missing_mask'][:]
            self.indicating_mask = hf[set_name]['indicating_mask'][:]

        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        forward = {'feature_vectors_hat': self.X_hat[idx], 'missing_mask': self.missing_mask[idx],
                   'deltas': parse_delta(self.missing_mask[idx], self.seq_len, self.feature_num)}
        backward = {'feature_vectors_hat': np.flip(forward['feature_vectors_hat'], axis=0).copy(),
                    'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
        backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
        sample = (
            torch.tensor(idx),
            # 前向数据
            torch.from_numpy(forward['feature_vectors_hat'].astype('float32')),
            torch.from_numpy(forward['missing_mask'].astype('float32')),
            torch.from_numpy(forward['deltas'].astype('float32')),
            # 后向数据
            torch.from_numpy(backward['feature_vectors_hat'].astype('float32')),
            torch.from_numpy(backward['missing_mask'].astype('float32')),
            torch.from_numpy(backward['deltas'].astype('float32')),

            torch.from_numpy(self.X[idx].astype('float32')),
            torch.from_numpy(self.indicating_mask[idx].astype('float32')),
        )
        return sample


# 加载训练集
class LoadTrainDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num):
        super(LoadTrainDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num

        assert 0 < train_artificial_missing_rate < 1, 'artificial_missing_rate should be greater than 0 and less than 1'

        with h5py.File(self.file_path, 'r') as hf:
            self.X = hf['train']['feature_vectors'][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        X = X.reshape(-1)
        indices = np.where(~np.isnan(X))[0].tolist()
        indices = np.random.choice(indices, round(len(indices) * train_artificial_missing_rate), replace=False)
        X_hat = np.copy(X)
        if len(indices)!=0:
            X_hat[indices] = np.nan  # mask values selected by indices
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)
        X = np.nan_to_num(X)
        X_hat = np.nan_to_num(X_hat)

        X = X.reshape(self.seq_len, self.feature_num)
        X_hat = X_hat.reshape(self.seq_len, self.feature_num)
        missing_mask = missing_mask.reshape(self.seq_len, self.feature_num)
        indicating_mask = indicating_mask.reshape(self.seq_len, self.feature_num)


        forward = {'feature_vectors_hat': X_hat, 'missing_mask': missing_mask,
                   'deltas': parse_delta(missing_mask, self.seq_len, self.feature_num)}

        backward = {'feature_vectors_hat': np.flip(forward['feature_vectors_hat'], axis=0).copy(),
                    'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
        backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
        sample = (
            torch.tensor(idx),
            # 前向
            torch.from_numpy(forward['feature_vectors_hat'].astype('float32')),
            torch.from_numpy(forward['missing_mask'].astype('float32')),
            torch.from_numpy(forward['deltas'].astype('float32')),
            # 后向
            torch.from_numpy(backward['feature_vectors_hat'].astype('float32')),
            torch.from_numpy(backward['missing_mask'].astype('float32')),
            torch.from_numpy(backward['deltas'].astype('float32')),

            torch.from_numpy(X.astype('float32')),
            torch.from_numpy(indicating_mask.astype('float32')),
        )
        return sample


class LoadDataForImputation(Dataset):
    def __init__(self, file_path, set_name, seq_len, feature_num):
        super(LoadDataForImputation, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num

        with h5py.File(self.file_path, 'r') as hf:
            self.X = hf[set_name]['feature_vectors'][:]
        self.missing_mask = (~np.isnan(self.X)).astype(np.float32)
        self.X = np.nan_to_num(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        forward = {'feature_vectors': self.X[idx], 'missing_mask': self.missing_mask[idx],
                   'deltas': parse_delta(self.missing_mask[idx], self.seq_len, self.feature_num)}

        backward = {'feature_vectors': np.flip(forward['feature_vectors'], axis=0).copy(),
                    'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
        backward['deltas'] = parse_delta(backward['missing_mask'], self.seq_len, self.feature_num)
        sample = (
            torch.tensor(idx),
            # 前向
            torch.from_numpy(forward['feature_vectors'].astype('float32')),
            torch.from_numpy(forward['missing_mask'].astype('float32')),
            torch.from_numpy(forward['deltas'].astype('float32')),
            # 后向
            torch.from_numpy(backward['feature_vectors'].astype('float32')),
            torch.from_numpy(backward['missing_mask'].astype('float32')),
            torch.from_numpy(backward['deltas'].astype('float32')),
        )
        return sample

class MyDataLoader:
    def __init__(self, dataset_path, seq_len, feature_num, batch_size=128, num_workers=4):
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None

    def get_train_val_dataloader(self):
        self.train_dataset = LoadTrainDataset(self.dataset_path, self.seq_len, self.feature_num)
        self.val_dataset = LoadValTestDataset(self.dataset_path, 'val', self.seq_len, self.feature_num)
        self.train_set_size = self.train_dataset.__len__()
        self.val_set_size = self.val_dataset.__len__()
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.train_loader, self.val_loader

    def get_test_dataloader(self):
        self.test_dataset = LoadValTestDataset(self.dataset_path, 'test', self.seq_len, self.feature_num)
        self.test_set_size = self.test_dataset.__len__()
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.test_loader

    def prepare_dataloader_for_imputation(self, set_name):
        data_for_imputation = LoadDataForImputation(self.dataset_path, set_name, self.seq_len, self.feature_num)
        dataloader_for_imputation = DataLoader(data_for_imputation, self.batch_size, shuffle=False)
        return dataloader_for_imputation

    def prepare_all_data_for_imputation(self):
        train_set_for_imputation = self.prepare_dataloader_for_imputation('train')
        val_set_for_imputation = self.prepare_dataloader_for_imputation('val')
        test_set_for_imputation = self.prepare_dataloader_for_imputation('test')
        return train_set_for_imputation, val_set_for_imputation, test_set_for_imputation
