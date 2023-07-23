import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import h5py
import pytorch_lightning as pl

sys.path.append('..')
from Global_Random_Seed import RANDOM_SEED
from Models.utils import setup_logger

pl.seed_everything(RANDOM_SEED)

# 参数设置
parser = argparse.ArgumentParser(description='Generate BeiJing_PM2.5 Dataset')
parser.add_argument("--raw_data_path", type=str, default='Raw_Data/BeiJing_PM2.5_Data')
parser.add_argument("--artificial_missing_rate", type=float, default=0.1)
parser.add_argument("--seq_len", type=int, default=36)
parser.add_argument('--saving_path', type=str, default='Generated_Datasets/BeiJing_PM2.5_Data')
args = parser.parse_args()
# 若文件夹不存在则创建文件夹
if not os.path.exists(args.saving_path):
    os.makedirs(args.saving_path)


def window_truncate(feature_vectors, seq_len):
    """ Generate time series samples, truncating windows from time-series data with a given sequence length.
    Parameters
    ----------
    feature_vectors: time series data, len(shape)=2, [total_length, feature_num]
    seq_len: sequence length
    """
    start_indices = np.asarray(range(feature_vectors.shape[0] // seq_len)) * seq_len
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx: idx + seq_len])
    return np.asarray(sample_collector).astype('float32')


# 保存为.h5文件
def saving_into_h5(saving_dir, data_dict, times):
    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        single_set.create_dataset('feature_vectors', data=data['feature_vectors'].astype(np.float32))
        if name in ['val', 'test']:
            single_set.create_dataset('feature_vectors_hat', data=data['feature_vectors_hat'].astype(np.float32))
            single_set.create_dataset('missing_mask', data=data['missing_mask'].astype(np.float32))
            single_set.create_dataset('indicating_mask', data=data['indicating_mask'].astype(np.float32))

    saving_path = os.path.join(saving_dir, 'datasets' + str(times) + '.h5')
    with h5py.File(saving_path, 'w') as hf:
        save_each_set(hf, 'train', data_dict['train'])
        save_each_set(hf, 'val', data_dict['val'])
        save_each_set(hf, 'test', data_dict['test'])


# 为验证集、测试集添加人工掩盖
def add_artificial_mask(X, artificial_missing_rate, set_name):
    sample_num, seq_len, feature_num = X.shape
    if set_name == 'train':
        data_dict = {
            'feature_vectors': X,
        }
    else:
        X = X.reshape(-1)
        assert len(X.shape) == 1
        indices = np.where(~np.isnan(X))[0].tolist()
        indices_for_holdout = np.random.choice(indices, int(len(indices) * artificial_missing_rate), replace=False)
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        data_dict = {
            'feature_vectors': X.reshape([sample_num, seq_len, feature_num]),
            'feature_vectors_hat': X_hat.reshape([sample_num, seq_len, feature_num]),
            'missing_mask': missing_mask.reshape([sample_num, seq_len, feature_num]),
            'indicating_mask': indicating_mask.reshape([sample_num, seq_len, feature_num])
        }

    return data_dict


if __name__ == '__main__':
    logger = setup_logger(os.path.join(args.saving_path + "/dataset_generating.log"),
                          'Generate BeiJing_PM2.5 dataset', mode='w')
    logger.info(args)
    ground_path = os.path.join(args.raw_data_path, 'pm25_ground.txt')
    df = pd.read_csv(ground_path, sep=",")
    feature_names = [feature for feature in df.columns[1:]]
    feature_num = len(feature_names)
    logger.info(f'Original df missing rate: '
                f'{(df.iloc[:, 1:].isna().sum().sum() / (df.shape[0] * feature_num)):.3f}')
    df['datetime'] = pd.to_datetime(df['datetime'])
    unique_months = df['datetime'].dt.to_period('M').unique()
    selected_as_test = unique_months.take([2, 5, 8, 11])
    logger.info(f'months selected as test set are {selected_as_test}')
    selected_as_val = unique_months.take([3, 6])
    logger.info(f'months selected as val set are {selected_as_val}')
    selected_as_train = unique_months.take([0, 1, 4, 7, 9, 10])
    logger.info(f'months selected as train set are {selected_as_train}')

    test_set = df[df['datetime'].dt.to_period('M').isin(selected_as_test)]
    val_set = df[df['datetime'].dt.to_period('M').isin(selected_as_val)]
    train_set = df[df['datetime'].dt.to_period('M').isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)
    val_set_X = window_truncate(val_set_X, args.seq_len)
    test_set_X = window_truncate(test_set_X, args.seq_len)

    for times in range(5):
        train_set_dict = add_artificial_mask(train_set_X, args.artificial_missing_rate, 'train')
        val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, 'val')
        test_set_dict = add_artificial_mask(test_set_X, args.artificial_missing_rate, 'test')
        logger.info(f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}')
        logger.info(f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}')

        processed_data = {
            'train': train_set_dict,
            'val': val_set_dict,
            'test': test_set_dict
        }

        logger.info(f'Feature num: {feature_num},\n'
                    f'Sample num in train set: {len(train_set_dict["feature_vectors"])}\n'
                    f'Sample num in val set: {len(val_set_dict["feature_vectors"])}\n'
                    f'Sample num in test set: {len(test_set_dict["feature_vectors"])}\n')

        saving_into_h5(args.saving_path, processed_data, times)
        logger.info(f'The {times + 1} time in five is done. Saved to {args.saving_path}.')
    logger.info(f'All done. Saved to {args.saving_path}.')
