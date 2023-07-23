import os
import h5py
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from sklearn.preprocessing import LabelBinarizer

sys.path.append('..')
from Global_Random_Seed import RANDOM_SEED
from Models.utils import setup_logger

pl.seed_everything(RANDOM_SEED)

# 参数设置
parser = argparse.ArgumentParser(description='Generate Localization_Data_for_Posture_Reconstruction Datasets')
parser.add_argument("--raw_data_path", type=str,
                    default='Raw_Data/Localization_Data_for_Posture_Reconstruction/ConfLongDemo_JSI.txt')
parser.add_argument("--seq_len", type=int, default=40)
parser.add_argument("--feature_num", type=int, default=4)
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--val_frac", type=float, default=0.2)
parser.add_argument("--artificial_missing_rate", type=float, default=0.1)
parser.add_argument('--saving_path', type=str, default='Generated_Datasets/New_Localization_Data_5flod_10%')
args = parser.parse_args()
# 若文件夹不存在则创建文件夹
if not os.path.exists(args.saving_path):
    os.makedirs(args.saving_path)


def random_missing(vector, artificial_missing_rate):
    b, l, f = vector.shape
    vector = vector.reshape(-1, args.feature_num).T.reshape(-1)
    x = vector.copy()
    # block missing
    indices = np.where(~np.isnan(vector))[0].tolist()
    n = int(len(indices) * artificial_missing_rate)
    miss_n = 0
    m = int((n * 0.5 - miss_n) / 10)
    while m > 0:
        m = int((n * 0.5 - miss_n) / 10)
        indices = np.where(~np.isnan(vector))[0].tolist()
        indices = np.random.choice(indices, m, replace=False)
        for i in range(m):
            vector[indices[i]:indices[i] + 10] = np.nan
        miss_n = len(np.where(np.isnan(vector))[0])

    # normal missing
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, n - miss_n, replace=False)
    vector[indices] = np.nan
    missing_mask = (~np.isnan(vector)).astype(np.float32)
    missing_mask = missing_mask.reshape(args.feature_num, -1).T.reshape(b, l, f)
    indicating_mask = ((~np.isnan(vector)) ^ (~np.isnan(x))).astype(np.float32)
    indicating_mask = indicating_mask.reshape(args.feature_num, -1).T.reshape(b, l, f)
    x_hat = vector.reshape(args.feature_num, -1).T.reshape(b, l, f)
    x = x.reshape(args.feature_num, -1).T.reshape(b, l, f)

    return x, x_hat, missing_mask, indicating_mask


# 保存为.h5文件
def saving_into_h5(saving_dir, data_dict, times):
    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        single_set.create_dataset('labels', data=data['labels'].astype(int))
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
    if set_name == 'train':
        data_dict = {
            'feature_vectors': X,
        }
    else:
        X, X_hat, missing_mask, indicating_mask = random_missing(X, artificial_missing_rate)
        data_dict = {
            'feature_vectors': X,
            'feature_vectors_hat': X_hat,
            'missing_mask': missing_mask,
            'indicating_mask': indicating_mask
        }

    return data_dict


# 数据处理流程
def process(times, logger, dataframe_list, lable, train_ID, test_ID):
    df_list = dataframe_list
    train_set_ids = train_ID
    test_set_ids = test_ID
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=args.val_frac)
    logger.info(f'There are total {len(train_set_ids)} patients in train set.')
    logger.info(f'There are total {len(val_set_ids)} patients in val set.')
    logger.info(f'There are total {len(test_set_ids)} patients in test set.')

    train_set = np.array(df_list)[train_set_ids]
    val_set = np.array(df_list)[val_set_ids]
    test_set = np.array(df_list)[test_set_ids]

    train_set_y = lable[train_set_ids]
    val_set_y = lable[val_set_ids]
    test_set_y = lable[test_set_ids]

    # 标准化
    scaler = StandardScaler()  # StandardScaler类是处理数据归一化和标准化。
    train_set = train_set.copy()
    val_set = val_set.copy()
    test_set = test_set.copy()
    train_set_X = scaler.fit_transform(train_set[:, :, 0:4].reshape(-1, 4)).reshape(-1, 40, 4)
    val_set_X = scaler.transform(val_set[:, :, 0:4].reshape(-1, 4)).reshape(-1, 40, 4)
    test_set_X = scaler.transform(test_set[:, :, 0:4].reshape(-1, 4)).reshape(-1, 40, 4)

    if args.artificial_missing_rate > 0:
        _, train_set_X, _, _ = random_missing(train_set_X, args.artificial_missing_rate)
        logger.info(f'Already masked out {args.artificial_missing_rate * 100}% values in train set')

    # 添加掩盖矩阵
    train_set_dict = add_artificial_mask(train_set_X, args.artificial_missing_rate, 'train')
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, 'val')
    test_set_dict = add_artificial_mask(test_set_X, args.artificial_missing_rate, 'test')

    logger.info(f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}')
    logger.info(f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}')

    train_set_dict['labels'] = train_set_y
    val_set_dict['labels'] = val_set_y
    test_set_dict['labels'] = test_set_y

    forward_data = {
        'train': train_set_dict,
        'val': val_set_dict,
        'test': test_set_dict
    }

    saving_into_h5(args.saving_path, forward_data, times=times)
    logger.info(f'All done. Saved to {args.saving_path}.')


if __name__ == '__main__':

    # 创建记录文件
    logger = setup_logger(os.path.join(args.saving_path + "/dataset_generating.log"),
                          'Localization_Data_for_Posture_Reconstruction Datasets', mode='w')
    logger.info(args)

    df_list = []
    with open(args.raw_data_path, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f, header=None,
                         names=['Sequence_Name', 'Tag_ID', 'Timestamp', 'Date', 'X', 'Y', 'Z', 'labels'])
        df = df.drop('Timestamp', axis=1)
        df = df.replace(
            {'Tag_ID': {'010-000-024-033': 1, '010-000-030-096': 2, '020-000-033-111': 3, '020-000-032-221': 4}})
        for i in ['A', 'B', 'C', 'D', 'E']:
            for j in ['01', '02', '03', '04', '05']:
                df_list.append(df[df['Sequence_Name'] == i + j].drop(['Sequence_Name', 'Date'], axis=1))

    data_list = []
    for item in df_list:
        n = item.shape[0]
        for index in range(0, n, 40):
            if index <= n - args.seq_len:
                df_tem = item[index:index + args.seq_len]
                data_list.append(df_tem)
                # if df_tem['labels'].nunique() == 1:
                #     data_list.append(df_tem)
                # else:
                #     continue
            else:
                break

    lable = np.array(data_list)[:, :, -1]
    one_hot = LabelBinarizer()  # 创建one-hot编码器
    lable = one_hot.fit_transform(lable.reshape(-1)).reshape(-1,40,11)  # 对特征进行one-hot编码
    logger.info(f'标签依次为: {one_hot.classes_}')

    all_recordID = [i for i in range(len(data_list))]
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    times = 0
    for train_index, test_index in kf.split(all_recordID):
        logger.info(f'This is the {times} time in a five-fold cross-validation.')
        train_ID = np.array(all_recordID)[train_index].tolist()
        test_ID = np.array(all_recordID)[test_index].tolist()
        process(times, logger, data_list, lable, train_ID, test_ID)
        times += 1
