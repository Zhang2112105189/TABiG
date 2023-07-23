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

sys.path.append('..')
from Global_Random_Seed import RANDOM_SEED
from Models.utils import setup_logger

pl.seed_everything(RANDOM_SEED)

# 参数设置
parser = argparse.ArgumentParser(description='Generate PhysioNet_2012 Datasets')
parser.add_argument("--raw_data_path", type=str, default='Raw_Data/PhysioNet_2012/mega')
parser.add_argument("--outcome_files_dir", type=str, default='Raw_Data/PhysioNet_2012/')
parser.add_argument("--seq_len", type=int, default=48)
parser.add_argument("--artificial_missing_rate", type=float, default=0.1)
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--val_frac", type=float, default=0.2)
parser.add_argument('--saving_path', type=str, default='Generated_Datasets/PhysioNet2012_5flod')
args = parser.parse_args()
# 若文件夹不存在则创建文件夹
if not os.path.exists(args.saving_path):
    os.makedirs(args.saving_path)


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


# 获取对应标签
def process_each_set(set_df, all_labels):
    # 获取对应标签
    sample_ids = set_df['RecordID'].to_numpy().reshape(-1, 48)[:, 0]
    y = all_labels.loc[sample_ids].to_numpy().reshape(-1, 1)
    # 生成特征向量
    set_df = set_df.drop('RecordID', axis=1)
    feature_names = set_df.columns.tolist()
    X = set_df.to_numpy()
    X = X.reshape(len(sample_ids), 48, len(feature_names))

    return X, y, feature_names


# 数据处理流程
def process(times, logger, df, all_outcomes, train_ID, test_ID):
    # 划分
    train_set_ids = train_ID
    test_set_ids = test_ID
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=args.val_frac)
    logger.info(f'There are total {len(train_set_ids)} patients in train set.')
    logger.info(f'There are total {len(val_set_ids)} patients in val set.')
    logger.info(f'There are total {len(test_set_ids)} patients in test set.')

    feats_to_normalize = df.columns.tolist()
    feats_to_normalize.remove('RecordID')

    train_set = df[df['RecordID'].isin(train_set_ids)]
    val_set = df[df['RecordID'].isin(val_set_ids)]
    test_set = df[df['RecordID'].isin(test_set_ids)]

    # 标准化
    scaler = StandardScaler()  # StandardScaler类是处理数据归一化和标准化。
    train_set = train_set.copy()
    val_set = val_set.copy()
    test_set = test_set.copy()
    train_set.loc[:, feats_to_normalize] = scaler.fit_transform(train_set.loc[:, feats_to_normalize])
    val_set.loc[:, feats_to_normalize] = scaler.transform(val_set.loc[:, feats_to_normalize])
    test_set.loc[:, feats_to_normalize] = scaler.transform(test_set.loc[:, feats_to_normalize])
    # 获取标签
    train_set_X, train_set_y, feature_names = process_each_set(train_set, all_outcomes)
    val_set_X, val_set_y, _ = process_each_set(val_set, all_outcomes)
    test_set_X, test_set_y, _ = process_each_set(test_set, all_outcomes)
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

    logger.info(f'All saved features: {feature_names}')
    saved_df = df.loc[:, feature_names]

    total_sample_num = 0
    total_positive_num = 0
    for set_name, rec in zip(['train', 'val', 'test'], [train_set_dict, val_set_dict, test_set_dict]):
        total_sample_num += len(rec["labels"])
        total_positive_num += rec["labels"].sum()
        logger.info(f'Positive rate in {set_name} set: {rec["labels"].sum()}/{len(rec["labels"])}='
                    f'{(rec["labels"].sum() / len(rec["labels"])):.3f}')
    logger.info(f'Dataset overall positive rate: {(total_positive_num / total_sample_num):.3f}')

    missing_part = np.isnan(saved_df.to_numpy())
    logger.info(f'Dataset overall missing rate of original feature vectors (without any artificial mask): '
                f'{(missing_part.sum() / missing_part.shape[0] / missing_part.shape[1]):.3f}')
    saving_into_h5(args.saving_path, forward_data, times=times)
    logger.info(f'The {times} time in a five-fold cross-validation is done. Saved to {args.saving_path}.')


if __name__ == '__main__':

    # 创建记录文件
    logger = setup_logger(os.path.join(args.saving_path + "/dataset_generating.log"),
                          'Generate PhysioNet2012 dataset', mode='w')
    logger.info(args)
    # 标签所在文件名
    outcome_files_list = ['Outcomes-a.txt', 'Outcomes-b.txt', 'Outcomes-c.txt']
    outcome_collector = []
    # 遍历标签文件，提取ID与标签信息
    for file in outcome_files_list:
        outcome_file_path = os.path.join(args.outcome_files_dir, file)
        with open(outcome_file_path, 'r') as f:
            outcome = pd.read_csv(f)[['In-hospital_death', 'RecordID']]
        outcome = outcome.set_index('RecordID')
        outcome_collector.append(outcome)
    all_outcomes = pd.concat(outcome_collector)

    all_recordID = []
    df_collector = []
    # 遍历数据文件，提取特征
    for filename in os.listdir(args.raw_data_path):
        recordID = int(filename.split('.txt')[0])
        with open(os.path.join(args.raw_data_path, filename), 'r') as f:
            df_temp = pd.read_csv(f)
        df_temp['Time'] = df_temp['Time'].apply(lambda x: int(x.split(':')[0]))
        df_temp = df_temp.pivot_table('Value', 'Time', 'Parameter')
        df_temp = df_temp.reset_index()  # take Time from index as a col
        if len(df_temp) == 1:
            logger.info(f'Pass {recordID}, because its len==1, having no time series data')
            continue
        all_recordID.append(recordID)  # only count valid recordID
        if df_temp.shape[0] != 48:
            missing = list(set(range(0, 48)).difference(set(df_temp['Time'])))
            missing_part = pd.DataFrame({'Time': missing})
            # df_temp = df_temp.append(missing_part, ignore_index=False, sort=False)
            df_temp = pd.concat([df_temp, missing_part], ignore_index=False, sort=False)
            df_temp = df_temp.set_index('Time').sort_index().reset_index()
        df_temp = df_temp.iloc[:48]  # only take 48 hours, some samples may have more records, like 49 hours
        df_temp['RecordID'] = recordID
        df_temp['Age'] = df_temp.loc[0, 'Age']
        df_temp['Height'] = df_temp.loc[0, 'Height']
        df_collector.append(df_temp)
    df = pd.concat(df_collector, sort=True)
    # 去除不需要的特征
    df = df.drop(['Age', 'Gender', 'ICUType', 'Height', 'MechVent', 'Weight'], axis=1)
    df = df.reset_index(drop=True)
    df = df.drop('Time', axis=1)
    feature_names = df.columns.tolist()
    feature_names.remove('RecordID')
    for id in all_recordID:
        if df[df['RecordID'] == id].loc[:, feature_names].isna().all(axis=None):
            all_recordID.remove(id)
            logger.info(f'Pass {id}, because its all feature is nan, having no time series data')
    # 五折交叉验证，生成5个数据集
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    times = 0
    for train_index, test_index in kf.split(all_recordID):
        logger.info(f'This is the {times} time in a five-fold cross-validation.')
        train_ID = np.array(all_recordID)[train_index].tolist()
        test_ID = np.array(all_recordID)[test_index].tolist()
        process(times, logger, df, all_outcomes, train_ID, test_ID)
        times += 1
