import nni
import os
import h5py
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from configparser import ConfigParser, ExtendedInterpolation
from Global_Random_Seed import RANDOM_SEED
from Models.My_Model import My_Models
from Models.DataLoader import MyDataLoader
from Models.utils import Controller, setup_logger, save_model, load_model, check_saving_dir_for_model, masked_mae_cal, \
    masked_rmse_cal, masked_mre_cal
import pytorch_lightning as pl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
pl.seed_everything(RANDOM_SEED)


# 从配置文件读取参数
def read_arguments(arg_parser, cfg_parser):
    # file path
    arg_parser.dataset_base_dir = cfg_parser.get('file_path', 'dataset_base_dir')
    arg_parser.result_saving_base_dir = cfg_parser.get('file_path', 'result_saving_base_dir')
    # dataset info
    arg_parser.seq_len = cfg_parser.getint('dataset', 'seq_len')
    arg_parser.batch_size = cfg_parser.getint('dataset', 'batch_size')
    arg_parser.num_workers = cfg_parser.getint('dataset', 'num_workers')
    arg_parser.feature_num = cfg_parser.getint('dataset', 'feature_num')
    arg_parser.eval_every_n_steps = cfg_parser.getint('dataset', 'eval_every_n_steps')
    arg_parser.dataset_path = os.path.join(arg_parser.dataset_base_dir, arg_parser.dataset_name)
    # training settings
    arg_parser.lr = cfg_parser.getfloat('training', 'lr')
    arg_parser.device = cfg_parser.get('training', 'device')
    arg_parser.epochs = cfg_parser.getint('training', 'epochs')
    arg_parser.early_stop_patience = cfg_parser.getint('training', 'early_stop_patience')
    arg_parser.model_saving_strategy = cfg_parser.get('training', 'model_saving_strategy')
    arg_parser.max_norm = cfg_parser.getfloat('training', 'max_norm')
    arg_parser.imputation_loss_weight = cfg_parser.getfloat('training', 'imputation_loss_weight')
    arg_parser.reconstruction_loss_weight = cfg_parser.getfloat('training', 'reconstruction_loss_weight')
    arg_parser.consistency_loss_weight = cfg.getfloat('training', 'consistency_loss_weight')
    # model settings
    arg_parser.hidden_size = cfg.getint('model', 'hidden_size')
    arg_parser.groups_num = cfg.getint('model', 'groups_num')
    arg_parser.inner_layers_size = cfg.getint('model', 'inner_layers_size')
    arg_parser.head_num = cfg.getint('model', 'head_num')
    arg_parser.attention_size = cfg.getint('model', 'attention_size')
    arg_parser.dropout_rate = cfg.getfloat('model', 'dropout_rate')

    return arg_parser


# 训练流程
def train(model, optimizer, train_dataloader, test_dataloader, summary_writer, training_controller, logger):
    # 开始迭代
    for epoch in range(args.epochs):
        early_stopping = False
        args.final_epoch = True if epoch == args.epochs - 1 else False
        # 遍历数据集，获取一个batch
        for idx, data in enumerate(train_dataloader):
            # 模型进入训练模式
            model.train()
            # 进入模型处理流程
            early_stopping = model_processing(data, model, 'train', optimizer, test_dataloader, summary_writer,
                                              training_controller, logger)
            if early_stopping:
                break
        if early_stopping:
            logger.info(f'Totally, {epoch} epochs were running.')
            break
        # 迭代次数加一
        training_controller.epoch_num_plus_1()
    logger.info('Finished all epochs. Stop training now.')


# 模型处理流程
def model_processing(data, model, stage, optimizer=None, val_dataloader=None, summary_writer=None,
                     training_controller=None, logger=None):
    # 训练模式
    if stage == 'train':
        # 梯度归零
        optimizer.zero_grad()
        # 获取输入数据
        indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, \
        indicating_mask = map(lambda x: x.to(args.device), data)
        # 输入数据字典
        inputs = {'indices': indices, 'X_holdout': X_holdout, 'indicating_mask': indicating_mask,
                  'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                  'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
        # model()输入模型，result_processing()处理模型输出
        # model(inputs, stage) 等同于forward(),开始训练
        results = result_processing(model(inputs, stage))
        # 处理训练步骤，判断是否早停止训练
        early_stopping = process_each_training_step(results, optimizer, val_dataloader,
                                                    training_controller, summary_writer, logger)
        return early_stopping

    # 验证/测试模式
    else:
        # 获取输入数据
        indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, \
        indicating_mask = map(lambda x: x.to(args.device), data)
        # 输入数据字典
        inputs = {'indices': indices, 'X_holdout': X_holdout, 'indicating_mask': indicating_mask,
                  'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                  'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
        # 输入模型，得到结果
        results = model(inputs, stage)
        # 处理模型结果
        results = result_processing(results)
        return inputs, results


# 模型结果处理，计算总损失
def result_processing(results):
    results['total_loss'] = torch.tensor(0.0, device=args.device)
    results['consistency_loss'] = results['consistency_loss'] * args.consistency_loss_weight
    results['reconstruction_loss'] = results['reconstruction_loss'] * args.reconstruction_loss_weight
    results['imputation_loss'] = results['imputation_loss'] * args.imputation_loss_weight

    results['total_loss'] += results['consistency_loss']
    results['total_loss'] += results['imputation_loss']
    results['total_loss'] += results['reconstruction_loss']

    return results


# 训练步骤处理，判断是否停止
def process_each_training_step(results, optimizer, val_dataloader, training_controller, summary_writer, logger):
    state_dict = training_controller(stage='train')
    # 如果采用梯度裁剪
    # 当神经网络深度逐渐增加，网络参数量增多的时候，反向传播过程中链式法则里的梯度连乘项数便会增多，更易引起梯度消失和梯度爆炸。
    # 对于梯度爆炸问题，解决方法之一便是进行梯度剪裁，即设置一个梯度大小的上限。max_norm即该组网络参数梯度的范数上限
    if args.max_norm != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    # 总损失的反向传播
    results['total_loss'].backward()
    # 执行一次优化步骤,通过梯度下降法来更新参数的值
    optimizer.step()
    # 将summary写入tensorboard文件
    summary_write_into_tb(summary_writer, results, state_dict['train_step'], 'train')
    # 是否到达验证步骤
    if state_dict['train_step'] % args.eval_every_n_steps == 0:
        # 进入验证流程，并返回状态字典，判断是否早停止
        state_dict_from_val = validate(model, val_dataloader, summary_writer, training_controller, logger)
        if state_dict_from_val['should_stop']:
            logger.info(f'Early stopping worked, stop now...')
            return True
    return False


# 将summary写入tensorboard文件
def summary_write_into_tb(summary_writer, info_dict, step, stage):
    summary_writer.add_scalar(f'total_loss/{stage}', info_dict['total_loss'], step)
    summary_writer.add_scalar(f'consistency_loss/{stage}', info_dict['consistency_loss'], step)
    summary_writer.add_scalar(f'imputation_loss/{stage}', info_dict['imputation_loss'], step)
    summary_writer.add_scalar(f'imputation_MAE/{stage}', info_dict['imputation_MAE'], step)
    summary_writer.add_scalar(f'reconstruction_loss/{stage}', info_dict['reconstruction_loss'], step)
    summary_writer.add_scalar(f'reconstruction_MAE/{stage}', info_dict['reconstruction_MAE'], step)


# 验证流程
def validate(model, val_iter, summary_writer, training_controller, logger):
    # 模型进入验证模式
    model.eval()
    # 用于收集数据
    X_holdout_collector, indicating_mask_collector, imputed_data_collector = [], [], []
    total_loss_collector, consistency_loss_collector, reconstruction_loss_collector, imputation_loss_collector, reconstruction_MAE_collector = [], [], [], [], []

    with torch.no_grad():
        # 遍历验证集
        for idx, data in enumerate(val_iter):
            # 进入模型处理流程
            inputs, results = model_processing(data, model, 'val')
            # 收集数据
            X_holdout_collector.append(inputs['X_holdout'])
            indicating_mask_collector.append(inputs['indicating_mask'])
            imputed_data_collector.append(results['imputed_data'])
            # 收集损失
            total_loss_collector.append(results['total_loss'].data.cpu().numpy())
            consistency_loss_collector.append(results['consistency_loss'].data.cpu().numpy())
            reconstruction_loss_collector.append(results['reconstruction_loss'].data.cpu().numpy())
            imputation_loss_collector.append(results['imputation_loss'].data.cpu().numpy())
            reconstruction_MAE_collector.append(results['reconstruction_MAE'].data.cpu().numpy())

        # 拼接数据
        X_holdout_collector = torch.cat(X_holdout_collector)
        indicating_mask_collector = torch.cat(indicating_mask_collector)
        imputed_data_collector = torch.cat(imputed_data_collector)
        # 计算插补MAE
        imputation_MAE = masked_mae_cal(imputed_data_collector, X_holdout_collector, indicating_mask_collector)
    # 取均值，形成字典
    info_dict = {'total_loss': np.asarray(total_loss_collector).mean(),
                 'consistency_loss': np.asarray(consistency_loss_collector).mean(),
                 'reconstruction_loss': np.asarray(reconstruction_loss_collector).mean(),
                 'imputation_loss': np.asarray(imputation_loss_collector).mean(),
                 'reconstruction_MAE': np.asarray(reconstruction_MAE_collector).mean(),
                 'imputation_MAE': imputation_MAE.cpu().numpy().mean()
                 }

    state_dict = training_controller('val', info_dict, logger)
    # 将summary写入tensorboard文件
    summary_write_into_tb(summary_writer, info_dict, state_dict['val_step'], 'val')

    # 保存模型
    if (state_dict['save_model'] and args.model_saving_strategy) or args.model_saving_strategy == 'all':

        del_list = os.listdir(args.model_saving)
        for f in del_list:
            file_path = os.path.join(args.model_saving, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # 保存路径
        saving_path = os.path.join(
            args.model_saving, 'model_trainStep_{}_valStep_{}_imputationMAE_{:.4f}'.
                format(state_dict['train_step'], state_dict['val_step'], info_dict['imputation_MAE']))
        # 保存模型
        save_model(model, optimizer, state_dict, args, saving_path)
        logger.info(f'Saved model -> {saving_path}')
    return state_dict


# 测试训练好的模型
def test_trained_model(model, test_dataloader):
    logger.info(f'Start evaluating on whole test set...')
    # 模型进入测试模式
    model.eval()
    # 用于收集数据
    X_holdout_collector, indicating_mask_collector, imputed_data_collector = [], [], []
    with torch.no_grad():
        # 遍历测试集
        for idx, data in enumerate(test_dataloader):
            # 经过模型处理流程，得到输入与结果
            inputs, results = model_processing(data, model, 'test')
            # 收集数据
            X_holdout_collector.append(inputs['X_holdout'])
            indicating_mask_collector.append(inputs['indicating_mask'])
            imputed_data_collector.append(results['imputed_data'])
        # 拼接数据
        X_holdout_collector = torch.cat(X_holdout_collector)
        indicating_mask_collector = torch.cat(indicating_mask_collector)
        imputed_data_collector = torch.cat(imputed_data_collector)
        # 计算MAE MRE RMSE
        imputation_MAE = masked_mae_cal(imputed_data_collector, X_holdout_collector, indicating_mask_collector)
        imputation_MRE = masked_mre_cal(imputed_data_collector, X_holdout_collector, indicating_mask_collector)
        imputation_RMSE = masked_rmse_cal(imputed_data_collector, X_holdout_collector, indicating_mask_collector)
    # 评估指标字典
    assessment_metrics = {'imputation_MAE on the test set': imputation_MAE,
                          'imputation_MRE on the test set': imputation_MRE,
                          'imputation_RMSE on the test set': imputation_RMSE,
                          'trainable parameter num': args.total_params}
    # 记录评估指标
    with open(os.path.join(args.result_saving_path, 'overall_performance_metrics.out'), 'w') as f:
        logger.info('Overall performance metrics are listed as follows:')
        for k, v in assessment_metrics.items():
            logger.info(f'{k}: {v}')
            f.write(k + ': ' + str(v))
            f.write('\n')


# 补全缺失值
def impute_all_missing_data(model, train_data, val_data, test_data):
    logger.info(f'Start imputing all missing data in all train/val/test sets...')
    # 模型进入测试模式
    model.eval()
    imputed_data_dict = {}
    with torch.no_grad():
        for dataloader, set_name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
            indices_collector, imputed_data_collector = [], []
            # 遍历所有数据
            for idx, data in enumerate(dataloader):
                # 获取输入数据
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = \
                    map(lambda x: x.to(args.device), data)
                # 输入字典
                inputs = {'indices': indices, 'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                          'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
                # 输入模型，获取插补结果
                imputed_data = model(inputs, 'test')['imputed_data']
                # 收集数据
                indices_collector.append(indices)
                imputed_data_collector.append(imputed_data)
            # 拼接数据
            indices_collector = torch.cat(indices_collector)
            indices = indices_collector.cpu().numpy().reshape(-1)
            imputed_data_collector = torch.cat(imputed_data_collector)
            imputations = imputed_data_collector.data.cpu().numpy()
            # 对indices从小到大排序，使数据顺序与原划分好的数据集一致
            ordered = imputations[np.argsort(indices)]
            imputed_data_dict[set_name] = ordered
    # 把插补完成的数据保存为.h5文件
    imputation_saving_path = os.path.join(args.result_saving_path, 'imputations.h5')
    with h5py.File(imputation_saving_path, 'w') as hf:
        hf.create_dataset('imputed_train_set', data=imputed_data_dict['train'])
        hf.create_dataset('imputed_val_set', data=imputed_data_dict['val'])
        hf.create_dataset('imputed_test_set', data=imputed_data_dict['test'])

    logger.info(f'Done saving all imputed data into {imputation_saving_path}.')


if __name__ == '__main__':
    # 获取模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='Configs/New_Artificial_Localization_Data_Config.ini')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true',
                        help='test mode to test saved model', default=0)
    parser.add_argument('--dataset_name', type=str, default='New_Artificial_Localization_Data_5flod_70%/datasets0.h5')
    parser.add_argument('--model_name', type=str, default='test/datasets0')
    parser.add_argument('--cuda', type=str, default='5')
    parser.add_argument('--test_step', type=str, default='')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config_path)
    args = read_arguments(args, cfg)
    assert os.path.exists(args.config_path), f'Given config file "{args.config_path}" does not exists'
    dict_args = vars(args)

    # 模型参数字典
    model_args = {
        'seq_len': args.seq_len,
        'feature_num': args.feature_num,
        'hidden_size': args.hidden_size,
        'groups_num': args.groups_num,
        'inner_layers_size': args.inner_layers_size,
        'head_num': args.head_num,
        'attention_size': args.attention_size,
        'dropout_rate': args.dropout_rate,
        'device': args.device
    }
    # 当前时间
    time_now = datetime.now().__format__('%Y-%m-%d_T%H:%M:%S')
    # 检查路径是否存在，并创建所需的model与logs文件夹
    args.model_saving, args.log_saving = check_saving_dir_for_model(args, time_now)
    # 创建记录文件
    logger = setup_logger(args.log_saving + '_' + time_now, 'w')
    logger.info(f'args: {args}')
    logger.info(f'Config file path: {args.config_path}')
    logger.info(f'Model name: {args.model_name}')
    # 创建并初始化数据集对象
    my_dataloader = MyDataLoader(args.dataset_path, args.seq_len, args.feature_num, args.batch_size,
                                 args.num_workers)
    # 创建并初始化模型对象
    model = My_Models(**model_args)
    # 计算模型总参数
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Num of total trainable params is: {args.total_params}')
    # 放入cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        model = model.to(args.device)
    # 训练模式
    if not args.test_mode:
        logger.info(f'Creating Adam optimizer...')
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=dict_args['lr'])
        logger.info('Entering training mode...')
        # 加载训练集、测试集
        train_dataloader, val_dataloader = my_dataloader.get_train_val_dataloader()
        # 训练控制器，控制早停止，早停止（Early Stopping）是 当达到某种或某些条件时，认为模型已经收敛，结束模型训练，保存现有模型的一种手段。
        training_controller = Controller(args.early_stop_patience)

        logger.info(f'train set len is {my_dataloader.train_set_size}, batch size is {args.batch_size},'
                    f'so each epoch has {math.ceil(my_dataloader.train_set_size / args.batch_size)} steps')
        # 创建summary_writer
        tb_summary_writer = SummaryWriter(os.path.join(args.log_saving, 'tensorboard_' + time_now))
        # 进入训练流程
        train(model, optimizer, train_dataloader, val_dataloader, tb_summary_writer, training_controller, logger)
    # 测试模式
    else:
        logger.info('Entering testing mode...')
        # 获取测试所需参数
        args.model_path = os.path.join(args.result_saving_base_dir, args.model_name, 'models', args.test_step)
        args.result_saving_path = os.path.join(args.result_saving_base_dir, args.model_name)
        os.makedirs(args.result_saving_path) if not os.path.exists(args.result_saving_path) else None
        # 加载测试所用的模型
        model = load_model(model, args.model_path, logger)
        # 加载测试集
        test_dataloader = my_dataloader.get_test_dataloader()
        # 进入测试流程
        test_trained_model(model, test_dataloader)
        # 加载需要补全的训练集、验证集、测试集
        train_data, val_data, test_data = my_dataloader.prepare_all_data_for_imputation()
        # 补全数据集中的缺失值
        impute_all_missing_data(model, train_data, val_data, test_data)

    logger.info('All Done.')
