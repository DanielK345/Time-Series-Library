import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.xPatch.exp_main import Exp_Main  # Added for xPatch
from utils.print_args import print_args
import random
import numpy as np
import logging
from datetime import datetime
import warnings

# Helper: Namespace for dict-to-attr
class Namespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# All defaults as in run.py
RUNPY_DEFAULTS = {
    'task_name': 'long_term_forecast',
    'is_training': 1,
    'model_id': 'test',
    'model': 'Autoformer',
    'train_multiple': False,
    'data': 'ETTh1',
    'root_path': './data/ETT/',
    'data_path': 'ETTh1.csv',
    'features': 'M',
    'target': 'OT',
    'freq': 'h',
    'checkpoints': './model_checkpoints/',
    'seq_len': 96,
    'label_len': 48,
    'pred_len': 96,
    'seasonal_patterns': 'Monthly',
    'inverse': False,
    'mask_rate': 0.25,
    'anomaly_ratio': 0.25,
    'expand': 2,
    'd_conv': 4,
    'top_k': 5,
    'num_kernels': 6,
    'enc_in': 7,
    'dec_in': 7,
    'c_out': 7,
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 2048,
    'moving_avg': 25,
    'factor': 1,
    'distil': True,
    'dropout': 0.1,
    'embed': 'timeF',
    'activation': 'gelu',
    'channel_independence': 1,
    'decomp_method': 'moving_avg',
    'use_norm': 1,
    'down_sampling_layers': 0,
    'down_sampling_window': 1,
    'down_sampling_method': None,
    'seg_len': 96,
    'num_workers': 10,
    'itr': 1,
    'train_epochs': 10,
    'batch_size': 32,
    'patience': 3,
    'learning_rate': 0.0001,
    'des': 'test',
    'loss': 'MSE',
    'lradj': 'type1',
    'use_amp': False,
    'use_gpu': True,
    'gpu': 0,
    'gpu_type': 'cuda',
    'use_multi_gpu': False,
    'devices': '0,1,2,3',
    'p_hidden_dims': [128, 128],
    'p_hidden_layers': 2,
    'use_dtw': False,
    'augmentation_ratio': 0,
    'seed': 2,
    'jitter': False,
    'scaling': False,
    'permutation': False,
    'randompermutation': False,
    'magwarp': False,
    'timewarp': False,
    'windowslice': False,
    'windowwarp': False,
    'rotation': False,
    'spawner': False,
    'dtwwarp': False,
    'shapedtwwarp': False,
    'wdba': False,
    'discdtw': False,
    'discsdtw': False,
    'extra_tag': '',
    'patch_len': 16,
    'stride': 8,
    'padding_patch': 'end',
    'ma_type': 'ema',
    'alpha': 0.3,
    'beta': 0.3,
    'revin': 1,
    'QAM_start': 0.1,
    'QAM_end': 0.3,
    'pct_start': 0.3,
    'query_independence': False,
    'store_attn': False,
}

def main(args_dict, model_path=None):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Merge user args with defaults
    args_data = dict(RUNPY_DEFAULTS)
    if args_dict is not None:
        if isinstance(args_dict, dict):
            args_data.update(args_dict)
        elif hasattr(args_dict, '__dict__'):
            args_data.update(vars(args_dict))
        else:
            raise ValueError('args_dict must be a dict or Namespace-like object')
    args = Namespace(**args_data)

    # Set random seed
    fix_seed = getattr(args, 'seed', 2021)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    logger.info(f"Set random seed to {fix_seed}")

    # Handle model argument: string or list
    model_arg = getattr(args, 'model', 'Autoformer')
    if isinstance(model_arg, list):
        model_list = model_arg
    else:
        model_list = [model_arg]

    # If not training multiple, only use the first model
    if not getattr(args, 'train_multiple', False):
        model_list = [model_list[0]]

    # Error/warning handling for train_multiple and model
    if not getattr(args, 'train_multiple', False):
        if len(model_list) > 1:
            warnings.warn("train_multiple is False but multiple models provided. Only the first model will be trained.")
        model_list = [model_list[0]]
    else:
        if not isinstance(args.model, list) or len(args.model) < 2:
            warnings.warn("train_multiple is True but model is not a list of 2 or more models. Only one model will be trained.")
            model_list = [model_list[0]]

    results = []
    for model_name in model_list:
        args.model = model_name
        # Device setup (CUDA/MPS/CPU)
        if torch.cuda.is_available() and getattr(args, 'use_gpu', True):
            args.device = torch.device(f'cuda:{getattr(args, "gpu", 0)}')
            logger.info('Using GPU')
        else:
            if hasattr(torch.backends, "mps"):
                args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            else:
                args.device = torch.device("cpu")
            logger.info('Using cpu or mps')

        # Multi-GPU logic
        if getattr(args, 'use_gpu', True) and getattr(args, 'use_multi_gpu', False):
            args.devices = getattr(args, 'devices', '0,1,2,3').replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
            logger.info(f"Using multiple GPUs: {args.device_ids}")

        logger.info('Experiment configuration:')
        for arg in vars(args):
            logger.info(f"{arg}: {getattr(args, arg)}")

        # Model/Exp selection
        if args.model == 'xPatch':
            Exp = Exp_Main
        elif args.task_name == 'long_term_forecast':
            Exp = Exp_Long_Term_Forecast
        elif args.task_name == 'short_term_forecast':
            Exp = Exp_Short_Term_Forecast
        elif args.task_name == 'imputation':
            Exp = Exp_Imputation
        elif args.task_name == 'anomaly_detection':
            Exp = Exp_Anomaly_Detection
        elif args.task_name == 'classification':
            Exp = Exp_Classification
        else:
            Exp = Exp_Long_Term_Forecast

        trained_model = None
        if getattr(args, 'is_training', 1):
            for ii in range(getattr(args, 'itr', 1)):
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.expand,
                    args.d_conv,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)

                logger.info(f"Starting iteration {ii+1}/{getattr(args, 'itr', 1)}")
                logger.info(f"Experiment setting: {setting}")

                exp = Exp(args)
                logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                start_time = datetime.now()
                if model_path is None:
                    trained_model = exp.train(setting)
                    training_time = datetime.now() - start_time
                    logger.info(f'Training completed in {training_time}')
                else:
                    if os.path.exists(model_path):
                        exp.model.load_state_dict(torch.load(model_path, map_location=args.device))
                        exp.model.eval()
                        logger.info(f"Loaded model from {model_path}")
                    else:
                        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

                logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                start_time = datetime.now()
                exp.test(setting)
                testing_time = datetime.now() - start_time
                logger.info(f'Testing completed in {testing_time}')

                if getattr(args, 'do_predict', False):
                    logger.info('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
                logger.info(f"GPU/CPU memory cleared after iteration {ii+1}")
            results.append(trained_model)
        else:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)
            logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
            results.append(exp.model)
    if len(results) == 1:
        return results[0]
    return results