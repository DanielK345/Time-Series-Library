from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, xPatch

args_dict = {
        # Basic config
        'task_name': 'long_term_forecast',
        'is_training': 1,
        'model_id': 'test',
        # 'model': model,  # or 'xPatch', 'Transformer', 'TimesNet', etc.

        # Data loader
        'data': 'ETTh1',
        # 'root_path': root_path,
        'data_path': 'ETTh1.csv',
        'features': 'M',
        'target': 'OT',
        'freq': 'h',
        'checkpoints': './model_checkpoints/',

        # Forecasting task
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 96,
        'seasonal_patterns': 'Monthly',
        'inverse': False,

        # Imputation task
        'mask_rate': 0.25,

        # Anomaly detection task
        'anomaly_ratio': 0.25,

        # Model define
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
        # 'distil': True,
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

        # Optimization
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

        # GPU
        'use_gpu': True,
        'gpu': 0,
        'gpu_type': 'cuda',
        'use_multi_gpu': False,
        'devices': '0,1,2,3',

        # De-stationary projector params
        'p_hidden_dims': [128, 128],
        'p_hidden_layers': 2,

        # Metrics (dtw)
        'use_dtw': False,

        # Augmentation
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
        'extra_tag': "",

        # TimeXer & xPatch
        'patch_len': 16,

        # xPatch-specific
        'stride': 8,
        'padding_patch': 'end',
        'ma_type': 'ema',
        'alpha': 0.3,
        'beta': 0.3,
        'revin': 1,
        'train_only': False,
}

model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'xPatch': xPatch
}