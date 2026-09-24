# Code from: https://github.com/Ladbaby/PyOmniTS
import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from utils.ExpConfigs import ExpConfigs

'''
variable `configs` are ONLY determined by argparse, not .yaml files.

.yaml files are automatically saved and maintained as references:

- configs.yaml in training directory:

    Exact configs used in that training.

- *.yaml in "configs" folder

    Reference default configs for each model and dataset combination. Only automatically updated if new argument are added in argparse.

Note: Models and datasets only access the arguments they need.
'''

def get_configs(args=None) -> ExpConfigs:
    parser = argparse.ArgumentParser(description='SPAQR irregular time-series forecasting')

    # basic config
    parser.add_argument('--checkpoints', type=str, default='storage/results/', help='where to save model checkpoints in training. By default, results are organized under storage/results/DATASET_NAME/MODEL_NAME/MODEL_ID/SEQ_LEN_PRED_LEN/Y_md_HM/iter0')
    parser.add_argument('--is_training', type=int, default=1, help='training or testing')
    parser.add_argument('--model_id', type=str, default='SPAQR_fixed_budget_P8_B144_QCSU_D4', help='Experiment identifier used in result and reference-config paths.')
    parser.add_argument('--model_name', type=str, default='SPAQR', choices=['SPAQR'], help='Model module name.')
    parser.add_argument('--task_name', type=str, choices=["long_term_forecast", "short_term_forecast", "imputation", "classification", "anomaly_detection", "representation_learning", "causal_discovery"], default='short_term_forecast', help='task name')

    # dataset & data loader
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment in regular time series forecasting datasets.")
    parser.add_argument('--collate_fn', type=str, default='collate_fn', help='Irregular-series collation function exposed by the selected dataset module.')
    parser.add_argument('--dataset_file_name', type=str, default=None, help='Some datasets contains more than one files, then this argument may be ignored.')
    parser.add_argument('--dataset_id', type=str, default='P12', help='Dataset identifier used in result paths.')
    parser.add_argument('--dataset_name', type=str, default='P12', choices=['HumanActivity', 'MIMIC_III', 'P12', 'USHCN'], help='Benchmark dataset module name.')
    parser.add_argument('--dataset_root_path', type=str, default='storage/datasets/P12', help='Dataset root path (the four supplied scripts set the correct value).')
    parser.add_argument('--dataset_subset_name', type=str, default=None, help='Some datasets may be composed of multiple subsets (e.g., UCR128).')
    parser.add_argument('--embed', type=str, choices=["timeF", "fixed", "learned"], default='timeF', help='time features encoding. Only used in some regular time series datasets.')
    parser.add_argument('--features', type=str, choices=['M', 'S', "MS"], default='M', help='M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate. WARNING: not thoroughly tested yet.')
    parser.add_argument('--freq', type=str, choices=['s', 't', 'h', 'd', 'b', 'w', 'm', 'others'], default='h', help='freq for time features encoding in regular time series forecasting datasets, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--missing_rate', type=float, default=0., help="Manually mask out some observations. WARNING: not thoroughly tested yet.")
    parser.add_argument('--target_variable_index', type=int, default=0, help='target variable index in datasets. Should not be used together with target_variable_name. WARNING: not thoroughly tested yet.')
    parser.add_argument('--target_variable_name', type=str, default="OT", help='target variable name in regular time series forecasting datasets. Originally named as --target.')
    parser.add_argument('--train_val_loader_drop_last', type=int, default=0, help="By default, train and val loader will not drop the last batch if the number of samples is not sufficient.")
    parser.add_argument('--train_val_loader_shuffle', type=int, default=1, help="By default, train and val loader are shuffled.")

    # forecasting task
    parser.add_argument('--label_len', type=int, default=0, help='start token length. In other words, the length prepanded to the input of decoder. Only used in some regular time series forecasting models.')
    parser.add_argument('--pred_len', type=int, default=3, help='prediction horizon')
    parser.add_argument('--seq_len', type=int, default=36, help='observation horizon')

    # classification task
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes')

    # GPU
    parser.add_argument('--allow_tf32', type=int, default=1, help='Whether PyTorch is allowed to use the TensorFloat32 (TF32) tensor cores, available on NVIDIA GPUs since Ampere. If set to 1, model accuracy will be lower, but speed can be faster. Refer to https://docs.pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere')
    parser.add_argument('--gpu_id', type=int, default=0, help='primary gpu id, will be overwritten when use_multi_gpu is 1. Originally named as --gpu.')
    parser.add_argument('--gpu_ids', type=str, default=None, help='string of device ids for multile gpus. Originally named as --devices. WARNING: not thoroughly tested yet. Currently, setting --use_multi_gpu 1 is enough.')
    parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use gpu.')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus, via huggingface accelerate library')

    # training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--itr', type=int, default=5, help='Number of runs for each experiment setting. By default, we run every experiment setting for 5 times using 5 different random seeds (2024~2028), then calculate the average and std of these 5 metrics.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function. PyOmniTS will try to construct the loss function from class Loss in loss_fns/LOSS.py')
    parser.add_argument('--lr_scheduler', type=str, choices=["ExponentialDecayLR", "ManualMilestonesLR", "DelayedStepDecayLR", "CosineAnnealingLR", "MultiStepLR", "StepLR"], default='DelayedStepDecayLR', help='learning rate scheduler. Originally named as --lradj')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1, help='gamma for StepLR and MultiStepLR.')
    parser.add_argument('--n_train_stages', type=int, default=1, help="Some models have multiple training stages, like pretraining + finetuning. e.g., --n_train_stages 2 will pass train_stage=1 and train_stage=2 to model during training.")
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--pretrained_checkpoint_file_name', type=str, default=None, help="file name of pretrained model's checkpoints, including file type extension")
    parser.add_argument('--pretrained_checkpoint_root_path', type=str, default=None, help="Path to folder containing pretrained model's checkpoints")
    parser.add_argument('--retain_graph', type=int, default=0, help='whether to retain compute graph in back propagation. Used in special models like HD_TTS.')
    parser.add_argument('--sweep', type=int, default=0, help='whether to use weight & bias for hyperparameter searching')
    parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval relative to training epochs')
    parser.add_argument('--wandb', type=int, default=0, help='whether to use weight & bias for monitoring')

    # testing
    parser.add_argument('--checkpoints_test', type=str, default=None, help='folder where model checkpoint file is saved, for testing. If not given, Exp_Main.test() in exp/exp_main.py will try to find the latest checkpoint.')
    parser.add_argument('--load_checkpoints_test', type=int, default=1, help='whether to load checkpoint during testing')
    parser.add_argument('--save_arrays', type=int, default=0, help='whether to save model input and output as .npy files during testing, for later visualization')
    parser.add_argument('--save_cache_arrays', type=int, default=0, help='whether to save model output (not input) as cache .npy files during every test iteration, such that the testing can recover from interruption. Designed for extremely slow models like diffusion models.')
    parser.add_argument('--test_all', type=int, default=0, help='By default, we test the model on test set only. Setting this option to 1 will test the model on all samples from train, val, and test sets. Some datasets may not support this option yet.')
    parser.add_argument('--test_dataset_statistics', type=int, default=0, help="Test dataset's statistics. See function Exp_Main.test() in exp/exp_main.py for implementation details.")
    parser.add_argument('--test_flop', type=int, default=0, help='Test model flops. See utils/tools for implementation details.')
    parser.add_argument('--test_gpu_memory', type=int, default=0, help="Test model's gpu memory usage. See utils/tools for implementation details.")
    parser.add_argument('--test_train_time', type=int, default=0, help="Test model's training time. See utils/tools for implementation details.")

    # model configs
    # common
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--c_out', type=int, default=36, help='number of output variables')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS/TimeMixer model')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers. Used to replace n_layers in some model decoders.')
    parser.add_argument('--d_model', type=int, default=512, help='size of latent embeddings')
    parser.add_argument('--d_timesteps', type=int, default=1, help='UNUSED. Size of last dimension of `x_mark`/`y_mark`. Many Regular/Spatiotemporal datasets stack time in day, day in week, etc. along the last dimension. Others default to size 1.')
    parser.add_argument('--dec_in', type=int, default=36, help='number of decoder variables')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers. Used to replace n_layers in some model encoders.')
    parser.add_argument('--embed_type', type=int, choices=[0, 1, 2, 3, 4], default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding; Only used in some regular time series forecasting models.')
    parser.add_argument('--enc_in', type=int, default=36, help='number of observed variables') 
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--output_attention', type=int, default=0, help='DEPRECATED. output attention weight')
    parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers. WARNING: Will be replaced by --n_layers in the future, due to duplicated meaning.')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0. Only used in some regular time series forecasting models.')
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel size')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads (in attention)')
    parser.add_argument('--n_layers', type=int, default=1, help='num of layers')
    parser.add_argument('--n_patches_list', type=int, nargs="+", default=[2], help='List of number of patches. --patch_len is more commonly used, and in most cases, you wont have to set both --n_patches_list and --patch_len.')
    parser.add_argument('--node_dim', type=int, default=10, help='hidden dimension of nodes used in a few GNNs, like tPatchGNN')
    parser.add_argument('--patch_len', type=int, default=12, help='patch length. Also used as period_len in some models (SparseTSF).')
    parser.add_argument('--patch_len_list', type=int, nargs="+", default=[12, 6], help="list of patch length. Note: you don't have to set --patch_len at the same time.")
    parser.add_argument('--patch_stride', type=int, default=12, help='stride when splitting patches. Originally named as --stride.')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--revin_affine', type=int, default=0, help='RevIN-affine; True 1 False 0. Originally named as --affine.')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample. Only used in some regular time series forecasting models.')
    parser.add_argument('--top_k', type=int, default=5, help='top k selection')
    # Adaptor class
    parser.add_argument('--ts_backbone_name', type=str, default="", help='For adaptor class, which time series backbone to use. e.g., LSTM')
    parser.add_argument('--ts_backbone_overwrite_config_list', type=str, nargs='+', default=[], help='List of argument names that we want to overwrite in time series backbones, using argument values in adaptor class. E.g., [d_model, n_layers]')

    # ─── SPAQR-specific arguments ────────────────────────────────────────
    parser.add_argument('--spaqr_d_model', type=int, default=64, help='SPAQR hidden dimension')
    parser.add_argument('--spaqr_p_target', type=int, default=8,
                        help='Maximum support tokens permitted for one sample-variable pair')
    parser.add_argument('--spaqr_min_patch_mass', type=int, default=4,
                        help='Target observation mass used by the tokenizer count-distortion term')
    parser.add_argument('--spaqr_token_budget', type=int, default=144,
                        help='Fixed positive support-token budget per sample; required by SPAQR')
    parser.add_argument('--spaqr_count_penalty', type=float, default=1.0,
                        help='Weight of support mass distortion in budgeted tokenization')
    parser.add_argument('--spaqr_span_penalty', type=float, default=1.0,
                        help='Weight of physical-time span distortion in budgeted tokenization')
    parser.add_argument('--spaqr_gap_penalty', type=float, default=1.0,
                        help='Weight of large internal gaps in budgeted tokenization')
    parser.add_argument('--spaqr_num_geometric_freq', type=int, default=6, help='SPAQR: number of geometric Fourier bands (in addition to physical-period bands)')
    parser.add_argument('--spaqr_fourier_base_freq', type=float, default=1.0, help='SPAQR: base frequency for geometric Fourier bands')
    parser.add_argument('--spaqr_fourier_scale_factor', type=float, default=1.5, help='SPAQR: scale factor between adjacent geometric bands')
    parser.add_argument('--spaqr_var_embed_dim', type=int, default=16, help='SPAQR: variable embedding dim')
    parser.add_argument('--spaqr_n_heads', type=int, default=1, help='SPAQR attention heads')
    parser.add_argument('--spaqr_decoder_layers', type=int, default=4,
                        help='Depth of coarse target-conditioned global retrieval before support unfolding')
    parser.add_argument('--spaqr_dropout', type=float, default=0.0, help='SPAQR dropout rate')
    parser.add_argument('--spaqr_debug', type=int, default=0, help='SPAQR: 1 to print diagnostics for first batches')


    # Used to be compatible with ipython. Never used
    parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")

    configs = ExpConfigs(**vars(parser.parse_args(args))) # enable type hints

    # .yaml reference file maintainance
    yaml_configs_path_deprecated = Path(f"configs/{configs.model_name}/{configs.dataset_name}.yaml") # backward compatibility
    yaml_configs_path = Path(f"configs/{configs.model_name}/{configs.model_id}/{configs.dataset_name}.yaml")
    if yaml_configs_path.exists():
        with open(yaml_configs_path, 'r', encoding="utf-8") as stream:
            try:
                yaml_configs: dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"utils/configs.py: Exception when parsing {yaml_configs_path}: {exc}")
                exit(1)
        if yaml_configs is not None:
            # update yaml only if new args are added in argparse
            if_update = False
            for key, value in configs.__dict__.items():
                if key not in yaml_configs.keys():
                    if_update = True
                    yaml_configs[key] = value

            if if_update:
                with open(yaml_configs_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_configs, f, default_flow_style=False)
    else:
        Path(f"configs/{configs.model_name}/{configs.model_id}").mkdir(parents=True, exist_ok=True)
        if yaml_configs_path_deprecated.exists():
            # migrate from deprecated folder structure to new one
            yaml_configs_path_deprecated.replace(yaml_configs_path) # will overwrite if destination exists
        else:
            # save yaml if not exist
            with open(yaml_configs_path, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(configs), f, default_flow_style=False)
                
    return configs
