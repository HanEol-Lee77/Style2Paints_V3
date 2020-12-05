import argparse
import configparser
import os
import yaml
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from utils import create_dir
from dataloader import get_dataloader
from models import get_model
from experiments import experiment


def load_config_file():
    cnf_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    cnf_parser.add_argument('--config_dir', type=str, default='./config')
    cnf_parser.add_argument('--config', type=str)
    args, remaining_argv = cnf_parser.parse_known_args()

    config_args = None
    if args.config:
        with open(args.config_dir + '/' + args.config) as fin:
            config_args = yaml.load(fin)

    return args.config, config_args, remaining_argv

def get_arguments():
    config, config_args, remaining_argv = load_config_file()
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', type=str, choices=['train', 'test'])
    parser.add_argument('--task', type=str, choices=['sketch2color', 'gray2color'])
    parser.add_argument('--experiment', type=str, default='psnr')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--processed_dir', type=str, default='/home/userB/junsulee/youngin/resources/processed')

    parser.add_argument('--distributed_backend', type=str, default='ddp_spawn')
    parser.add_argument('--gpus', type=int, nargs='+', default=[])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--save_period', type=int, default=1)

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--seed', type=int, default=42)

    # data loader
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--input_height', type=int, default=256)
    parser.add_argument('--input_width', type=int, default=256)

    # hint
    parser.add_argument('--stop_prob', type=float, default=0.125)
    parser.add_argument('--max_hints', type=int, default=30)

    # model
    parser.add_argument('--model', type=str, choices=['ask', 'user_guided'])

    # training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min_tau', type=float, default=1e-1)
    parser.add_argument('--decay_ratio', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--smoothness_method', type=str, choices=['four_l1', 'eight_l1', 'eight_l2'], default='eight_l2')
    parser.add_argument('--lambda_rec', type=float, default=30.0)
    parser.add_argument('--lambda_adv', type=float, default=1.0)
    parser.add_argument('--lambda_tvr', type=float, default=5.0)
    parser.add_argument('--lambda_smt', type=float, default=1.0)

    # testing
    parser.add_argument('--test_deterministic', type=bool, default=False)
    parser.add_argument('--test_samples', type=int, default=1)
    parser.add_argument('--test_example_steps', type=int, default=8)

    # only for AskModel
    parser.add_argument('--use_bilinear', type=bool, default=True)

    # only for user guided model
    parser.add_argument('--ug_sample_Ps', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='patch sizes')
    parser.add_argument('--ug_test_sample_Ps', type=int, nargs='+', default=[7], help='patch sizes')
    parser.add_argument('--ug_mask_cent', type=float, default=.5, help='mask centering factor')
    parser.add_argument('--ug_init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')

    # set args from config file
    if config_args:
        parser.set_defaults(**config_args)

    args = parser.parse_args(remaining_argv)
    args_dict = vars(args)

    if args.phase == 'train':
        working_dir = os.path.join(args.save_dir, 'train')
    else:
        working_dir = os.path.join(args.save_dir, 'exp_{0}'.format(args.experiment))
    
    print(working_dir)
    create_dir(working_dir)
    with open(os.path.join(working_dir, config if config else 'config.yml'), 'w') as fout:
        yaml.dump(args_dict, fout, default_flow_style=None)
    pprint(args_dict)

    # assigning constants to arguments
    args.working_dir = working_dir
    args.input_channels = 1
    if args.task == 'sketch2color':
        args.output_channels = 3
    elif args.task == 'gray2color':
        raise NotImplementedError("gray2color is not yet supported.")
        args.output_channels = 2
    
    if args.phase == 'test' and args.experiment == 'example':
        args.batch_size = 1
    
    return args
# check point  torch vision  몇 에퐄 부터 되는지
def main():
    # arguments
    args = get_arguments()

    # seed
    pl.seed_everything(args.seed)

    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)

def train(args):
    # model
    model = get_model(args)
    
    # dataloaders
    train_dataloader = get_dataloader(args.dataset, 'train', args.batch_size, args.workers, args.input_height, args.input_width, args.processed_dir)
    val_dataloader = get_dataloader(args.dataset, 'val', args.batch_size, args.workers, args.input_height, args.input_width, args.processed_dir)
    
    # logger
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.working_dir, 'logs'))

    # callback
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.working_dir, 'checkpoints', '{epoch:d}'),
        verbose=True,
        save_last=True,
        save_top_k=args.save_top_k,
        monitor='checkpoint_on',
        mode='max'
    )

    # trainer
    trainer_args = {
        'checkpoint_callback': checkpoint_callback,
        'logger': tb_logger,
        'max_epochs': args.epochs
    }
    if args.checkpoint is not None:
        trainer_args['resume_from_checkpoint'] = os.path.join(args.working_dir, 'checkpoints', args.checkpoint)
    if args.distributed_backend is not None:
        trainer_args['distributed_backend'] = args.distributed_backend
    if len(args.gpus) > 0:
        trainer_args['gpus'] = args.gpus
    
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_dataloader, val_dataloader)

def test(args):
    # model
    model = get_model(args)

    # dataloader
    test_dataloader = get_dataloader(args.dataset, 'test', args.batch_size, args.workers, args.input_height, args.input_width, args.processed_dir)

    experiment(args, model, test_dataloader)

if __name__=='__main__':
    main()