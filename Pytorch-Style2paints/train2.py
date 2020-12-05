# Amend Code (by Haneol-Lee77)
import argparse
import configparser
import os
import yaml
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from utils import create_dir
from models import get_model
from experiments import experiment

# coding: utf-8
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import sys
import time
from optparse import OptionParser
import numpy as np
import glob  
import os.path as osp
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils

from transforms import GroupRandomCrop
from transforms import GroupScale

from eval import eval_net
from unet import UNet
from unet import Discriminator
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
import easydict

from dataset_multi import ClothDataSet
##
from tensorboardX import SummaryWriter
writer = SummaryWriter()

    # dataset, # ========
    # phase, # ========
    # batch_size,
    # workers=8, 
    # input_height=256, # 해결
    # input_width=256,
    # processed_dir='/home/userB/junsulee/youngin/resources/processed'

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

def get_argments():
    config, config_args, remaining_argv = load_config_file()
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', type=str, choices=['train', 'test'])
    parser.add_argument('--task', type=str, choices=['sketch2color'])
    parser.add_argument('--experiment', type=str, default='psnr')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--processed_dir', type=str, default='/home/ubuntu/data/processed')

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

# parser = argparse.ArgumentParser() 의 parser.add_argument로 대체
# args = easydict.EasyDict({
#     'epochs' : 100,
#     'batch_size' : 16,
#     # ============================= #
#     'train_path' : 'train data path'#'/data4/wangpengxiao/danbooru2017/original/train',
#     'val_path' : 'val data path'#'/data4/wangpengxiao/danbooru2017/original/val',
#     'sketch_path' : 'sketch path'#"/data4/wangpengxiao/danbooru2017/original_sketch",
#     'draft_path' : 'STL path'#"/data4/wangpengxiao/danbooru2017/original_STL",
#     'save_path' : 'result path' # save_dir  #"/data4/wangpengxiao/danbooru2017/result" ,
#     # ============================= #
#     # 'img_size' : 270, # train, val_loader에만 쓰임
#     'img_width': 256,
#     'img_height': 256,
#     're_size' : 256,  # train, val_loader에만 쓰임 # XXXXXX 삭제
#     'learning_rate' : 1e-5, # changed from original paper
#     'gpus' : '[0,1,2,3]',
#     'lr_steps' : [5, 10, 15, 20],
#     "lr_decay" : 0.1,
#     'lamda_L1' : 0.01, # changed from original paper
#     'workers' : 16, # richard zhang에서는 8
#     'weight_decay' : 1e-4
# })


Unet = UNet(in_channels=4, out_channels=3)
D = Discriminator(in_channels=3, out_channels=1)


writer.add_graph(Unet, (Variable(torch.randn(1,2,4,256,256), requires_grad=True)[0], Variable(torch.randn(1,2,3,224,224), requires_grad=True)[0]))

Unet = torch.nn.DataParallel(Unet, device_ids=eval(args.gpus)).cuda()

D = torch.nn.DataParallel(D, device_ids=eval(args.gpus)).cuda()

cudnn.benchmark = True # faster convolutions, but more memory

#####################################################################
from dataloader import get_dataloader
train_loader =
# torch.utils.data.DataLoader
    get_dataloader(
        dataset=args.dataset,
        phase=args.phase,
        workers=args.workers,
        input_height=args.input_height,
        input_width=args.input_width,
        batch_size=args.batch_size,
        # shuffle=True, # 뒤에 파일에 이미 있음
        # num_workers=args.workers, # 중복
        pin_memory=True
    )

    # dataset,
    # phase,
    # batch_size,
    # workers=8,
    # input_height=256,
    # input_width=256,
    # # processed_dir='/home/userB/junsulee/youngin/resources/processed' # 데이터를 채워줘야 하는 부분** #$#$
    # processed_dir='/home/ubuntu/data/processed/processed_yumi' # 데이터를 채워줘야 하는 부분** #$#$
    # ):
    # """
    # dataset: the name of dataset. ex) 'yumi', 'celeba', 'tag2pix'
    # phase: use 'train' for training, 'val' for validation, 'test' for testing
    # batch_size: the size of batch
    # workers: the number of workers used for making batch
    # input_height: the height of input image. Do not touch!
    # input_width: the width of input image. Do not touch!
    # processed_dir: directory which contains datasets. You do not need to change it as long as working in korea university server.
    # """

    # assert phase in ['train', 'val', 'test']

    # dataset = Sketch2ColorDataset(dataset, phase, input_height, input_width, processed_dir)

    # if phase == 'train':
    #     return DataLoader(
    #         dataset=dataset,
    #         num_workers=workers,
    #         batch_size=batch_size,
    #         shuffle=True
    #     )
    # elif phase == 'val': 
    #     return DataLoader(
    #         dataset=dataset,
    #         num_workers=workers,
    #         batch_size=batch_size,
    #         shuffle=False
    #     )
    # else:
    #     return DataLoader(
    #         dataset=dataset,
    #         num_workers=workers,
    #         batch_size=batch_size,
    #         shuffle=False
    #     )


val_loader = get_dataloader(
        dataset=args.dataset,
        phase=args.phase,
        workers=args.workers,
        input_height=args.input_height,
        input_width=args.input_width,
        batch_size=args.batch_size,
        # shuffle=True, # 뒤에 파일에 이미 있음
        # num_workers=args.workers, # 중복
        pin_memory=True
    )

# train_loader = torch.utils.data.DataLoader(
#     ClothDataSet(
#         args.train_path,
#         args.sketch_path,
#         args.draft_path,
#         args.img_width,
#         args.img_height,
#         args.re_size,
#         is_train = True
#         ),
#     batch_size=args.batch_size, shuffle=True,
#     num_workers=args.workers, pin_memory=True)

# val_loader = torch.utils.data.DataLoader(
#     ClothDataSet(
#         args.val_path,
#         args.sketch_path,
#         args.draft_path,
#         args.img_width,
#         args.img_height,
#         args.re_size,
#         is_train = False
#         ),
#     batch_size=int(args.batch_size/4), shuffle=False,
#     num_workers=args.workers, pin_memory=True)
#####################################################################

G_optimizer = torch.optim.Adam(
    Unet.parameters(),
    weight_decay=args.weight_decay,
    lr = args.learning_rate,
    betas=(0.9, 0.99),
    eps=0.001)

D_optimizer = torch.optim.Adam(
    D.parameters(),
    weight_decay=args.weight_decay,
    lr = args.learning_rate,
    betas=(0.9, 0.99),
    eps=0.001)



criterionD = torch.nn.BCEWithLogitsLoss().cuda()
criterionG = torch.nn.L1Loss().cuda()



SAVE_FREQ = 40
PRINT_FREQ = 20
VAL_NUM = 30
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[ ]:


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.learning_rate * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = wd
    return lr


# In[ ]:


def train_net(args, train_loader, Unet, D, epoch, save_epoch, last_count_train, cur_lr):
    batch_time = AverageMeter()
    lossg = AverageMeter()
    lossd = AverageMeter()

    end = time.time()
    
    
    
    for i, (input, df, gt) in enumerate(train_loader):
        #import pdb;pdb.set_trace()
        input=input.float()
        df=df.float()
        gt=gt.float()
        input.size
        df.size

        input = Variable(input)
        df = Variable(df)
        gt = Variable(gt)
        label = Variable(torch.ones(input.size(0),int(gt.size(2)/8),int(gt.size(3)/8))) # 1 for real
        
        input = input.cuda()
        df = df.cuda()
        gt = gt.cuda()
        # ----- train netd -----
        D.zero_grad()   
        ## train netd with real img
        #import pdb;pdb.set_trace()
        output=D(gt)
        error_real=criterionD(output.squeeze(),label.cuda().squeeze())
        ## train netd with fake img
        fake_pic=Unet(input,df)
        output2=D(fake_pic)
        label.data.fill_(0) # 0 for fake
        error_fake=criterionD(output2.squeeze(),label.cuda().squeeze())
        error_D=(error_real + error_fake)*0.5
        error_D.backward()
        D_optimizer.step()

        # ------ train netg -------
        Unet.zero_grad()
        label.data.fill_(1)
        #import pdb;pdb.set_trace()
        fake_pic = Unet(input,df)
        output = D(fake_pic)
        error_G = criterionD(output.squeeze(),label.cuda().squeeze())
        error_L1 = criterionG(fake_pic.cuda(),gt.cuda())
        error_G = error_G*args.lamda_L1 + error_L1
#         error_G.backward(retain_graph=True)
        
        error_G.backward()
        G_optimizer.step()
        
        lossg.update(error_G.item(), input.size(0))
        lossd.update(error_D.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        last_count_train += 1
        writer.add_scalar('Tdata/batch_time_val', batch_time.val, last_count_train)
        writer.add_scalar('Tdata/batch_time_avg', batch_time.avg, last_count_train)
        writer.add_scalar('Tdata/lossG_val', lossg.val, last_count_train)
        writer.add_scalar('Tdata/lossG_avg', lossg.avg, last_count_train)  
        writer.add_scalar('Tdata/lossD_val', lossd.val, last_count_train)
        writer.add_scalar('Tdata/lossD_avg', lossd.avg, last_count_train)   
        
        
        
        if i % PRINT_FREQ == 0:
            tb_view_pic(input, df, gt, fake_pic)
            #print('max = '+str(output.max().item())+'   '+'min = '+str(output.min().item()))
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'LossG {lossG.val:.4f} ({lossG.avg:.4f})\t'
                   'LossD {lossD.val:.4f} ({lossD.avg:.4f})'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       lossG=lossg,
                       lossD=lossd,
                       lr=cur_lr)))
        #save Unet
    torch.save({
                'epoch': epoch,
                'arch': 'Unet',
                'state_dict': Unet.state_dict(),
            }, osp.join(save_epoch,str(epoch)+'_Unet_'+'checkpoint.pth.tar'))
    #save D
    torch.save({
                'epoch': epoch,
                'arch': 'D',
                'state_dict': D.state_dict(),
            }, osp.join(save_epoch,str(epoch)+'_D_'+'checkpoint.pth.tar'))
    
    return last_count_train


# In[ ]:


def val_net(args, val_loader, Unet, D, epoch, save_epoch, last_count_val):
    batch_time_val = AverageMeter()
    lossg_val = AverageMeter()
    lossd_val = AverageMeter()
    
    Unet.eval()
    
    end = time.time()
    for i, (input, df, gt) in enumerate(val_loader):
        if i >= int(VAL_NUM):
            break
        input=input.float()
        df=df.float()
        gt=gt.float()
        with torch.no_grad():            
            input_var = input
            df_var = df
            gt_var = gt
            label = torch.ones(input.size(0),int(gt.size(2)/8),int(gt.size(3)/8)) # 1 for real

#         input_var = input_var.cuda()
#         df_var = df_var.cuda()
#         gt_var = gt_var.cuda()

        # ------ val netg -------
        label.data.fill_(1)
        fake_pic = Unet(input_var,df_var)
        output = D(fake_pic)
        error_GAN_G = criterionD(output.squeeze(),label.cuda().squeeze())
        error_L1 = criterionG(fake_pic.cuda(),gt_var.cuda())
        error_G = error_GAN_G*args.lamda_L1 + error_L1
        
        lossg_val.update(error_G.item(), input.size(0))
        batch_time_val.update(time.time() - end)
        end = time.time()
        
        save_pic(save_epoch, i, input_var, df_var, gt_var, fake_pic)
        
        last_count_val += 1
        writer.add_scalar('Vdata/batch_time_val', batch_time_val.val, last_count_val)
        writer.add_scalar('Vdata/batch_time_avg', batch_time_val.avg, last_count_val)
        writer.add_scalar('Vdata/lossG_val', lossg_val.val, last_count_val)
        writer.add_scalar('Vdata/lossG_avg', lossg_val.avg, last_count_val)  
        if i % PRINT_FREQ == 0:
            #print('max = '+str(output.max().item())+'   '+'min = '+str(output.min().item()))
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'LossG {lossG.val:.4f} ({lossG.avg:.4f})'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time_val,
                       lossG=lossg_val,
                       lr=args.learning_rate)))
#     return np.transpose(fake_pic.detach().cpu().numpy(), (0,2,3,1))
    return last_count_val


# In[ ]:


def mkdir(path):
    
    isExists=os.path.exists(path)
    
    if not isExists:
        os.makedirs(path) 

        return True
    else:
        print (path+' 目录已存在')
        return False


# In[ ]:


def tb_view_pic(input, df, gt, fake_pic):
    
    sketch = input[:,0,:,:].view(input[:,0,:,:].shape[0],1,input[:,0,:,:].shape[1],input[:,0,:,:].shape[2])
    point_map = input[:,1:,:,:]
    draft = df
    fake = fake_pic
    ground_truth = gt
    
    sketch = vutils.make_grid(sketch, normalize=True, scale_each=True)
    point_map = vutils.make_grid(point_map, normalize=True, scale_each=True)
    draft = vutils.make_grid(draft, normalize=True, scale_each=True)
    fake = vutils.make_grid(fake, normalize=True, scale_each=True)
    ground_truth = vutils.make_grid(ground_truth, normalize=True, scale_each=True)
    
    writer.add_image('sketch', sketch, 1)
    writer.add_image('point_map', point_map, 2)
    writer.add_image('draft', draft, 3)
    writer.add_image('fake', fake, 4)
    writer.add_image('ground_truth', ground_truth, 5)
    
    return 


# In[ ]:


def save_image(image_path, image_numpy):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    
    
def save_pic(save_epoch, i, input_var, df_var, gt_var, fake_pic):
    fake_pic = (np.transpose(fake_pic.detach().cpu().numpy(), (0,2,3,1))+1) / 2.0 * 255.0
    point_map = (np.transpose(input_var.detach().cpu().numpy(), (0,2,3,1))[:,:,:,1:]+1) / 2.0 * 255.0
    sketch = (np.transpose(input_var.detach().cpu().numpy(), (0,2,3,1))[:,:,:,0]+1) / 2.0 * 255.0
    df_var = (np.transpose(df_var.detach().cpu().numpy(), (0,2,3,1))+1) / 2.0 * 255.0
    gt_var = (np.transpose(gt_var.detach().cpu().numpy(), (0,2,3,1))+1) / 2.0 * 255.0
    
    p = osp.join(save_epoch, str(i))
    mkdir(p)
    
    for j in range(len(fake_pic[:,0,0,0])):
        save_image(osp.join(p, str(j)+'fake.jpg'),fake_pic[j].astype('uint8'))
        save_image(osp.join(p, str(j)+'input_sketch.jpg'),sketch[j].astype('uint8'))
        save_image(osp.join(p, str(j)+'input_pointmap.jpg'),point_map[j].astype('uint8'))
        save_image(osp.join(p, str(j)+'df.jpg'),df_var[j].astype('uint8'))
        save_image(osp.join(p, str(j)+'gt.jpg'),gt_var[j].astype('uint8'))


# In[ ]:


last_count_train = 0
last_count_val = 0

for epoch in range(args.epochs):
    
    cur_lr = adjust_learning_rate(G_optimizer, epoch, args.lr_steps, args.lr_decay)
    cur_lr = adjust_learning_rate(D_optimizer, epoch, args.lr_steps, args.lr_decay)
    
    save_epoch = osp.join(args.save_path,str(epoch))
    mkdir(save_epoch)
    
      
    last_count_train = train_net(args, train_loader, Unet, D, epoch, save_epoch, last_count_train, cur_lr)
    last_count_val = val_net(args, val_loader, Unet, D, epoch, save_epoch, last_count_val) 
     
#     for i in range(len(g_pic[:,0,0,0])-1):
#         cv2.imwrite(osp.join(save_epoch, str(i)+'.jpg'),g_pic[i])
writer.close()