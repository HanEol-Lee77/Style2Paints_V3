"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

# coding: utf-8

# In[ ]:
#################=================== paths ===================#################
# region_picture 생성하는 부분이 있음
region_picture_path = "/home/ubuntu/data/processed/yumi/region_picture" # line 41 # region_patch_save_path
source_data_path = "/home/ubuntu/data/processed/yumi/train/color" # line 166 #"/data4/wangpengxiao/danbooru2017/original"
random_crop_path = "/home/ubuntu/data/processed/yumi/crop_picture" # line 181 #"/data4/wangpengxiao/zalando_random_crop"
patch_path = "/home/ubuntu/data/processed/yumi/patch_picture" # line 182 #"/data4/wangpengxiao/zalando_center_patch"
#    'train_path' : '/home/ubuntu/data/processed/yumi/train/color',#'/data4/wangpengxiao/danbooru2017/original/train',
#     'val_path' : '/home/ubuntu/data/processed/yumi/val/color',#'/data4/wangpengxiao/danbooru2017/original/val',
#     'sketch_path' : '/home/ubuntu/data/processed/yumi/train/sketch',#"/data4/wangpengxiao/danbooru2017/original_sketch",
#     'draft_path' : 'STL path',#"/data4/wangpengxiao/danbooru2017/original_STL",
#     'save_path' : '/home/ubuntu/data/save/yumi',#"/data4/wangpengxiao/danbooru2017/result" ,
#################=================== paths ===================#################




import numpy as np
import torch
import torch.utils.data as data

import os
import random
import glob  
import os.path as osp
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms




# #####################################################################################################################
# ######################################## (stage1 -> ) stage2 preprocess #############################################

# ####### dataset input ########

# # img_path에 있는 모든 이미지 이름 가져오기
# # img_paths = '/home/ubuntu/data/processed/yumi/train/color'

# source_img_path = glob.glob(osp.join(source_data_path,'*.jpg'))
# print(source_img_path)
# wait = input("PRESS ENTER TO CONTINUE.")

# source_img_path = sorted(source_img_path) # 한 번 정렬해주기 



# ### hint 생성부분?? => Richard Zhang 방식과 동일하게 진행! random.choice같은 함수를 쓰던데..
# # print(source_img_path)
# # wait = input("PRESS ENTER TO CONTINUE.")

# #####################################################################################################################
# ######################################## (stage1 -> ) stage2 preprocess #############################################
# import process as P
# import cv2
# min_size, max_size = 64, 255
# for p in source_img_path:
#     # min_size, max_size = np.random.randint(64,255,1)[0], np.random.randint(64,255,1)[0]
#     # if max_size < minsize: max_size, min_size = min_size, max_size
#     # elif max_size == min_size: 

# ########################################
# # 이미 트레인 된 걸 가지고.. 하는 논의


#     # 어떤 이미지들을 몇 장이나 crop 할 건지 찾기**
#     img_crop = P.RandomCenterCrop (p, min_size, max_size)
#     cv2.imwrite('./res/imag_CenterCrop.jpg',  img_crop)
# #####################################
# #====== region_picture구하기 =======#
# #####################################

# #####################################
# #====== Spatial Net구하기 =======#
# #####################################


#     img_patch = P.get_patch(p, 64, 256)
#     cv2.imwrite('./res/img_patch.jpg',  img_patch)
#     input("stop1")
#     # P.edge_detection(p)

#     # 이미지 샘플 저장하기
#     # P.get_region_picture(p)

#     # print(len(p))
#     # print(p)
#     # wait = input("PRESS ENTER TO CONTINUE.")

#     # gt = Image.open(p).convert('RGB')
#     # print(type(gt))

#     # print(np.array(gt).shape[0])

#     # print(np.array(gt).shape[1])

#     # wait = input("PRESS ENTER TO CONTINUE.")

# ######################################################################################

#     # sk = Image.open(osp.join('sketch_path', osp.basename(p))).convert('L')
#     # df = gt.copy()
#     # STL = Image.open(osp.joint(self.STL_path, osp.basename(p))).convet('RGB')
#     # df = make_draft(STL, refion_img_path)

# # p = self.gt_img[index]

# # gt = Image.open(p).convert('RGB')
# # sk = Image.open(osp.join(self._sketch_path, osp.basename(p))).convert('L')
# # #学习gt的上色
# # #df = gt.copy()
# # STL = Image.open(osp.join(self._STL_path, osp.basename(p))).convert('RGB')
# # df = make_draft(STL, self.region_img_path)

# # if self._is_train:
# #     gt = gt.resize((self._img_size, self._img_size), Image.BICUBIC)
# #     sk = sk.resize((self._img_size, self._img_size), Image.BICUBIC)
# # else:
# #     gt = gt.resize((self._re_size, self._re_size), Image.BICUBIC)
# #     sk = sk.resize((self._re_size, self._re_size), Image.BICUBIC)
# # df = df.resize((224, 224), Image.BICUBIC)

# # #make point map
# # gt = np.array(gt)
# # point_map = np.zeros(gt.shape)

# # #coordinate = np.where(np.sum(gt,axis=2) < np.sum(np.array([255,255,255])))

# # num_of_point = np.random.randint(0, 20)
# # x = random.sample(range(0,gt.shape[0]),num_of_point)
# # y = random.sample(range(0,gt.shape[1]),num_of_point)

# # for i in range(len(x)):  
# #     r,g,b = gt[x[i],y[i],:]
# #     cv2.circle(point_map,(y[i],x[i]),1,(int(r),int(g),int(b)),-1)  

# # #finish making point map
# # gt = Image.fromarray(gt)
# # point_map = Image.fromarray(point_map.astype('uint8'))

# # #transform sk,point_map,gt
# #     #to tensor
# # sk = transforms.ToTensor()(sk)
# # point_map = transforms.ToTensor()(point_map)
# # gt = transforms.ToTensor()(gt)
# #     #random crop
# # if self._is_train:
# #     # if img_size, re_size are 270, 256, 
# #     w_offset = random.randint(0, max(0, self._img_size - self._re_size - 1)) # max( 0, 270 - 256 - 1) = 270 - 256 - 1
# #     h_offset = random.randint(0, max(0, self._img_size - self._re_size - 1)) # 270 - 256 - 1

# #     sk = sk[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]
# #     point_map = point_map[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]
# #     gt = gt[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]

# #     #normalize
# # sk = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sk)
# # point_map = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(point_map)
# # gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt)
# #     #flip image
# # #         if self._is_train and random.random() < 0.5:
# # #             idx = [i for i in range(gt.size(2) - 1, -1, -1)]
# # #             idx = torch.LongTensor(idx)
# # #             sk = sk.index_select(2, idx)
# # #             point_map = point_map.index_select(2, idx)
# # #             gt = gt.index_select(2, idx)



# # input = torch.cat((sk,point_map),0)         
# # df = self._inception_transform(df)

# # return input, df, gt 

class ClothDataSet_draft(data.Dataset):
    def __init__(self, data_path, sketch_path, img_size, re_size, is_train):
        # region_picture_path = "/data4/wangpengxiao/zalando_region_picture" 
        # region_picture_path = "/home/ubuntu/data/processed/yumi/region_picture"
        # region_img_path = glob.glob(osp.join(region_picture_path,'*.jpg')) 
        # region_img_path = sorted(region_img_path)

        self._data_path = data_path
        self._sketch_path = sketch_path
        self._img_size = img_size
        self._re_size = re_size
        self._is_train = is_train
        
        self._get_ground_truth()
        # self._inception_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5),
        #                                      (0.5, 0.5, 0.5))
        #     ])

    def _get_ground_truth(self):

        gt_img = glob.glob(osp.join(self._data_path,'*.jpg'))
        gt_img += glob.glob(osp.join(self._data_path,'*.png'))
        self.gt_img = sorted(gt_img)

    def __getitem__(self, index):

        p = self.gt_img[index]

        gt = Image.open(p).convert('RGB')
        sk = Image.open(osp.join(self._sketch_path, osp.basename(p))).convert('L')
        #学习gt的上色
        #df = gt.copy()
        # STL = Image.open(os.path.osp.join(self._STL_path, os.path.osp.basename(p))).convert('RGB')
        # df = make_draft(STL, self.region_img_path) #$

        # train 이라면 제거**
        if self._is_train:
            gt = gt.resize((self._img_size, self._img_size), Image.BICUBIC)
            sk = sk.resize((self._img_size, self._img_size), Image.BICUBIC)
        else:
            gt = gt.resize((self._re_size, self._re_size), Image.BICUBIC)
            sk = sk.resize((self._re_size, self._re_size), Image.BICUBIC)
        # df = df.resize((224, 224), Image.BICUBIC) #$

        #make point map # 이게 힌트 만드는 부분?
        gt = np.array(gt)
        point_map = np.zeros(gt.shape) # 원본 사진이랑 동일하게 height, width 구성

        #coordinate = np.where(np.sum(gt,axis=2) < np.sum(np.array([255,255,255])))
        #################################===== 힌트를 추출하는 코드 ====##########################################
        # 뽑을 좌표의 개수를 뽑는 코드
        num_of_point = np.random.randint(0, 20) # 0~20
        # 임의의 좌표를 뽑는 코드
        x = random.sample(range(0,gt.shape[0]),num_of_point) #0 ~ height-1 : ex> 0~255 중에서  (0~20 개 포인트 찍음 ex> 5개)
        y = random.sample(range(0,gt.shape[1]),num_of_point) 
        # 좌표의 갯수 만큼.. 해당 좌표의 r,g,b 값을 가져온다.
        for i in range(len(x)):  
            r,g,b = gt[x[i],y[i],:] # x, y, 3 채널이므로.. x, y에 속하는 모든 3채널 R,G,B값을 리스트(nd array 형태로  바눤)
            cv2.circle(point_map,(y[i],x[i]),1,(int(r),int(g),int(b)),-1)  
        #################################===== 힌트를 추출하는 코드 ====##########################################

        #finish making point map
        gt = Image.fromarray(gt)
        point_map = Image.fromarray(point_map.astype('uint8'))

        #transform sk,point_map,gt
            #to tensor
        point_map = transforms.ToTensor()(point_map)
        gt = transforms.ToTensor()(gt)
        sk = transforms.ToTensor()(sk) # transforms.ToTensor() #$#$ ?

            #random crop
        if self._is_train:
            # if img_size, re_size are 270, 256, 
            w_offset = random.randint(0, max(0, self._img_size - self._re_size - 1)) # max( 0, 270 - 256 - 1) = 270 - 256 - 1
            h_offset = random.randint(0, max(0, self._img_size - self._re_size - 1)) # 270 - 256 - 1

            sk = sk[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]
            point_map = point_map[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]
            gt = gt[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]

            #normalize
        # print(sk.shape)
        # print(gt.shape)
        # print(point_map.shape)
        
        point_map = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(point_map) 
        gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt)
        sk = transforms.Normalize((0.5,), (0.5,))(sk) #$#$#$#$ 0.5 3개에서 1개로 수정! 

            #flip image
#         if self._is_train and random.random() < 0.5:
#             idx = [i for i in range(gt.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             sk = sk.index_select(2, idx)
#             point_map = point_map.index_select(2, idx)
#             gt = gt.index_select(2, idx)



        input = torch.cat((sk,point_map),0)         
        # df = self._inception_transform(df)

        return input, gt 

    def __len__(self):
        return len(self.gt_img)





################################################################################################
                    #train_path, sketch_path, draft_path,img_size, re_size, is_train = True 
class ClothDataSet(data.Dataset):
    def __init__(self, data_path, sketch_path, STL_path, img_size, re_size, is_train):
        # region_picture_path = "/data4/wangpengxiao/zalando_region_picture" 
        region_picture_path = "/home/ubuntu/data/processed/yumi/region_picture"
        region_img_path = glob.glob(osp.join(region_picture_path,'*.jpg')) 
        region_img_path = sorted(region_img_path)

        self._data_path = data_path
        self._sketch_path = sketch_path
        self._STL_path = STL_path
        self._img_size = img_size
        self._re_size = re_size
        self._is_train = is_train
        self.region_img_path = region_img_path

        self._get_ground_truth()
        # self._inception_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5),
        #                                      (0.5, 0.5, 0.5))
        #     ])

    def _get_ground_truth(self):

        gt_img = glob.glob(osp.join(self._data_path,'*.jpg'))
        gt_img += glob.glob(osp.join(self._data_path,'*.png'))
        self.gt_img = sorted(gt_img)



    def __getitem__(self, index):

        p = self.gt_img[index]

        gt = Image.open(p).convert('RGB')
        sk = Image.open(osp.join(self._sketch_path, osp.basename(p))).convert('L')
        #学习gt的上色
        #df = gt.copy()
        STL = Image.open(osp.join(self._STL_path, osp.basename(p))).convert('RGB')
        df = make_draft(STL, self.region_img_path)

        if self._is_train:
            gt = gt.resize((self._img_size, self._img_size), Image.BICUBIC)
            sk = sk.resize((self._img_size, self._img_size), Image.BICUBIC)
        else:
            gt = gt.resize((self._re_size, self._re_size), Image.BICUBIC)
            sk = sk.resize((self._re_size, self._re_size), Image.BICUBIC)
        df = df.resize((224, 224), Image.BICUBIC)

        #make point map
        gt = np.array(gt)
        point_map = np.zeros(gt.shape)

        #coordinate = np.where(np.sum(gt,axis=2) < np.sum(np.array([255,255,255])))
        
        num_of_point = np.random.randint(0, 20)
        x = random.sample(range(0,gt.shape[0]),num_of_point)
        y = random.sample(range(0,gt.shape[1]),num_of_point)

        for i in range(len(x)):  
            r,g,b = gt[x[i],y[i],:]
            cv2.circle(point_map,(y[i],x[i]),1,(int(r),int(g),int(b)),-1)  

        #finish making point map
        gt = Image.fromarray(gt)
        point_map = Image.fromarray(point_map.astype('uint8'))

        #transform sk,point_map,gt
            #to tensor
        sk = transforms.ToTensor()(sk)
        point_map = transforms.ToTensor()(point_map)
        gt = transforms.ToTensor()(gt)
            #random crop
        if self._is_train:
            # if img_size, re_size are 270, 256, 
            w_offset = random.randint(0, max(0, self._img_size - self._re_size - 1)) # max( 0, 270 - 256 - 1) = 270 - 256 - 1
            h_offset = random.randint(0, max(0, self._img_size - self._re_size - 1)) # 270 - 256 - 1

            sk = sk[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]
            point_map = point_map[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]
            gt = gt[:, h_offset:h_offset + self._re_size, w_offset:w_offset + self._re_size]

        print(sk.shape)
        print(gt.shape)
        print(point_map.shape)
        
            #normalize
        sk = transforms.Normalize((0.5,), (0.5,))(sk)
        point_map = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(point_map)
        gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt)
            #flip image
#         if self._is_train and random.random() < 0.5:
#             idx = [i for i in range(gt.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             sk = sk.index_select(2, idx)
#             point_map = point_map.index_select(2, idx)
#             gt = gt.index_select(2, idx)



        input = torch.cat((sk,point_map),0)         
        df = self._inception_transform(df)

        return input, df, gt 

    def __len__(self):
        return len(self.gt_img)


def make_draft(STL, region_img_path):
        #第一步：step2 of paper： STL
        ori_img = STL.copy() # STL이미지를 가져와서

        #第二步：step1 of paper： paste
        color = get_dominant_color(ori_img)#but get color first  # STL이미지에서 주요 컬러를 뽑고
        # (R, G, B) 대표 컬러값 추출

        region_num = np.random.randint(1, 3) # 그 중에서 region 숫자는 랜덤하게 3개의 숫자를 뽑는다.
        for i in range(region_num):
            region_img = Image.open(random.choice(region_img_path))
            ori_img = Random_paste_region_img(ori_img, region_img)

        #第三步: step3 of paper： spray
        img = np.array(ori_img)
        h = int(img.shape[0]/30)
        w = int(img.shape[1]/30)
        a_x = np.random.randint(0, h)
        a_y = np.random.randint(0, w)
        b_x = np.random.randint(0, h)
        b_y = np.random.randint(0, w)
        begin_point = np.array([min(a_x,b_x),a_y])
        end_point = np.array([max(a_x,b_x),b_y])
        tan = (begin_point[1] - end_point[1]) / (begin_point[0] - end_point[0]+0.001)


        center_point_list = []
        for i in range(begin_point[0],end_point[0]+1):
            a = i
            b = (i-begin_point[0])*tan + begin_point[1]
            center_point_list.append(np.array([int(a),int(b)]))
        center_point_list = np.array(center_point_list)    


        lamda = random.uniform(0.01, 10) #一个超参
        paper = np.zeros((h,w,3)) # RGB ..?
        mask = np.zeros((h,w)) # single channel
        center = [int(h/2),int(w/2)]
        paper[center[0],center[1],:] = color
        for i in range(h):
            for j in range(w):
                dis = min_dis([i, j],center_point_list)
                paper[i,j,:] = np.array(color)/np.exp(lamda*dis)#*lamda/dis
                mask[i,j] = np.array([255])/np.exp(lamda*dis)#*lamda/dis

        paper = (paper).astype('uint8')
        mask = (mask).astype('uint8')

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        im = cv2.resize(paper, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        imq = Image.fromarray(im)
        imp = ori_img.copy()

        imp.paste(imq,(0, 0, imp.size[0], imp.size[1]),mask = Image.fromarray(mask))

        return imp   



import colorsys
 
def get_dominant_color(image):#获取图片主要颜色
    
#颜色模式转换，以便输出rgb颜色值
    image = image.convert('RGBA')
    
#生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))
    
    max_score = 0#原来的代码此处为None
    dominant_color = 0#原来的代码此处为None，但运行出错，改为0以后 运行成功，原因在于在下面的 score > max_score的比较中，max_score的初始格式不定
    
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue
        
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
       
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
       
        y = (y - 16.0) / (235 - 16)
        
        # 忽略高亮色
        if y > 0.9:
            continue
            
        # 忽略白背景
        if ((r>230)&(g>230)&(b>230)):
            continue
        
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
        
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
    
    return dominant_color



def Random_paste_region_img(ori_img, region_img):

    paste_x = np.random.randint(0, ori_img.size[0])
    paste_y = np.random.randint(0, ori_img.size[1])
    rotate_angle = np.random.randint(1, 359)
    resize_x = np.random.randint(64, 384)
    resize_y = np.random.randint(64, 384)
    region_img = region_img.resize((resize_x,resize_y))
    tem = ori_img.copy()
    tem.paste(region_img.rotate(rotate_angle),(paste_x,paste_y))
    tem = np.array(tem)
    ori_img = np.array(ori_img)
#     for i in range(ori_img.shape[0]):
#         for j in range(ori_img.shape[1]):
#             if (tem[i,j,:] == np.array([0,0,0])).all():
#                 tem[i,j,:] = ori_img[i,j,:]
    coordinate = np.where(tem == np.array([0,0,0]))
    for i in range(len(coordinate[0])):
        tem[coordinate[0][i],coordinate[1][i],:] = ori_img[coordinate[0][i],coordinate[1][i],:]
    ori_img = np.array(tem)
    ori_img = Image.fromarray(ori_img)
#     plt.imshow(ori_img)
    
    return ori_img


def min_dis(point, point_list):
    dis = []
    for p in point_list:
        dis.append(np.sqrt(np.sum(np.square(np.array(point)-np.array(p)))))
    
    return min(dis) 

#########################################################################################

####### dataset preprocessing #######


# STL, region_img_path ????????????????//
# make_draft(STL, region_img_path)