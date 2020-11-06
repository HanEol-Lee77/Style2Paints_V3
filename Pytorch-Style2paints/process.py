
from tensorboardX import SummaryWriter

from helper import *

import numpy as np
import cv2
from tqdm import tqdm
import glob
import os.path as osp
import random
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import matplotlib.pyplot as plt


def RandomCenterCrop(path, min_size, max_size):
    '''
    simulate dataset step 1: Crop Randomly
    '''
    size = np.random.randint(min_size, max_size)
    
    img = cv2.imread(path)
    h, w, _ = img.shape

    top = np.random.randint(0, h - size)
    left = np.random.randint(0, w - size)

    return img[top:size+top, left:size+left, :]


def get_patch(path, min_patch_size, max_patch_size):
    '''
    get patch from clothes
    '''
    patch_size = np.random.randint(min_patch_size, max_patch_size)
    
    img = cv2.imread(path)
    h, w, _ = img.shape
    
    center_h = h/2
    center_w = w/2
    
    patch = img[int(center_h - patch_size/2):int(center_h + patch_size/2), int(center_w - patch_size/2):int(center_w + patch_size/2), :]
    
    return patch


def edge_detecton(path):
    '''
    get sketch
    '''
    from_mat = cv2.imread(path)
    width = float(from_mat.shape[1])
    height = float(from_mat.shape[0])
    new_width = 0
    new_height = 0
    if (width > height):
        from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512
    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])
    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)
    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    #sketchKeras_colored = show_active_img_and_save('sketchKeras_colored', line_mat, 'sketchKeras_colored.jpg')
    line_mat = np.amax(line_mat, 2)
    #sketchKeras_enhanced = show_active_img_and_save_denoise_filter2('sketchKeras_enhanced', line_mat, 'sketchKeras_enhanced.jpg')
    #sketchKeras_pured = show_active_img_and_save_denoise_filter('sketchKeras_pured', line_mat, 'sketchKeras_pured.jpg')
    sketchKeras = show_active_img_and_save_denoise('sketchKeras', line_mat, 'sketchKeras.jpg')
    cv2.waitKey(0)
    return sketchKeras


def get_mask(path):
    '''
    提取衣服的mask
    返回numpy数组
    '''
    from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map
    from linefiller.thinning import thinning

    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ret, binary = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)

    fills = []
    result = binary

    fill = trapped_ball_fill_multi(result, 3, method='max')
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 2, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = flood_fill_multi(result)
    fills += fill

    fillmap = build_fill_map(result, fills)

    fillmap = merge_fill(fillmap)


    for i in range(len(fillmap[:,0])):
        for j in range(len(fillmap[0,:])):
            if fillmap[i,j] == 1:
                fillmap[i,j] = 0
            else:
                fillmap[i,j] = 1
    
    return fillmap


from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map
from linefiller.thinning import thinning

def get_region_picture(path):
    '''
    获取不规则形状的图片，背景是黑色0，方便rotate
    '''
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ret, binary = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)

    fills = []
    result = binary

    fill = trapped_ball_fill_multi(result, 3, method='max')
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 2, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = flood_fill_multi(result)
    fills += fill

    fillmap = build_fill_map(result, fills)

    fillmap = merge_fill(fillmap)

    fillmap = thinning(fillmap)

    #获得region mask
    for i in range(len(fillmap[:,0])):
        for j in range(len(fillmap[0,:])):
            if fillmap[i,j] == 0:
                fillmap[i,j] = 1
            else:
                fillmap[i,j] = 0
    #获得region picture    
    im = cv2.imread(path)
#     plt.imshow(im)
    rgb_fillmap = np.zeros(im.shape)
    rgb_fillmap[:,:,0] = fillmap
    rgb_fillmap[:,:,1] = fillmap
    rgb_fillmap[:,:,2] = fillmap
    im = im * rgb_fillmap
    
    return im.astype('uint8')

def Random_paste_patch_img(ori_img, patch_img):

    paste_x = np.random.randint(0, ori_img.size[0] - patch_img.size[0])
    paste_y = np.random.randint(0, ori_img.size[1] - patch_img.size[1])
    rotate_angle = np.random.randint(1, 359)
    resize_x = np.random.randint(64, 384)
    resize_y = np.random.randint(64, 384)
    patch_img = patch_img.resize((resize_x,resize_y))
    tem = ori_img.copy()
    tem.paste(patch_img.rotate(rotate_angle),(paste_x,paste_y))
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


def get_STL(path, num_batch):
    h = 1000
    w = 700
    im = cv2.imread(path[0])
    im = im / 255.
#     h = im.shape[0]
#     w = im.shape[1]
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
    
    im = im.reshape(1, h, w, 3)
    im = im.astype('float32')
    
    batch = np.append(im, im, axis=0)
    for p in path: 
        im = cv2.imread(p)
        im = im / 255.
    #     h = im.shape[0]
    #     w = im.shape[1]
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
        im = im.reshape(1, h, w, 3)
        im = im.astype('float32')
        batch = np.append(batch, im, axis=0)
    
#     print(batch.shape)
    batch = batch[2:,:,:,:]
#     print(batch.shape)

    out_size = (h, w)

    # %% Simulate batch
#     batch = np.append(im, im, axis=0)
    # batch.append(im)
    # batch = np.append(batch, im, axis=0)
#     num_batch = 1

    x = tf.placeholder(tf.float32, [None, h, w, 3])
    x = tf.cast(batch, 'float32')

    # %% Create localisation network and convolutional layer
    with tf.variable_scope('spatial_transformer_0'):

        # %% Create a fully-connected layer with 6 output nodes
        n_fc = 6
        W_fc1 = tf.Variable(tf.zeros([h * w * 3, n_fc]), name='W_fc1')

        # %% Zoom into the image
        a = np.random.randint(5, 10)/10
        b = np.random.randint(0, 3)/10
        c = np.random.randint(0, 3)/10
        d = np.random.randint(5, 10)/10 
#         initial = np.array([[s, 0, tx], [0, s,ty]])
        initial = np.array([[a, b, 0], [b, d, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()

        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(tf.zeros([num_batch, h * w * 3]), W_fc1) + b_fc1
        h_trans = transformer(x, h_fc1, out_size)

    # %% Run session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y = sess.run(h_trans, feed_dict={x: batch})
#     y = batch
    
    return y


#提取图片主要色
import colorsys
 
def get_dominant_color(image):
    
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


def min_dis(point, point_list):
    dis = []
    for p in point_list:
        dis.append(np.sqrt(np.sum(np.square(np.array(point)-np.array(p)))))
    
    return min(dis) 

    
import cv2
import numpy as np


def get_ball_structuring_element(radius):
    """Get a ball shape structuring element with specific radius for morphology operation.
    The radius of ball usually equals to (leaking_gap_size / 2).
    
    # Arguments
        radius: radius of ball shape.
             
    # Returns
        an array of ball structuring element.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def get_unfilled_point(image):
    """Get points belong to unfilled(value==255) area.

    # Arguments
        image: an image.

    # Returns
        an array of points.
    """
    y, x = np.where(image == 255)

    return np.stack((x.astype(int), y.astype(int)), axis=-1)


def exclude_area(image, radius):
    """Perform erosion on image to exclude points near the boundary.
    We want to pick part using floodfill from the seed point after dilation. 
    When the seed point is near boundary, it might not stay in the fill, and would
    not be a valid point for next floodfill operation. So we ignore these points with erosion.

    # Arguments
        image: an image.
        radius: radius of ball shape.

    # Returns
        an image after dilation.
    """
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, get_ball_structuring_element(radius), anchor=(-1, -1), iterations=1)


def trapped_ball_fill_single(image, seed_point, radius):
    """Perform a single trapped ball fill operation.

    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
        radius: radius of ball shape.
    # Returns
        an image after filling.
    """
    ball = get_ball_structuring_element(radius)

    pass1 = np.full(image.shape, 255, np.uint8)
    pass2 = np.full(image.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(image)

    # Floodfill the image
    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    # Perform dilation on image. The fill areas between gaps became disconnected.
    pass1 = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, ball, anchor=(-1, -1), iterations=1)
    mask2 = cv2.copyMakeBorder(pass1, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

    # Floodfill with seed point again to select one fill area.
    _, pass2, _, rect = cv2.floodFill(pass2, mask2, seed_point, 0, 0, 0, 4)
    # Perform erosion on the fill result leaking-proof fill.
    pass2 = cv2.morphologyEx(pass2, cv2.MORPH_ERODE, ball, anchor=(-1, -1), iterations=1)

    return pass2


def trapped_ball_fill_multi(image, radius, method='mean', max_iter=1000):
    """Perform multi trapped ball fill operations until all valid areas are filled.

    # Arguments
        image: an image. The image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        radius: radius of ball shape.
        method: method for filtering the fills. 
               'max' is usually with large radius for select large area such as background.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    # print('trapped-ball ' + str(radius))
 
    unfill_area = image
    filled_area, filled_area_size, result = [], [], []

    for _ in range(max_iter):
        points = get_unfilled_point(exclude_area(unfill_area, radius))

        if not len(points) > 0:
            break

        fill = trapped_ball_fill_single(unfill_area, (points[0][0], points[0][1]), radius)
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))
        filled_area_size.append(len(np.where(fill == 0)[0]))

    filled_area_size = np.asarray(filled_area_size)

    if method == 'max':
        area_size_filter = np.max(filled_area_size)
    elif method == 'median':
        area_size_filter = np.median(filled_area_size)
    elif method == 'mean':
        area_size_filter = np.mean(filled_area_size)
    else:
        area_size_filter = 0

    result_idx = np.where(filled_area_size >= area_size_filter)[0]

    for i in result_idx:
        result.append(filled_area[i])

    return result


def flood_fill_single(im, seed_point):
    """Perform a single flood fill operation.

    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
    # Returns
        an image after filling.
    """
    pass1 = np.full(im.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(im)

    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    return pass1


def flood_fill_multi(image, max_iter=20000):
    """Perform multi flood fill operations until all valid areas are filled.
    This operation will fill all rest areas, which may result large amount of fills.

    # Arguments
        image: an image. the image should contain white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    # print('floodfill')

    unfill_area = image
    filled_area = []

    for _ in range(max_iter):
        points = get_unfilled_point(unfill_area)

        if not len(points) > 0:
            break

        fill = flood_fill_single(unfill_area, (points[0][0], points[0][1]))
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))

    return filled_area


def mark_fill(image, fills):
    """Mark filled areas with 0.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    result = image.copy()

    for fill in fills:
        result[fill] = 0

    return result


def build_fill_map(image, fills):
    """Make an image(array) with each pixel(element) marked with fills' id. id of line is 0.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an array.
    """
    result = np.zeros(image.shape[:2], np.int)

    for index, fill in enumerate(fills):
        result[fill] = index + 1

    return result


def show_fill_map(fillmap):
    """Mark filled areas with colors. It is useful for visualization.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    # Generate color for each fill randomly.
    colors = np.random.randint(0, 255, (np.max(fillmap) + 1, 3))
    # Id of line is 0, and its color is black.
    colors[0] = [0, 0, 0]

    return colors[fillmap]


def get_bounding_rect(points):
    """Get a bounding rect of points.

    # Arguments
        points: array of points.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = np.min(points[1]), np.min(points[0]), np.max(points[1]), np.max(points[0])
    return x1, y1, x2, y2


def get_border_bounding_rect(h, w, p1, p2, r):
    """Get a valid bounding rect in the image with border of specific size.

    # Arguments
        h: image max height.
        w: image max width.
        p1: start point of rect.
        p2: end point of rect.
        r: border radius.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

    x1 = x1 - r if 0 < x1 - r else 0
    y1 = y1 - r if 0 < y1 - r else 0
    x2 = x2 + r + 1 if x2 + r + 1 < w else w
    y2 = y2 + r + 1 if y2 + r + 1 < h else h

    return x1, y1, x2, y2


def get_border_point(points, rect, max_height, max_width):
    """Get border points of a fill area

    # Arguments
        points: points of fill .
        rect: bounding rect of fill.
        max_height: image max height.
        max_width: image max width.
    # Returns
        points , convex shape of points
    """
    # Get a local bounding rect.
    border_rect = get_border_bounding_rect(max_height, max_width, rect[:2], rect[2:], 2)

    # Get fill in rect.
    fill = np.zeros((border_rect[3] - border_rect[1], border_rect[2] - border_rect[0]), np.uint8)
    # Move points to the rect.
    fill[(points[0] - border_rect[1], points[1] - border_rect[0])] = 255

    # Get shape.
    _, contours, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_shape = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)

    # Get border pixel.
    # Structuring element in cross shape is used instead of box to get 4-connected border.
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    border_pixel_mask = cv2.morphologyEx(fill, cv2.MORPH_DILATE, cross, anchor=(-1, -1), iterations=1) - fill
    border_pixel_points = np.where(border_pixel_mask == 255)

    # Transform points back to fillmap.
    border_pixel_points = (border_pixel_points[0] + border_rect[1], border_pixel_points[1] + border_rect[0])

    return border_pixel_points, approx_shape


def merge_fill(fillmap, max_iter=10):
    """Merge fill areas.

    # Arguments
        fillmap: an image.
        max_iter: max iteration number.
    # Returns
        an image.
    """
    max_height, max_width = fillmap.shape[:2]
    result = fillmap.copy()

    for i in range(max_iter):
        # print('merge ' + str(i + 1))

        result[np.where(fillmap == 0)] = 0

        fill_id = np.unique(result.flatten())
        fills = []

        for j in fill_id:
            point = np.where(result == j)

            fills.append({
                'id': j,
                'point': point,
                'area': len(point[0]),
                'rect': get_bounding_rect(point)
            })

        for j, f in enumerate(fills):
            # ignore lines
            if f['id'] == 0:
                continue

            border_points, approx_shape = get_border_point(f['point'], f['rect'], max_height, max_width)
            border_pixels = result[border_points]
            pixel_ids, counts = np.unique(border_pixels, return_counts=True)

            ids = pixel_ids[np.nonzero(pixel_ids)]
            new_id = f['id']
            if len(ids) == 0:
                # points with lines around color change to line color
                # regions surrounded by line remain the same
                if f['area'] < 5:
                    new_id = 0
            else:
                # region id may be set to region with largest contact
                new_id = ids[0]

            # a point
            if len(approx_shape) == 1 or f['area'] == 1:
                result[f['point']] = new_id

            #
            if len(approx_shape) in [2, 3, 4, 5] and f['area'] < 500:
                result[f['point']] = new_id

            if f['area'] < 250 and len(ids) == 1:
                result[f['point']] = new_id

            if f['area'] < 50:
                result[f['point']] = new_id

        if len(fill_id) == len(np.unique(result.flatten())):
            break

    return result
