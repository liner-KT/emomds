import random
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from scipy import ndimage,signal
from timm.models.layers import to_3tuple
# from PIL import Image,ImageFilter
# from scipy.interpolate import RegularGridInterpolator,Rbf
from matplotlib import pyplot as plt
# import tricubic
import cv2
from torchvision import transforms as tf
try:
    from datasets.autoaugment import ImageNetPolicy
except:
    from autoaugment import ImageNetPolicy

def load_nii_file(nii_image):
    image = sitk.ReadImage(nii_image)
    image_array = sitk.GetArrayFromImage(image)
    return image_array

def cut_3Dimage(image, patch_num):
    depth, width, height = image.shape
    item_width = int(width / patch_num)
    item_depth = int(depth / patch_num)
    item_height = int(height / patch_num)
    box_list = []
    # (front,behind,left,right,  upper, lower)
    for i in range(0, patch_num):  # 三重循环，生成n张图片基于原图的位置
        for j in range(0, patch_num):
            for k in range(0, patch_num):
                box = ( i*item_depth, (i+1)*item_depth, j * item_width, (j + 1) * item_width, k * item_height,  (k + 1) * item_height,)
                box_list.append(box)
    print(box_list)
    # image_list = [image.crop(box) for box in box_list]  # Image.crop(left, up, right, below)
    image_list = [image[box[0]:box[1],box[2]:box[3],box[4]:box[5]] for box in box_list]
    return np.array(image_list)
def resize3D(image, size, mode):
    size = to_3tuple(size)
    image = image.astype(np.float32)
    if mode == 'trilinear':
        image_s = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(image_s, size=size, mode='trilinear', align_corners=True).squeeze(0).unsqueeze(0)
        return x.cpu().numpy()
    # elif mode == 'tricubic':
    #     ip = tricubic.tricubic(list(image),list(image.shape))
    #     x = np.zeros(size,dtype=np.float32)
    #     z_ = np.linspace(0,image.shape[0]-1,num=size[0],endpoint=True)
    #     h_ = np.linspace(0,image.shape[1]-1,num=size[1],endpoint=True)
    #     w_ = np.linspace(0,image.shape[2]-1,num=size[2],endpoint=True)
    #     for i, z in enumerate(z_):
    #         for j, h in enumerate(h_):
    #             for k, w in enumerate(w_):
    #                 x[i][j][k] = ip.ip([z, h, w])

        return x
def resizePatch3D(image, size, mode):
    size = to_3tuple(size)
    image = image.astype(np.float32)
    if mode == 'trilinear':
        image_s = torch.from_numpy(image).unsqueeze(0)
        x = F.interpolate(image_s, size=size, mode='trilinear', align_corners=True).squeeze(0)
        return x.cpu().numpy()
    # elif mode == 'tricubic':
    #     ip = tricubic.tricubic(list(image),list(image.shape))
    #     x = np.zeros(size,dtype=np.float32)
    #     z_ = np.linspace(0,image.shape[0]-1,num=size[0],endpoint=True)
    #     h_ = np.linspace(0,image.shape[1]-1,num=size[1],endpoint=True)
    #     w_ = np.linspace(0,image.shape[2]-1,num=size[2],endpoint=True)
    #     for i, z in enumerate(z_):
    #         for j, h in enumerate(h_):
    #             for k, w in enumerate(w_):
    #                 x[i][j][k] = ip.ip([z, h, w])

        return x
def image_normalization(image, win=None, adaptive=True):
    if win is not None:
        image = 1. * (image - win[0]) / (win[1] - win[0])
        image[image < 0] = 0.
        image[image > 1] = 1.
        return image
    elif adaptive:
        min, max = np.min(image), np.max(image)
        image = (image - min) / (max - min)
        return image
    else:
        return image

def random_crop(image, crop_shape):
    crop_shape = to_3tuple(crop_shape)
    _, z_shape, y_shape, x_shape = image.shape
    z_min = np.random.randint(0, z_shape - crop_shape[0])
    y_min = np.random.randint(0, y_shape - crop_shape[1])
    x_min = np.random.randint(0, x_shape - crop_shape[2])
    image = image[..., z_min:z_min+crop_shape[0], y_min:y_min+crop_shape[1], x_min:x_min+crop_shape[2]]
    return image

def center_crop(image, target_shape=(10, 80, 80)):
    target_shape = to_3tuple(target_shape)
    b, z_shape, y_shape, x_shape = image.shape
    z_min = z_shape // 2 - target_shape[0] // 2
    y_min = y_shape // 2 - target_shape[1] // 2
    x_min = x_shape // 2 - target_shape[2] // 2
    image = image[:, z_min:z_min+target_shape[0], y_min:y_min+target_shape[1], x_min:x_min+target_shape[2]]
    return image

def randomflip_z(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, ::-1, ...]

def randomflip_x(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[..., ::-1]

def randomflip_y(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, :, ::-1, ...]

def random_flip(image, mode='x', p=0.5):
    if mode == 'x':
        image = randomflip_x(image, p=p)
    elif mode == 'y':
        image = randomflip_y(image, p=p)
    elif mode == 'z':
        image = randomflip_z(image, p=p)
    else:
        raise NotImplementedError(f'Unknown flip mode ({mode})')
    return image

def rotate(image, angle=10):
    angle = random.randint(-10, 10)
    r_image = ndimage.rotate(image, angle=angle, axes=(-2, -1), reshape=True)
    if r_image.shape != image.shape:
        r_image = center_crop(r_image, target_shape=image.shape[1:])
    return r_image

#边缘检测
def edge(image):
    T,Z,H,W = image.shape
    seed1=random.random()
    #[0.8,0.1,0.1]
    if seed1>0.2:
        edge_mode=cv2.BORDER_REFLECT
    elif seed1>0.1:
        edge_mode=cv2.BORDER_REPLICATE
    elif seed1>0.:
        edge_mode=cv2.BORDER_CONSTANT

    image_ls=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_seg = image[i,j,:,:]
            sobelx = cv2.Sobel(img_seg, cv2.CV_64F, 1, 0, ksize=3,borderType=edge_mode)
            sobely = cv2.Sobel(img_seg, cv2.CV_64F, 0, 1, ksize=3,borderType=edge_mode)
            gradient_img = np.sqrt(sobelx**2 + sobely**2)
            image_ls.append(np.asarray(gradient_img)[None, ...])
    image_ls=np.concatenate(image_ls, axis=0)
    image=image_ls.reshape(T,Z,H,W)
    image=image.astype(np.float32)
    
    return image

def blur(image):
    seed1=random.random()
    #[0.2,0.4,0.4]
    if seed1>0.8:
        image = ndimage.gaussian_filter(image, sigma=1)#高斯模糊#[1~1.5]#0.3s/gragh
    elif seed1>0.4:
        image = ndimage.median_filter(image,size=2)#中值滤波模糊#size=3,5s/graph;size=2,1s/gragh
    elif seed1>0.:
        image = signal.wiener(image, mysize=3)#维纳滤波模糊#size=5,2s/graph;size=3,1.6s/gragh
        image=image.astype(np.float32)
    return image



def sharpen(image):
    T,Z,H,W = image.shape
    image_ls=[]
    seed1=random.random()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_seg=Image.fromarray(np.uint8(255*image[i,j,:,:]))
            #[0.3 0.3 0.2 0.2]
            if seed1>0.7:
                img_seg=img_seg.filter(ImageFilter.EDGE_ENHANCE_MORE)
            elif seed1>0.4:
                img_seg=img_seg.filter(ImageFilter.EDGE_ENHANCE)
            elif seed1>0.2:
                img_seg=img_seg.filter(ImageFilter.DETAIL)  
            elif seed1>0.:
                img_seg=img_seg.filter(ImageFilter.SHARPEN)
            image_ls.append(np.asarray(img_seg)[None, ...])
    image_ls=np.concatenate(image_ls, axis=0)
    image=image_ls.reshape(T,Z,H,W)/255.
    image=image.astype(np.float32)
    
    return image

def emboss(image):
    T,Z,H,W = image.shape
    image_ls=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_seg=Image.fromarray(np.uint8(255*image[i,j,:,:]))
            img_seg=img_seg.filter(ImageFilter.EMBOSS)
            image_ls.append(np.asarray(img_seg)[None, ...])
    image_ls=np.concatenate(image_ls, axis=0)
    image=image_ls.reshape(T,Z,H,W)/255.
    # image = torch.Tensor(image)#https://blog.csdn.net/qq_42346574/article/details/120100424
    image=image.astype(np.float32)
    return image

#钝化掩蔽
def mask(image):
    mask1 = ndimage.gaussian_filter(image, sigma=1)#钝化
    mask3 = ndimage.gaussian_filter(image, sigma=3)#钝化
    image = mask3 +6*(mask3-mask1)#钝化掩蔽作差
    return image

def image_net_autoaugment(image):
    autoaugment_transform = tf.Compose([ImageNetPolicy(fillcolor=(0,))])
    T,Z,H,W = image.shape
    image_ls=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_seg=Image.fromarray(np.uint8(255*image[i,j,:,:]))
            img_seg = autoaugment_transform(img_seg)
            image_ls.append(np.asarray(img_seg)[None, ...])
    
    image_ls=np.concatenate(image_ls, axis=0)
    image=image_ls.reshape(T,Z,H,W)/255.
    # image = torch.Tensor(image)#https://blog.csdn.net/qq_42346574/article/details/120100424
    image=image.astype(np.float32)
    return image

def diffframe(image):
        """
        image [8, Z, H, W]
        """
        num_model, Z, H, W = image.shape
        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        
        diff_img = []
        for m in range(num_model):
            prev_f = image[m, 0, :]
            for frame in range(Z-1):
                next_f = image[m, frame+1, :]
                diff = cv2.absdiff(next_f, prev_f)
                diff = cv2.dilate(diff, es, iterations=2)
                # TODO: np.max(diff) == np.min(diff)
                if np.max(diff) == np.min(diff):
                    diff = diff
                else:
                    diff = image_normalization(diff)
                diff_img.append(diff)
                # diff_img_vis.append(diff_vis)
                prev_f = next_f
            diff_img.append(np.zeros((H, W)))
        
            # # vis
            # diff_img_2D = np.concatenate(diff_img, axis=0) *255.0
            # img_ori = image[m, :].reshape(Z*H, W) * 255.0
            # cv2.imwrite('test.png', np.concatenate([img_ori, diff_img_2D], axis=1))
        diff_img_3D = np.stack(diff_img, axis=0).reshape(num_model, Z, H, W)
        diff_img_3D = diff_img_3D.astype(np.float32)
        return diff_img_3D
if __name__ == "__main__":
    import os
    # import nibabel as nib
    # import matplotlib.pyplot as plt
    img_path = "E:/Git/gitproject/FLARE22/nnunet/DATASET/nnUNet_raw/nnUNet_raw_data/Task022_FLARE22/imagesTr/FLARE22_Tr_0001_0000.nii.gz"
    image = load_nii_file(img_path)



    image_list = cut_3Dimage(image,2)
    image_re = resize3D(image_list,(48,256,256),'trilinear')
    # image_nor = image_normalization(image_re)


    # def showNii(img,patch):
    #     for i in range(img.shape[1]):
    #         plt.imshow(img[patch,i, :, :], cmap='gray')
    #         plt.show()
    #
    #
    # showNii(image_re,2)

