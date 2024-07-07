
import cv2
import os
import os.path as osp
import numpy as np
import torch
from torchvision import transforms
import onnxruntime as ort

import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy.ndimage import rotate, zoom
from tqdm import tqdm

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import math
import copy
from copy import deepcopy
import sys

def visualize_masks(image, mask):

    ### RGB image ###
    image = gray_to_rgb(image)
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    if np.any(mask == 1):
        one_mask = deepcopy(mask)
        one_mask[one_mask != 1] = 0
        mask_colored[:, :, 0] = one_mask * 100  # Red channel
    
    if np.any(mask == 2):
        two_mask = deepcopy(mask)
        two_mask[two_mask != 2] = 0
        two_mask[two_mask == 2] = 1
        mask_colored[:, :, 2] = two_mask * 100
    # mask_colored[:, :, 1] = 0           # Green channel
    # mask_colored[:, :, 2] = 0           # Blue channel

    alpha = 0.5 
    overlay_image = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

    return overlay_image

def visualize_multiple_images(images, points = None):
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(10, 5))
    for i in range(len(images)):
        if len(images[i].shape) == 2:
            axs[i].imshow(images[i], cmap="gray")
        else:
            axs[i].imshow(images[i])
        # axs[i].set_xticks([])
        # axs[i].set_yticks([])
        if points is not None:
            axs[i].scatter(points[0], points[1], c='red', marker='o', s = 2)
            axs[i].scatter(points[2], points[3], c='red', marker='o', s = 2)

    plt.tight_layout()
    plt.show()

def visualize(image):
    fig, axs = plt.subplots(1, 3, figsize=(9, 5))
    x, y, z = image.shape
    image_1, image_2, image_3 = image[x//2, :, :], image[:, y//2, :], image[:, :, z//2]
    axs[0].imshow(image_1, cmap='gray')
    axs[1].imshow(image_2, cmap='gray')
    axs[2].imshow(image_3, cmap='gray')
    axs[0].set_xticks([]) 
    axs[0].set_yticks([])
    axs[1].set_xticks([]) 
    axs[1].set_yticks([])
    axs[2].set_xticks([]) 
    axs[2].set_yticks([])
    
    plt.tight_layout()
    plt.show()




def get_unet_processor(image_size = 224):
    def preprocess(image):
        x, y = image.shape
        if x != image_size or y != image_size:
            image = zoom(image, (image_size / x, image_size / y), order=3)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        return image
    return preprocess

def get_resnet_processor(image_size = 224, order = 1):
    
    interpolation = transforms.InterpolationMode.BICUBIC if order == 3 else transforms.InterpolationMode.BILINEAR

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), interpolation),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_dicom(dcm_dir):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return image

def normalize(original_image):
    new_spacing = [1.0, 1.0, 1.0] 
    original_size = original_image.GetSize()
    original_spacing = original_image.GetSpacing()
    
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(original_image.GetDirection())
    resampler.SetOutputOrigin(original_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(original_image.GetPixelIDValue())

  
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled_image = resampler.Execute(original_image)

    return resampled_image

def gray_to_rgb(gray_image, cmap='gray'):
    colormap = plt.get_cmap(cmap)
    normalized_gray_image = gray_image / np.max(gray_image)
    rgb_image = colormap(normalized_gray_image)
    rgb_image = rgb_image[:, :, :3]

    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

def register(args, image):

    x, y, z = image.shape
    x_slice, y_slice, z_slice = gray_to_rgb(image[x//2, :, :]), gray_to_rgb(image[:, y//2, :]), gray_to_rgb(image[:, :, z//2])
    images = [x_slice, y_slice, z_slice]

    provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
    model = ort.InferenceSession(osp.join(args.model_dir, "register.onnx"), providers=[provider])
    processor = get_resnet_processor()

    for idx, im in enumerate(images):
        images[idx] = processor(im)
    
    images = torch.stack(images)
    pos_logits, op_logits = model.run(None, {'input':images.numpy()})
    
    pos_idx = np.argmax(pos_logits, axis = -1)
    op_idx = np.argmax(op_logits, axis = -1)

    ### adjust dimension ###
    zero_pos = np.where(pos_idx ==0)[0].item()
    one_pos = np.where(pos_idx ==1)[0].item()
    two_pos = np.where(pos_idx ==2)[0].item()

    #image = image.transpose(zero_pos, one_pos, two_pos)

    ### adjust rotation ###
    op = op_idx[zero_pos].item()

    image = rotate(image, op * 90, axes = (one_pos, two_pos), reshape = False)
    if zero_pos == 1:
        image = image.transpose(1, 2, 0)
    elif zero_pos == 2:
        image = image.transpose(2, 0, 1)

    #image = np.rot90(image, op, axes=(one_pos,two_pos))


    return image


### adjust head ###
def calculate_angle(image, max_iter = 100000):
    X = []  
    y = []  
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 0:  
                X.append([j, i])
                y.append(image[i, j])
    X = np.array(X)
    y = np.array(y)

    # scaler = MinMaxScaler()
    # model = make_pipeline(scaler, svm.SVC(kernel='linear', max_iter=max_iter))
    model = svm.SVC(kernel='linear', max_iter=max_iter)
    model.fit(X, y)
    #model = model.named_steps['svc']
    # scaler.fit_transform()
    w = model.coef_[0]
    b = model.intercept_[0]
    slope = -w[0] / w[1]
    intercept = -b / w[1]

    return slope, intercept

def calculate_midline(masks, threshold = None):
    slopes = []
    intercepts = []
    for i in tqdm(range(len(masks))):
        mask = masks[i]
        if len(np.unique(mask)) != 3:  ## 未能正确分割出三种
            continue
        slope, intercept = calculate_angle(mask)

        if threshold is not None:
            if abs(slope) > threshold:
                continue
        slopes.append(slope)
        intercepts.append(intercept)
    slopes = np.array(slopes)
    Q1 = np.percentile(slopes, 25)
    Q3 = np.percentile(slopes, 75)
    IQR = Q3 - Q1
    normal_points = slopes[((slopes > (Q1 - 1.5 * IQR)) & (slopes < (Q3 + 1.5 * IQR)))]

    ## 将剩余的正常值取平均，计算偏移角
    slope = np.mean(normal_points)
    angle = math.atan(slope) * 180 / math.pi
    print("The offset angle is {:.2f}°".format(angle))

    intercept = intercepts[np.argmin(np.abs(slopes - slope))]

    return slope, intercept, angle


def segmentation(slices, ori_images, model):
    restored_mask = []
    for i in range(len(slices)):
        sli = slices[i].unsqueeze(0).numpy()
        outputs = model.run(None, {"input":sli})[0]
        outputs = torch.tensor(outputs)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        outputs = outputs.data.cpu().numpy().squeeze()

        x, y = ori_images[0].shape
        ### 将得到的mask恢复到原始尺度
        mask = zoom(outputs, (x / 224, y / 224), order=1)
        mask = np.clip(mask, 0, 2)
        restored_mask.append(mask)
    restored_mask = np.stack(restored_mask)

    ### mask后处理，消除离群点
    kernel = np.ones((3,3), np.uint8)
    for i in range(len(restored_mask)):
        r_mask = restored_mask[i, :, :]
        binary_image_1 = np.where(r_mask == 1, 1, 0).astype(np.uint8)
        binary_image_2 = np.where(r_mask == 2, 1, 0).astype(np.uint8)

        cleaned_image_1 = cv2.morphologyEx(binary_image_1, cv2.MORPH_OPEN, kernel)
        cleaned_image_2 = cv2.morphologyEx(binary_image_2, cv2.MORPH_OPEN, kernel)        
        r_mask = cleaned_image_1 * 1 + cleaned_image_2 * 2
        restored_mask[i, :, :] = r_mask
    
    return restored_mask


def adjust_z(args, image):

    provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
    model = ort.InferenceSession(osp.join(args.model_dir, "brain_seg_z.onnx"), providers=[provider])
    preprocess = get_unet_processor()
    bound = image.shape[-1]
    low_bound = int(bound * 0.2)
    high_bound = int(bound * 0.8)

    miss_low_bound, miss_high_bound = int(bound * 0.4), int(bound * 0.6)

    slices = torch.stack([preprocess(image[:, :, i]) for i in range(low_bound, high_bound, 3) if i <= miss_low_bound or i >= miss_high_bound])
    ori_images = [image[:, :, i] for i in range(low_bound, high_bound, 3) if i <= miss_low_bound or i >= miss_high_bound]
    masks = segmentation(slices, ori_images, model)
    # interval = len(masks) // 5
    # vis_images = []
    # i = 0
    # while i < len(masks):
    #     vis_images.append(visualize_masks(ori_images[i], masks[i]))
    #     i+= interval

    # visualize_multiple_images(vis_images)

    slope, intercept, angle = calculate_midline(masks, 0.5)
    adjusted_image = rotate(image, angle=angle, axes=(0,1), reshape=False, mode="constant", cval=0.0)
    return adjusted_image

def adjust_y(args, image):

    provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
    EvansModel = ort.InferenceSession("/home/qinyixin/workspace/Swin-Unet/judge_segy.onnx", providers=[provider])
    x, y, z = image.shape
    low_bound, high_bound = int(0.3 * y), int(0.7 * y)
    ori_y_images = image[:, low_bound:high_bound, :]
    resnet_processor = get_resnet_processor(224, 3)

    y_images = [resnet_processor(gray_to_rgb(ori_y_images[:, i, :])) for i in range(ori_y_images.shape[1])]
    ori_y_images = np.stack([ori_y_images[:, i, :] for i in range(ori_y_images.shape[1])])


    y_images = torch.stack(y_images)
    logits = EvansModel.run(None, {'input':y_images.numpy()})[0]
    positive_indexes = np.argmax(logits, axis = 1)

    candidates = ori_y_images[positive_indexes == 1]
    ori_images = candidates[::2]

    preprocess = get_unet_processor()
    slices = torch.stack([preprocess(ori_images[i]) for i in len(ori_images)]) 


    provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
    model = ort.InferenceSession(osp.join(args.model_dir, "brain_seg_y.onnx"), providers=[provider])

    # bound = image.shape[1]
    # low_bound = int(bound * 0.2)
    # high_bound = int(bound * 0.4)
    # slices = torch.stack([preprocess(image[:, i, :]) for i in range(low_bound, high_bound, 3)])
    # ori_images = [image[:, i, :] for i in range(low_bound, high_bound, 3)]
    masks = segmentation(slices, ori_images, model)
    slope, intercept, angle = calculate_midline(masks, 0.5)
    
    x_plot = np.arange(0, image.shape[-1])
    y_plot = slope * x_plot + intercept
    y_plot = np.round(y_plot).astype(np.int16)
    label = np.zeros((image.shape[0], image.shape[-1]))
    label[y_plot, x_plot] = 1
    adjusted_image = rotate(image, angle=angle, axes=(0,2), reshape=False, mode="constant", cval=0.0)
    adjusted_label = rotate(label, angle=angle, axes=(0,1), reshape=False, mode="constant", cval=0.0)
    
    y, x = np.where(adjusted_label > 0.1)
    y = round(np.mean(y))

    return adjusted_image, y

def head_move(args, image):
    image_adjust_z = adjust_z(args, image)
    image_adjust_y, mid_line = adjust_y(args, image_adjust_z)
    return image_adjust_y, mid_line


def postprocess_convert_points(points, new_size, ori_size):

    Q1 = np.percentile(points, 25, axis=0)
    Q3 = np.percentile(points, 75, axis=0)
    IQR = Q3 - Q1

    low_bound = Q1 - 1.5 * IQR
    high_bound = Q3 + 1.5 * IQR

    normal_points = (points > low_bound) & (points < high_bound)
    points = np.average(points, weights=normal_points, axis=0)
 
    ori_width, ori_height = ori_size
    new_width, new_height = new_size

    points = points.tolist()

    points[0] = points[0] / ori_height * new_height
    points[1] = points[1] / ori_width * new_width
    points[2] = points[2] / ori_height * new_height
    points[3] = points[3] / ori_width * new_width

    points = torch.round(torch.tensor(points)).to(torch.int16).tolist()
    return points


def find_acpc_line(args, images):
    provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
    model = ort.InferenceSession(osp.join(args.model_dir, "acpc_detector.onnx"), providers=[provider])
    # RESNET_WEIGHTS = "/mnt/hdd1/qinyixin/huaxiproj/AC-PC/models/resnet-152"
    # model = AcPc_FPN_Detector(RESNET_WEIGHTS)
    # ckpt = torch.load("/home/qinyixin/workspace/Swin-Unet/AC_PC/AC_PC_ckpt/resnet-152_fpn_256_999.pth", map_location="cpu")
    # new_ckpt = OrderedDict()
    # for k, v in ckpt.items():
    #     k = k.replace("module.", "")
    #     new_ckpt[k] = v
    # model.load_state_dict(new_ckpt)
    # model = model.cuda()

    processor = get_resnet_processor(args.acpc_image_size, 3)

    ori_width, ori_height = images.shape[1], images.shape[2]
    pre_images = torch.stack([processor(gray_to_rgb(im)) for im in images])
    pred_points = model.run(None, {"input": pre_images.numpy()})[0]
    #pred_points = model(pre_images.cuda()).cpu().detach().numpy()
    points = postprocess_convert_points(pred_points, (ori_width, ori_height), (args.acpc_image_size, args.acpc_image_size))
    
    ac_pt, pc_pt = points[:2], points[2:]

    return points

def rotate_coordinate(x, y, angle, image_shape):
    theta_rad = np.deg2rad(angle)
    
    center_x = image_shape[2] / 2
    center_y = image_shape[1] / 2

    x_shifted = x - center_x
    y_shifted = y - center_y
    
    x_rotated = x_shifted * np.cos(theta_rad) - y_shifted * np.sin(theta_rad)
    y_rotated = x_shifted * np.sin(theta_rad) + y_shifted * np.cos(theta_rad)
    
    x_new = x_rotated + center_x
    y_new = y_rotated + center_y
    
    return x_new, y_new

def adjust_acpc(points, image):
    delta_x = points[0] - points[2]
    delta_y = points[1] - points[3]

    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)
    rot_deg = 180 + angle_deg

    image = rotate(image, rot_deg, axes = (1, 2), reshape=False)

    new_points = [0, 0, 0, 0]

    new_points[0], new_points[1] = rotate_coordinate(points[0], points[1], -rot_deg, image.shape)
    new_points[2], new_points[3] = rotate_coordinate(points[2], points[3], -rot_deg, image.shape)

    new_points = np.round(new_points).astype(np.int16).tolist()
    
    visual_image = copy.deepcopy(image[image.shape[0] // 2, :, :])
    cv2.circle(visual_image, (new_points[0], new_points[1]), 2, (255, 255, 255), -1)
    cv2.circle(visual_image, (new_points[2], new_points[3]), 2, (255, 255, 255), -1)
    plt.imsave(f"tmp.png", visual_image, cmap="gray")

    return image, new_points


def infer_zEI_BVR(seg_image, mid_line):
    head = np.argwhere(seg_image == 2)
    centroid = np.argwhere(seg_image == 1)

    ### for Highest line ###
    head_height_indexes = np.argwhere(seg_image[:, mid_line] == 2)
    head_height = head_height_indexes.max() - head_height_indexes.min()

    ### for centroid height ###
    w_axis = list(centroid[:, 0])
    h_axis = centroid[:, 1]
    l = round(1/2 * len(list(set(w_axis))))

    left_centroid = centroid[:, :l]
    right_centroid = centroid[:, l:]

    left_max_height = np.argmax(left_centroid == 1, axis=0)
    left_min_height = np.argmin(left_centroid == 1, axis=0)

    left_pos_x = left_min_height
    left_pos_y = np.where(left_centroid[left_pos_x, :] == 1)
    left_pos_y = np.mean(left_pos_y)

    left_height = left_max_height - left_min_height
    
    right_max_height = np.argmax(right_centroid == 1, axis = 0)
    right_min_height = np.argmin(right_centroid == 1, axis= 0)
    right_height = right_max_height - right_min_height
    
    right_pos_x = right_min_height
    right_pos_y = np.where(right_centroid[right_pos_x, :] == 1)
    right_pos_y = np.mean(right_pos_y)
    
    ### calculate zEI ###
    zEI = max(left_max_height, right_max_height) / head_height

    ### calculate BVR ###
    if left_max_height > right_max_height:
        pos_x, pos_y = left_pos_x, left_pos_y
    else:
        pos_x, pos_y = right_pos_x, right_pos_y
    
    
def post_process_seg(seg_mask, area_threshold = 10):
    
    for i in [1, 2]:
        tmp = copy.deepcopy(seg_mask)
        tmp[tmp != i] = 0
        tmp[tmp == i] = 1

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area <= area_threshold:
                seg_mask[labels == j] = 0

    return seg_mask




    

    


