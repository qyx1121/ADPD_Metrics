import os
import json
import glob
import argparse

from utils import *
from functions.preprocess import Register, HeadMove
from functions.acpc import AcPcDetector
from functions.measure import MetricsDetector


model_name = "vmf"
seg_image_size = 256
def main(args):
    dicom_paths = glob.glob(args.dicom_dir + "/*")
    print(f"Find {len(dicom_paths)} dicom files")
    if len(dicom_paths) == 0:
        print("No dicom files found in this path")
        return

    register = Register(args)
    head_move = HeadMove(args, model_name)
    acpc_detector = AcPcDetector(args)

    metrics_detector = MetricsDetector(args, model_name)
    
    for idx, dcm_p in enumerate(dicom_paths):
        res = {}
        save_dir = osp.join(args.save_dir, osp.basename(dcm_p))
        os.makedirs(save_dir, exist_ok=True)
        original_image = load_dicom(dcm_p) 
        norm_image = normalize(original_image)
        ### 保存标准化后的nii文件 ###
        sitk.WriteImage(norm_image, osp.join(save_dir, "norm_image.nii"))
        
        image = sitk.GetArrayFromImage(norm_image)
        original_image = sitk.GetArrayFromImage(original_image)

        ### Register ###
        image, pos = register(image)
        if pos == 0:
            ori_width, ori_height = original_image[0, :, :].shape
        elif pos == 1:
            ori_width, ori_height = original_image[:, 0, :].shape
        else:
            ori_width, ori_height = original_image[:, :, 0].shape

        ### Head_Move ###
        image, mid_line = head_move(image, seg_image_size)
        registered_image = sitk.GetImageFromArray(image)
        sitk.WriteImage(registered_image, osp.join(save_dir, "registered_image.nii"))
        ### Find_AcPc ###
        acpc_slices = image[mid_line-2: mid_line+2, :, :]
        points = acpc_detector(acpc_slices)
        ### 将AcPc点摆动到水平线 ###
        if points[0] > points[2]:
            image = np.flip(image, axis=2)
            width = image.shape[-1]
            points[0] = width - points[0]
            points[2] = width - points[2]
        image, new_points = adjust_acpc(points, image)
        x, y, z = image.shape
        acpc_image = rotate(image, 90, axes = (0, 1))
        acpc_image = sitk.GetImageFromArray(acpc_image)
        sitk.WriteImage(acpc_image, osp.join(save_dir, "acpc_image.nii"))
        
        ac_slice = image[:, :, new_points[0]]
        pc_slice = image[:, :, new_points[2]]
        res["ACPC"] = {
            "acpc层": mid_line, 
            "ac_points": [new_points[0], y - new_points[1] - 1], 
            "pc_points": [new_points[2], y - new_points[3] - 1]
            }

        res["BVR"] = {
            "BVR层（冠状位）": new_points[0]
        }
        res["zEvans"] = {
            "zEvans层（冠状位）": new_points[0]
        }
        res["CA"] = {
            "CA层（冠状位）": new_points[2]
        }
        acpc_slices = np.stack([ac_slice, pc_slice])

        acpc_slices = rotate(acpc_slices, angle = -90, axes=(1, 2))

        ### 检测BVR和zEvans指标 ###
        bvr_result, bvr_zei_image = metrics_detector.det_bvr_zei(acpc_slices[0], mid_line, image_size = seg_image_size)
        res['BVR']['测量值'] = round(bvr_result['BVR']['data'].item(), 3)
        bvr_line1 = bvr_result['BVR']['line_1']
        bvr_line2 = bvr_result['BVR']['line_2']
        x, y = bvr_zei_image.shape
        res['BVR']['侧脑室正高度'] = {
            "point_1": [bvr_line1[0][0], x - bvr_line1[0][1] - 1], 
            "point_2": [bvr_line1[1][0], x - bvr_line1[1][1] - 1],
            "长度": f"{abs(bvr_line1[0][1] - bvr_line1[1][1])}mm"
            }
        res['BVR']['侧脑室正上方颅内高度'] = {
            "point_1": [bvr_line2[0][0], x - bvr_line2[0][1] - 1], 
            "point_2": [bvr_line2[1][0], x - bvr_line2[1][1] - 1],
            "长度": f"{abs(bvr_line2[0][1] - bvr_line2[1][1])}mm"
            }
        
        res['zEvans']['测量值'] = round(bvr_result['zEI']['data'].item(), 3)
        zei_line1 = bvr_result['zEI']['line_1']
        zei_line2 = bvr_result['zEI']['line_2']
        res['zEvans']['侧脑室高度'] = {
            "point_1": [zei_line1[0][0], x - zei_line1[0][1] - 1],
            "point_2": [zei_line1[1][0], x - zei_line1[1][1] - 1],
            "长度": f"{abs(zei_line1[0][1] - zei_line1[1][1])}mm"
        }
        res['zEvans']['颅内最大高度'] = {
            "point_1": [zei_line2[0][0], x - zei_line2[0][1] - 1],
            "point_2": [zei_line2[1][0], x - zei_line2[1][1] - 1],
            "长度": f"{abs(zei_line2[0][1] - zei_line2[1][1])}mm"
        }

        ### 检测CA指数 ###
        ca_image = acpc_slices[1]
        ca_result, ca_image = metrics_detector.det_ca(ca_image, image_size = seg_image_size)
        res['CA']['测量值'] = round(ca_result['data'].item(), 3)
        ca_result_points = [[i[0], x - i[1] - 1] for i in ca_result['points']]
        res['CA']['point_left'], res['CA']['point_middle'], res['CA']['point_right'] = ca_result_points

        ### 检测Evans指数 ###
        x, y, z = image.shape
        ei_result, ei_image, ei_layer_id = metrics_detector.det_evans(image, image_size = seg_image_size)
        res['Evans'] = {}
        res['Evans']['Evans层（横断位）'] = y - ei_layer_id - 1
        res['Evans']['测量值'] = round(ei_result['data'].item())
        res['Evans']['侧脑室前角最大间距'] = {
            "point_1": [ei_result['line_2'][0][0], ei_result['line_2'][0][1]],
            "point_2": [ei_result['line_2'][1][0], ei_result['line_2'][1][1]],
            "长度": f"{abs(ei_result['line_2'][0][0] - ei_result['line_2'][1][0])}mm"
        }
        res['Evans']['颅内最大间距'] = {
            "point_1": [ei_result['line_1'][0][0], ei_result['line_1'][0][1]],
            "point_2": [ei_result['line_1'][1][0], ei_result['line_1'][1][1]],
            "长度": f"{abs(ei_result['line_1'][0][0] - ei_result['line_1'][1][0])}mm"
        }
        json.dump(res, open(osp.join(save_dir, "results.json"), "w"), indent = 2, ensure_ascii=False)
        #plt.imsave(osp.join(save_dir, "bvr_zei_image.png"), bvr_zei_image, cmap="gray")
        #plt.imsave(osp.join(save_dir, "ca_image.png"), ca_image, cmap="gray")
        #plt.imsave(osp.join(save_dir, "ei_image.png"), ei_image, cmap="gray")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, default="./data",
                        help="the input directory of dicom files")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="directory where to save the measurement results")
    parser.add_argument("--gpu", action="store_true", help="enforce running with CPU rather than GPU.")
    parser.add_argument("--model_dir", type=str, default="./models", help="the directory where the models are stored")

    args = parser.parse_args()
    main(args)