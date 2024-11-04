import os
import sys
import copy
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
        res = {
            'Evans': {}, 'ACPC': {}, 'BVR': {},
            'zEvans':{}, 'CA': {}
               }
        save_dir = osp.join(args.save_dir, osp.basename(dcm_p))
        os.makedirs(save_dir, exist_ok=True)
        original_image = load_dicom(dcm_p) 
        norm_image = normalize(original_image)
        ### 保存标准化后的nii文件 ###
        # sitk.WriteImage(norm_image, osp.join(save_dir, "norm_image.nii"))

        image = sitk.GetArrayFromImage(norm_image)
        original_image = sitk.GetArrayFromImage(original_image)

        ### Register ###
        image, pos = register(image)
        ### Head_Move ###
        registered_image, mid_line = head_move(image, seg_image_size)
        ### Find_AcPc ###
        acpc_image, points, reverse = acpc_detector(registered_image, mid_line, res)
        ### 检测Evans指数 ###
        if reverse:
            registered_image = np.flip(registered_image, axis = 2)
        view_image = rotate(registered_image, 90, (0, 1))
        sitk.WriteImage(sitk.GetImageFromArray(view_image), osp.join(save_dir, "registered_image.nii"))

        ### for test ###
        # sitk.WriteImage(sitk.GetImageFromArray(rotate(acpc_image, 90, (0, 1))), osp.join(save_dir, "acpc_image.nii"))
        metrics_detector.det_evans(registered_image, seg_image_size, res)
        ### 检测BVR和zEvans指标 ###
        metrics_detector.det_bvr_zei(acpc_image, points[0], mid_line, seg_image_size, res)

        ### 检测CA指数 ###
        metrics_detector.det_ca(acpc_image, points[2], res, seg_image_size)
        
        save_path = osp.join(save_dir, "results.json")
        json.dump(res, open(save_path, "w"), indent = 2, ensure_ascii=False)

        print("AI automatic recognition is complete!")
        ### 手动调整 ###
        while True:
            '''
            如果是修改Evans层，比如最大层86，则输入：evans 86；
            如果是修改acpc点的坐标，则依次输入ac、pc的坐标：acpc 109 70 135 77
            如果是输入其他内容，则表示当前样本处理完成
            '''
            input_content = sys.stdin.readline().split(" ")
            adjust = input_content[0]
            # adjust = "acpc"
            if adjust == "evans":
                layer_id = int(input_content[1]) # 86 
                metrics_detector.det_evans(registered_image, seg_image_size, res, layer_id)
                json.dump(res, open(save_path, "w"), indent = 2, ensure_ascii=False)
                print("AI automatic recognition is complete!")
            elif adjust == "acpc":
                ac_point = [int(input_content[1]), int(input_content[2])] # [109, 70]
                pc_point = [int(input_content[3]), int(input_content[4])] # [135, 77]
                x, y, z = registered_image.shape
                ac_point[1] = y - ac_point[1] - 1
                pc_point[1] = y - pc_point[1] - 1
                acpc_image, points = adjust_acpc(ac_point + pc_point, registered_image)
                metrics_detector.det_bvr_zei(acpc_image, points[0], mid_line, seg_image_size, res)
                metrics_detector.det_ca(acpc_image, points[2], res, seg_image_size)
                json.dump(res, open(save_path, "w"), indent = 2, ensure_ascii=False)
                print("AI automatic recognition is complete!")
            else:
                print(f"The recoginition of this dicom file {dcm_p} is complete!\nThe result is saved in {save_path}")
                break
            

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