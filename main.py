import glob
import argparse

from utils import *
from functions.preprocess import Register, HeadMove
from functions.acpc import AcPcDetector
from functions.measure import MetricsDetector



def main(args):
    dicom_paths = glob.glob(args.dicom_dir + "/*")
    print(f"Find {len(dicom_paths)} dicom files")
    if len(dicom_paths) == 0:
        print("No dicom files found in this path")
        return

    register = Register(args)
    head_move = HeadMove(args)
    acpc_detector = AcPcDetector(args)

    metrics_detector = MetricsDetector(args)

    for dcm_p in dicom_paths:
        original_image = load_dicom(dcm_p) 
        norm_image = normalize(original_image)
        image = sitk.GetArrayFromImage(norm_image)

        ### Register ###
        image = register(image)

        ### Head_Move ###
        image, mid_line = head_move(image)
        ### Find_AcPc ###
        acpc_slices = image[mid_line-2: mid_line+2, :, :]
        points = acpc_detector(acpc_slices)

        if points[0] > points[2]:
            image = np.flip(image, axis=2)
            width = image.shape[-1]
            points[0] = width - points[0]
            points[2] = width - points[2]
        image, new_points = adjust_acpc(points, image)

        ac_slice = image[:, :, new_points[0]]
        pc_slice = image[:, :, new_points[2]]

        acpc_slices = np.stack([ac_slice, pc_slice])

        acpc_slices = rotate(acpc_slices, angle = -90, axes=(1, 2))

        metrics_detector.det_bvr_zei(acpc_slices[0], mid_line)
        metrics_detector.det_ca(acpc_slices[1])
        metrics_detector.det_evans(image)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, default="dicoms",
                        help="the input directory of dicom files")
    parser.add_argument("--path_predictions", type=str, default="results",
                        help="path where to save the measurement results")
    parser.add_argument("--gpu", action="store_true", help="enforce running with CPU rather than GPU.")
    parser.add_argument("--model_dir", type=str, default="models", help="the directory where the models are stored")
   
    args = parser.parse_args()
    main(args)