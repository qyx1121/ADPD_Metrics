from utils import * 

class AcPcDetector(object):
    def __init__(self, args, image_size = 256):
        provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
        self.image_size = image_size
        self.detector = ort.InferenceSession(osp.join(args.model_dir, "acpc_detector.onnx"), providers=[provider])
        self.processor = get_resnet_processor(self.image_size, order=3)
        
    def __call__(self, acpc_image, mid_line, res):
        reverse = False
        images = acpc_image[mid_line - 2: mid_line + 2, :, :]
        ori_width, ori_height = images.shape[1], images.shape[2]
        pre_images = torch.stack([self.processor(gray_to_rgb(im)) for im in images])
        pred_points = self.detector.run(None, {"input": pre_images.numpy()})[0]
        points = postprocess_convert_points(pred_points, (ori_width, ori_height), (self.image_size, self.image_size))
        
        if points[0] > points[2]:
            acpc_image = np.flip(acpc_image, axis=2)
            width = acpc_image.shape[-1]
            points[0] = width - points[0]
            points[2] = width - points[2]
            reverse = True

        x, y, z = acpc_image.shape 
        res["ACPC"] = {
            "acpc层": mid_line,
            "ac_points": [points[0], y - points[1] - 1],
            "pc_points": [points[2], y - points[3] - 1]
        }
        
        acpc_image, new_points = adjust_acpc(points, acpc_image)
        
        return acpc_image, new_points, reverse