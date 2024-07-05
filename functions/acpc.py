from utils import * 

class AcPcDetector(object):
    def __init__(self, args, image_size = 256):
        provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
        self.image_size = image_size
        self.detector = ort.InferenceSession(osp.join(args.model_dir, "acpc_detector.onnx"), providers=[provider])
        self.processor = get_resnet_processor(self.image_size, order=3)
        
    def __call__(self, images):
        ori_width, ori_height = images.shape[1], images.shape[2]
        pre_images = torch.stack([self.processor(gray_to_rgb(im)) for im in images])
        pred_points = self.detector.run(None, {"input": pre_images.numpy()})[0]
        points = postprocess_convert_points(pred_points, (ori_width, ori_height), (self.image_size, self.image_size))
        
        return points