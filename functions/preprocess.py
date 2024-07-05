from utils import *


class Register(object):
    def __init__(self, args):
        provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
        self.register = ort.InferenceSession(osp.join(args.model_dir, "register.onnx"), providers=[provider])
        self.processor = get_resnet_processor()

    def __call__(self, image):
        x, y, z = image.shape
        x_slice, y_slice, z_slice = gray_to_rgb(image[x//2, :, :]), gray_to_rgb(image[:, y//2, :]), gray_to_rgb(image[:, :, z//2])
        images = [x_slice, y_slice, z_slice]

        for idx, im in enumerate(images):
            images[idx] = self.processor(im)
        images = torch.stack(images)
        pos_logits, op_logits = self.register.run(None, {"input": images.numpy()})

        pos_idx = np.argmax(pos_logits, axis = -1)
        op_idx = np.argmax(op_logits, axis = -1)

        ### adjust dimension ###
        zero_pos = np.where(pos_idx ==0)[0].item()
        one_pos = np.where(pos_idx ==1)[0].item()
        two_pos = np.where(pos_idx ==2)[0].item()

        ### adjust rotation ###
        op = op_idx[zero_pos].item()

        image = rotate(image, op * 90, axes = (one_pos, two_pos), reshape = False)
        if zero_pos == 1:
            image = image.transpose(1, 2, 0)
        elif zero_pos == 2:
            image = image.transpose(2, 0, 1)

        return image

        
class HeadMove(object):
    def __init__(self, args):
        provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
        self.seg_z_model = ort.InferenceSession(osp.join(args.model_dir, "brain_seg_z.onnx"), providers=[provider])
        self.seg_y_model = ort.InferenceSession(osp.join(args.model_dir, "brain_seg_y.onnx"), providers=[provider])
        self.unet_processor = get_unet_processor()
        self.res_processor = get_resnet_processor(order = 3)

        self.judge_segy = ort.InferenceSession(osp.join(args.model_dir, "judge_segy.onnx"), providers=[provider])


    def __call__(self, image):
        image_z = self.adjust_z(image)
        image_y, mid_line = self.adjust_y(image_z)

        return image_y, mid_line
    
    def adjust_y(self, image):
        x, y, z = image.shape
        y_images = [self.res_processor(gray_to_rgb(image[:, i, :])) for i in range(0, y, 3)]
        ori_y_images = np.stack([image[:, i, :] for i in range(0, y, 3)])

        y_images = torch.stack(y_images)
        logits = self.judge_segy.run(None, {'input':y_images.numpy()})[0]
        positive_indexes = np.argmax(logits, axis = 1)
        ori_images = ori_y_images[positive_indexes == 1][::2]
        slices = torch.stack([self.unet_processor(ori_images[i]) for i in range(len(ori_images))])
        masks = segmentation(slices, ori_images, self.seg_y_model)
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

    def adjust_z(self, image):
        bound = image.shape[-1]
        low_bound, high_bound = int(bound * 0.4), int(bound * 0.6)
        
        ori_images = [image[:, :, i] for i in range(int(bound * 0.2), low_bound, 3)]
        ori_images += [image[:, :, i] for i in range(high_bound, int(bound * 0.8), 3)]
        slices = torch.stack([self.unet_processor(ori_images[i]) for i in range(len(ori_images))])
        masks = segmentation(slices, ori_images, self.seg_z_model)
        slope, intercept, angle = calculate_midline(masks, 0.5)
        adjusted_image = rotate(image, angle=angle, axes=(0,1), reshape=False, mode="constant", cval=0.0)
        return adjusted_image
