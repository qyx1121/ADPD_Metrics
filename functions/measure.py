from typing import Any
from utils import *

from sklearn.linear_model import LinearRegression


class MetricsDetector(object):
    def __init__(self, args, model_name):
        self.res_processor = get_resnet_processor(order=3)
        self.unet_processor = get_unet_processor()
        provider = 'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'

        self.bvr_seg = ort.InferenceSession(osp.join(args.model_dir, f"bvr_256_{model_name}.onnx"), providers=[provider])
        self.ca_seg = ort.InferenceSession(osp.join(args.model_dir, f"ca_{model_name}.onnx"), providers=[provider])

        self.judge_evans = ort.InferenceSession(osp.join(args.model_dir, "judge_evans.onnx"), providers=[provider])
        self.evans_seg = ort.InferenceSession(osp.join(args.model_dir, f"evans_{model_name}.onnx"), providers=[provider])
 
    def det_evans(self, ei_image, image_size):
        
        result = {}
        ### detect proper evans slices ###
        x, y, z = ei_image.shape
        low_bound, high_bound = int(0.3 * y), int(0.7 * y)
        ori_y_images = ei_image[:, low_bound:high_bound, :]

        y_images = [self.res_processor(gray_to_rgb(ori_y_images[:, i, :])) for i in range(ori_y_images.shape[1])]
        ori_y_images = np.stack([ori_y_images[:, i, :] for i in range(ori_y_images.shape[1])])

        y_images = torch.stack(y_images)
        logits = self.judge_evans.run(None, {'input':y_images.numpy()})[0]
        positive_indexes = np.argmax(logits, axis = 1)
        candidates = ori_y_images[positive_indexes == 1]
        zEvans_image = rotate(candidates, -90, axes = (1, 2))

        ### detect evans ###
        _, ori_height, ori_width = zEvans_image.shape
        for i in range(_):
            zEvans_image[i] = cv2.cvtColor(gray_to_rgb(zEvans_image[i]), cv2.COLOR_RGB2GRAY)
    
        process_zEvans_image = torch.stack([get_unet_processor(image_size)(zEvans_image[i]) for i in range(len(zEvans_image))])

        outputs = np.array([self.evans_seg.run(None, {"input": process_zEvans_image[i:i+1].numpy()})[0] for i in range(len(process_zEvans_image))])
        outputs = torch.tensor(outputs).squeeze()
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        outputs = outputs.data.cpu().numpy()

        restored_mask = zoom(outputs, (1, ori_height / image_size, ori_width / image_size), order=0)
        restored_mask = np.clip(restored_mask, 0, 2)
        restored_mask = restored_mask.astype(np.uint8)

        ### post process ###

        for i in range(len(restored_mask)):
            restored_mask[i] = post_process_seg(restored_mask[i])

        center_mask = deepcopy(restored_mask) 
        center_mask[center_mask == 1] = 0

        max_center_id = -1
        max_center_length = 0
        c_maxh, c_minh, c_maxw, c_minw = 0, 0, 0, 0
        for i in range(len(center_mask)):
            c_mask = center_mask[i]
            c_mask = cv2.medianBlur(c_mask, 3)

            c_tmp = np.nonzero(c_mask)
            x_axis = list(set(c_tmp[0]))
            crop_x = (max(x_axis) - min(x_axis))//3 + min(x_axis)
            
            c_mask = c_mask[:crop_x, :]
            axis_x = set(np.nonzero(c_mask)[0].tolist())

            for x in axis_x:
                row = c_mask[x]
                tmp_row = np.nonzero(row)[0]
                interval = tmp_row.max() - tmp_row.min()
                if interval > max_center_length:
                    max_center_length = interval
                    c_minw, c_maxw = tmp_row.min(), tmp_row.max()
                    c_minh = c_maxh = x
                    max_center_id = i
        ei_image = np.rot90(candidates[max_center_id], -1)
        ei_mask = restored_mask[max_center_id]  
        
        boundary_mask = deepcopy(restored_mask)   
        boundary_mask = boundary_mask[max_center_id]
        boundary_mask[boundary_mask == 2] = 0

        b_mask = post_process_seg(boundary_mask, 50)
        b_mask = cv2.blur(b_mask, (5, 5))

        b_tmp = np.nonzero(b_mask)
        x_axis = list(set(b_tmp[0]))
        max_boundary_length = 0
        b_maxh, b_minh, b_maxw, b_minw = 0, 0, 0, 0
        for x in x_axis:
            row = b_mask[x]
            tmp_row = np.nonzero(row)[0]
            center_line = (tmp_row.min() + tmp_row.max()) // 2
            
            left_nonzero, right_nonzero = np.nonzero(row[:center_line])[0], np.nonzero(row[center_line:])[0]
            if len(left_nonzero) == 0 or len(right_nonzero) == 0:
                continue

            left_max = left_nonzero.max()
            right_min = right_nonzero.min()
            
            interval = right_min - left_max + center_line
            if interval > max_boundary_length:
                max_boundary_length = interval
                b_minw, b_maxw = left_max, right_min + center_line
                b_minh = b_maxh = x
        
        #plt.plot([c_maxw, c_minw], [c_maxh, c_minh],  marker = 'o', color = 'r', markersize = 1)
        #plt.plot([b_maxw, b_minw], [b_maxh, b_minh],  marker = 'o', color = 'r', markersize = 1)
        
        result["data"] = max_center_length / max_boundary_length
        result["line_1"] = np.round([[b_maxw, b_maxh], [b_minw, b_minh]]).astype(np.int16).tolist()
        result["line_2"] = np.round([[c_maxw, c_maxh], [c_minw, c_minh]]).astype(np.int16).tolist()
        return result, np.rot90(candidates[max_center_id], -1)

    def tange_CA(self, px, py, ca_image):

        # coefficients = np.polyfit(np.array(py), np.array(px), 5)
        # func = np.poly1d(coefficients)

        lb, hb = int(0.2 * len(px)), int(0.8 * len(px))
        px, py = px[lb:hb], py[lb:hb]
        
        max_index = int(np.mean(np.where(px == np.max(px))[0]))
        
        left_xs, right_xs = px[:max_index+1], px[max_index:]
        left_ys, right_ys = np.array(py[:max_index+1]), np.array(py[max_index:])

        x_max, y_max = px[max_index], py[max_index]

        # find left max tan 
        max_degree = 0
        left_ymin, left_xmin = 0, 0
        for i in range(len(left_ys)):
            if y_max == left_ys[i]:
                continue
            tan = (x_max - left_xs[i]) / (y_max - left_ys[i])
            if tan > max_degree:
                max_degree = tan
                left_ymin, left_xmin = left_ys[i], left_xs[i]
        
        max_degree = 0
        right_ymin, right_xmin = 0, 0
        for i in range(len(right_ys)):
            if y_max == right_ys[i]:
                continue
            tan = (x_max - right_xs[i]) / (right_ys[i] - y_max)
            if tan > max_degree:
                max_degree = tan
                right_ymin, right_xmin = right_ys[i], right_xs[i]
        
        return left_ymin, left_xmin, y_max, x_max, right_ymin, right_xmin
    
    def fit_CA(self, px, py):
        lb, hb = int(len(px) * 0.2), int(len(px) * 0.8)
        px,py = px[lb:hb], py[lb:hb]
        #max_index = int(np.mean(np.where(px == np.max(px))[0]))
        #left_xs, right_xs = px[:max_index], px[max_index+1:]
        #left_ys, right_ys = np.array(py[:max_index]), np.array(py[max_index+1:])

        max_index_left, max_index_right = np.min(np.where(px == np.max(px))[0]), np.max(np.where(px == np.max(px))[0])
        left_xs, right_xs = px[:max_index_left + 1], px[max_index_right:]
        left_ys, right_ys = np.array(py[:max_index_left + 1]), np.array(py[max_index_right:])

        left_model, right_model = LinearRegression(), LinearRegression()

        left_xs, right_xs = np.expand_dims(left_xs, axis=1), np.expand_dims(right_xs, axis=1)
        left_ys, right_ys = np.expand_dims(left_ys, axis=1), np.expand_dims(right_ys, axis=1)

        left_model.fit(left_ys, left_xs)
        pred_left_xs = left_model.predict(left_ys)

        right_model.fit(right_ys, right_xs)
        pred_right_xs = right_model.predict(right_ys)

        x_max = sum(pred_left_xs[-1], pred_right_xs[0]) / 2
        y_max = (left_ys[-1] + right_ys[0]) / 2
        if pred_left_xs[-1] < pred_right_xs[0]:
            diff_right = pred_right_xs[0] - x_max 
            right_xmin = pred_right_xs[-1] - diff_right
            diff_left = x_max - pred_left_xs[-1]
            left_xmin = pred_left_xs[0] + diff_left
        else:
            diff_left = pred_left_xs[-1] - x_max 
            left_xmin = pred_left_xs[0] - diff_left
            diff_right = x_max - pred_right_xs[0]
            right_xmin = pred_right_xs[-1] + diff_right

        diff_right = right_ys[0] - y_max
        diff_left = y_max - left_ys[-1]
        left_ymin, right_ymin = left_ys[0], right_ys[-1]
        left_ymin = left_ymin + diff_left
        right_ymin = right_ymin - diff_right

        return list(map(int, [left_ymin.item(), left_xmin.item(), y_max.item(), \
                x_max.item(), right_ymin.item(), right_xmin.item()]))

    def minmax_CA(self, px, py):
        px = np.array(px)
        tmp = deepcopy(px)
        max_px = np.max(px)
        px[px==max_px] += 2
        tmp[tmp==max_px] = 0
        px[px==np.max(tmp)] += 1

        coefficients = np.polyfit(np.array(py), np.array(px), 4)
        p = np.poly1d(coefficients)

        l_b, h_b = int(len(py) * 0.2), int(len(py) * 0.8)
        py = py[l_b:h_b]
        p_fit = p(py)
        
        max_index = int(np.mean(np.where(p_fit == np.max(p_fit))[0]))
        x_max = p_fit[max_index]
        y_max = py[max_index]
    
        left_xmin, right_xmin = np.min(p_fit[:max_index+1]), np.min(p_fit[max_index:])
        left_index = np.where(p_fit[:max_index+1] == left_xmin)[0].mean()
        left_ymin = py[:max_index+1][int(left_index)]

        right_index = np.where(p_fit[max_index:] == right_xmin)[0].mean()
        right_ymin = py[max_index:][int(right_index)]

        return left_ymin, left_xmin, y_max, x_max, right_ymin, right_xmin
    
    def det_ca(self, ca_image, image_size = 224):
        
        result = {}
        # plt.imsave("tmp/tmp.png", ca_image, cmap="gray")
        # ca_image = cv2.imread
        ori_height, ori_width = ca_image.shape
        ca_image = cv2.cvtColor(gray_to_rgb(ca_image), cv2.COLOR_RGB2GRAY)
        images = get_unet_processor(image_size)(ca_image)
        _, new_height, new_width = images.shape
        outputs = self.ca_seg.run(None, {"input": images.unsqueeze(0).numpy()})[0]
        outputs = torch.argmax(torch.softmax(torch.tensor(outputs), dim=1), dim=1)
        outputs = outputs.data.cpu().numpy().squeeze()
        restored_mask = zoom(outputs, (ori_height / new_height, ori_width / new_width), order=0)
        restored_mask = np.clip(restored_mask, 0, 2)
        restored_mask[restored_mask == 2] = 0
        restored_mask = restored_mask.astype(np.uint8)

        ### post-process
        # CA_mask = post_process_seg(restored_mask)
        CA_mask = cv2.blur(restored_mask, (3, 3))
        CA_mask = post_process_seg(CA_mask, 20)
        xs, ys = np.where(CA_mask == 1)
        ys = set(ys.tolist())
        ys = sorted(ys)
        px, py =[], []
        for y in ys:
            x_tmp = np.where(CA_mask[:, y] == 1)[0]
            if len(x_tmp) != 0:
                px.append(x_tmp.min())
                py.append(y)

        '''------最高最低点-------'''
        #left_ymin, left_xmin, y_max, x_max, right_ymin, right_xmin = self.tange_CA(px, py, ca_image)
        left_ymin, left_xmin, y_max, x_max, right_ymin, right_xmin = self.fit_CA(px, py)
        #left_ymin, left_xmin, y_max, x_max, right_ymin, right_xmin = self.minmax_CA(px, py)
        '''------最高最低点-------'''
        A, B, C = np.array([left_ymin, left_xmin]), np.array([y_max, x_max]), np.array([right_ymin, right_xmin])
        BA, BC = A - B, C - B
        dot_product = np.dot(BA, BC)

        norm_BA = np.linalg.norm(BA)
        norm_BC = np.linalg.norm(BC)
        cos_theta = dot_product / (norm_BA * norm_BC)
        cos_theta = np.clip(cos_theta, -1, 1)
            
        theta_radians = np.arccos(cos_theta)
            
        theta_degrees = np.degrees(theta_radians)

        result["data"] = theta_degrees
        result["points"] = np.round([[left_ymin, left_xmin], [y_max, x_max], [right_ymin, right_xmin]]).astype(np.int16).tolist()
       #plt.plot([left_ymin, y_max, right_ymin], [left_xmin, x_max, right_xmin])
        return result, ca_image

    def det_bvr_zei(self, bvr_image, mid_line, image_size):
        #result = {"skull height":[], "lateral ventricles height":[], "brain above ventricles":[]}

        ori_height, ori_width  = bvr_image.shape
        bvr_image = cv2.cvtColor(gray_to_rgb(bvr_image), cv2.COLOR_RGB2GRAY)
        outputs = self.bvr_seg.run(None, {"input": get_unet_processor(image_size)(bvr_image).unsqueeze(0).numpy()})[0]

        outputs = torch.argmax(torch.softmax(torch.tensor(outputs), dim=1), dim=1)
        outputs = outputs.data.cpu().numpy().squeeze()
        restored_mask = zoom(outputs, (ori_height / image_size, ori_width / image_size), order=0)
        restored_mask = np.clip(restored_mask, 0, 2)
        restored_mask = cv2.medianBlur(restored_mask.astype(np.uint8), 3)

        #restored_mask = post_process_seg(restored_mask.astype(np.uint8))
        seg_image = restored_mask.copy()
        original_image = gray_to_rgb(bvr_image.copy())
        #plt.imshow(original_image)
        ### for Highest line ###
        head_height_indexes = np.argwhere(seg_image[:, mid_line] == 2)
        head_height = head_height_indexes.max() - head_height_indexes.min()

        #result["skull height"] = [(mid_line, head_height_indexes.max()), (mid_line, head_height_indexes.min())]
        #plt.plot([mid_line, mid_line], [head_height_indexes.max(), head_height_indexes.min()],  marker = 'o', color = 'r', markersize = 1)

        centroid = np.argwhere(seg_image == 1)
        sorted_indices = np.argsort(centroid[:, 1])
        sorted_array = centroid[sorted_indices]

        l = len(sorted_array) // 2
        left_ps, right_ps = sorted_array[:l], sorted_array[l:]

        left_xs = left_ps[:, 0]
        min_left_x, max_left_x = np.min(left_xs), np.max(left_xs)
        min_left_y = left_ps[np.argwhere(left_ps[:,0] == min_left_x)][:, :, 1].mean()
        ### find min xs -> ys ###

        right_xs = right_ps[:, 0]
        min_right_x, max_right_x = np.min(right_xs), np.max(right_xs)
        min_right_y = round(right_ps[np.argwhere(right_ps[:,0] == min_right_x)][:, :, 1].mean())

        ### find centorid height ###
        if (max_left_x - min_left_x) > (max_right_x - min_right_x):
            pos_x, pos_y = min_left_x, min_left_y
            max_x = max_left_x
        else:
            pos_x, pos_y = min_right_x, min_right_y
            max_x = max_right_x

        centroid_height = max_x - pos_x
        #result["lateral ventricles height"] = [(pos_y, pos_x), (pos_y, max_x)]
        #plt.plot([pos_y, pos_y], [pos_x, max_x],  marker = 'o', color = 'y', markersize = 1)

        ### calculate zEI ###
        zEI = centroid_height / head_height
        ### calculate BVR ###
        head_x = np.argwhere(seg_image[:, int(pos_y)] == 2).min()
        head_gap = pos_x - head_x
        BVR = head_gap / centroid_height

        #result["brain above ventricles"] = [(pos_y, pos_x), (pos_y, head_x)]
        #plt.plot([pos_y, pos_y], [pos_x, head_x],  marker = 'o', color = 'b', markersize = 1)
        result = {}
        result["zEI"] = {"data":zEI, "line_1": np.round([[pos_y, pos_x], [pos_y, max_x]]).astype(np.int16).tolist(), 
            "line_2":np.round([[mid_line, head_height_indexes.max()], [mid_line, head_height_indexes.min()]]).astype(np.int16).tolist()}
        result["BVR"] = {"data":BVR, "line_1":np.round([[pos_y, pos_x], [pos_y, max_x]]).astype(np.int16).tolist(), 
                         "line_2":np.round([[pos_y, pos_x], [pos_y, head_x]]).astype(np.int16).tolist()}
        #print(f"zEI: {centroid_height} mm / {head_height} mm = {zEI} \n BVR: {head_gap} mm / {centroid_height} mm = {BVR}")

        return result, bvr_image