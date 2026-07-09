#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, Optional
from .utils import preprocess_utils as pre_utils
from .utils import postprocess_utils as post_utils

class YoloV11_Pose:
    def __init__(self, opt):
        import hbm_runtime
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]
        self.score_thres = opt.score_thres
        self.conf_thres_raw = -np.log(1 / self.score_thres - 1)
        self.nms_thresh = opt.score_thres
        self.resize_type = 1
        self.reg = 16
        self.strides = [8, 16, 32]
        self.anchor_sizes = [80, 40, 20]
        self.weights_static = np.arange(self.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

    def set_scheduling_params(self, priority: Optional[int] = None, bpu_cores: Optional[list[int]] = None):
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        resize_img = pre_utils.resized_image(img, self.input_W, self.input_H, self.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))
        return {self.model_name: {self.input_names[0]: nv12}}

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self, outputs: Dict[str, np.ndarray], img_h: int, img_w: int):
        all_dbboxes, all_scores, all_ids, all_kpts_xy, all_kpts_score = [], [], [], [], []
        fp32_outputs = post_utils.dequantize_outputs(outputs, self.output_quants)
        for i, (stride, anchor_size) in enumerate(zip(self.strides, self.anchor_sizes)):
            cls_key = self.output_names[3 * i]
            box_key = self.output_names[3 * i + 1]
            kpts_key = self.output_names[3 * i + 2]
            scores, ids, valid_indices = post_utils.filter_classification(fp32_outputs[cls_key], self.conf_thres_raw)
            dbboxes = post_utils.decode_boxes(fp32_outputs[box_key], valid_indices, anchor_size, stride, self.weights_static)
            kpts_xy, kpts_score = post_utils.decode_kpts(fp32_outputs[kpts_key], valid_indices, anchor_size, stride)
            all_dbboxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)
            all_kpts_xy.append(kpts_xy)
            all_kpts_score.append(kpts_score)
        dbboxes = np.concatenate(all_dbboxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)
        kpts_xy = np.concatenate(all_kpts_xy, axis=0)
        kpts_score = np.concatenate(all_kpts_score, axis=0)
        keep = post_utils.NMS(dbboxes, scores, ids, self.nms_thresh)
        xyxy = post_utils.scale_coords_back(dbboxes[keep], img_w, img_h, self.input_W, self.input_H, self.resize_type)
        kpts_xy, kpts_score = post_utils.scale_keypoints_to_original_image(
            kpts_xy[keep], kpts_score[keep], xyxy, img_w, img_h, self.input_W, self.input_H, self.resize_type
        )
        return ids[keep], scores[keep], xyxy, kpts_xy, kpts_score

# ---------- 对外提供的接口函数 ----------
def init_pose_model(opt):
    """初始化姿态模型并设置调度参数"""
    model = YoloV11_Pose(opt)
    model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
    # 打印模型信息（可选）
    from .utils import common_utils as common
    common.print_model_info(model.model)
    return model

def infer_pose(model, frame, img_h, img_w):
    """执行一次姿态推理，返回 (ids, scores, boxes, kpts_xy, kpt_score)"""
    input_tensor = model.pre_process(frame)
    outputs = model.forward(input_tensor)
    return model.post_process(outputs, img_h, img_w)