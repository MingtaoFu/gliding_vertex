import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm
import cv2
import numpy as np

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.poly_nms.poly_nms import poly_nms
from maskrcnn_benchmark.config import cfg

def write( output_folder, pred_dict ):
    output_folder_txt = os.path.join( output_folder, "results" )
    if not os.path.exists( output_folder_txt ):
        os.mkdir( output_folder_txt )
    for key in pred_dict:
        detections = pred_dict[key]
        output_path = os.path.join( output_folder, "Task1_" + key + ".txt")
        with open(output_path, "w") as f:
            for det in detections:
                row = '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                det[0], det[1],
                det[2], det[3],
                det[4], det[5],
                det[6], det[7],
                det[8], det[9]
                )
                f.write(row)

def handle_ratio_prediction(prediction):
    hboxes = prediction.bbox.data.numpy()
    rboxes = prediction.get_field( "rboxes" ).data.numpy()
    ratios = prediction.get_field( "ratios" ).data.numpy()
    scores = prediction.get_field( "scores" ).data.numpy()
    labels = prediction.get_field( "labels" ).data.numpy()


    h_idx = np.where(ratios > 0.8)[0]
    h = hboxes[h_idx]
    hboxes_vtx = np.vstack( [h[:, 0], h[:, 1], h[:, 2], h[:, 1], h[:, 2], h[:, 3], h[:, 0], h[:, 3]] ).transpose((1,0))
    rboxes[h_idx] = hboxes_vtx
    keep = poly_nms( np.hstack( [rboxes, scores[:, np.newaxis]] ).astype( np.double ), 0.1 )

    rboxes = rboxes[keep].astype( np.int32 )
    scores = scores[keep]
    labels = labels[keep]

    if len( rboxes ) > 0:
        rboxes = np.vstack( rboxes )
        return rboxes, scores, labels
    else:
        return None, None, None

def do_dota_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    pred_dict = {label:[] for label in dataset.categories.values()}
    for image_id, prediction in tqdm( enumerate(predictions) ):
        original_id = dataset.id_to_img_map[image_id]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        if cfg.MODEL.RATIO_ON:
            rboxes, scores, labels = handle_ratio_prediction(prediction)
        else:
            raise NotImplementedError
        if rboxes is None:
            continue

        # img_name = img_info["file_name"].split( "/" )[-1].split( "." )[0]
        img_name = os.path.basename( img_info["file_name"] )[:-4]

        for rbox, score, label in zip(rboxes, scores, labels):
            json_label = dataset.contiguous_category_id_to_json_id[label]
            json_label = dataset.categories[json_label]
            object_row = rbox.tolist()
            object_row.insert(0, score)
            object_row.insert(0, img_name)
            pred_dict[json_label].append(object_row)

    write( output_folder, pred_dict )


