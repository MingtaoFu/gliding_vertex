# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.poly_nms.poly_nms import poly_nms
import numpy as np
import cv2

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

def minAreaRect( poly ):
    rbox = cv2.minAreaRect( poly.reshape( (4, 2) ).astype( np.int32 ) )
    poly = cv2.boxPoints( rbox ).reshape( (8, ) )

    xmin = np.min( poly[ ::2] )
    ymin = np.min( poly[1::2] )
    xmax = np.max( poly[ ::2] )
    ymax = np.max( poly[1::2] )
    poly2 = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

    x1 = poly[0]
    y1 = poly[1]
    x2 = poly[2]
    y2 = poly[3]
    x3 = poly[4]
    y3 = poly[5]
    x4 = poly[6]
    y4 = poly[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]
    # return rbox_p



def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    """
    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
    """

    from maskrcnn_benchmark.data.transforms.build import build_transforms
    from PIL import Image
    import torchvision.transforms.functional as F
    transform = build_transforms( cfg, is_train=False )

    img_dir = "demo_imgs"
    res_dir = "demo_res"
    model.eval()
    imgs = os.listdir( img_dir )
    for img in imgs:
        img_path = os.path.join( img_dir, img )
        img_pil = Image.open( img_path )
        # for i in range( 360 ):
        original_img = img_pil
        # original_img = F.rotate( img_pil, 45, expand=True )

        origin_w, origin_h = original_img.size
        img, target = transform( original_img, None )
        print(img.shape)
        img = img.view( (1, img.shape[0], img.shape[1], img.shape[2] ) )
        h, w = img.shape[2:]
        if h % 32 != 0:
            new_h = ( h // 32 + 1 ) * 32
        else:
            new_h = h
        if w % 32 != 0:
            new_w = ( w // 32 + 1 ) * 32
        else:
            new_w = w

        ratio_w = 1. * new_w / w
        ratio_h = 1. * new_h / h

        padded_img = torch.zeros( (1, 3, new_h, new_w)).float()
        padded_img[:, :, :h, :w] = img

        prediction = model( padded_img.cuda() )[0]
        prediction = prediction.resize((origin_w * ratio_w, origin_h * ratio_h))
        hboxes = prediction.bbox.cpu()
        rboxes = prediction.get_field( "rboxes" ).cpu()
        ratios = prediction.get_field( "ratios" ).cpu()
        scores = prediction.get_field( "scores" ).cpu()
        # labels = prediction.get_field( "labels" ).cpu()

        for rbox, ratio, score in zip( rboxes, ratios, scores ):
            print( rbox )
            print( ratio, score )

        h_idx = ratios > 0.8
        # print(hboxes)
        h = hboxes[h_idx]
        hboxes_vtx = torch.stack( [h[:, 0], h[:, 1], h[:, 2], h[:, 1], h[:, 2], h[:, 3], h[:, 0], h[:, 3]] ).permute( 1, 0 )
        rboxes[h_idx] = hboxes_vtx
        # rboxes = rboxes.data.numpy().astype( np.int32 )
        rboxes = rboxes.data.numpy()
        
        keep = poly_nms( np.hstack( [rboxes, scores.cpu().data.numpy()[:, np.newaxis]] ).astype( np.double ), 0.1 )

        rboxes = rboxes[keep].astype( np.int32 )
        scores = scores[keep]
        hboxes = hboxes[keep]

        keep = np.where( scores > 0.6 )
        rboxes = rboxes[keep]
        scores = scores[keep].tolist()
        hboxes = hboxes[keep]

        # rboxes = list( map( minAreaRect, rboxes ) )
        if len( rboxes ) > 0:
            rboxes = np.vstack( rboxes )
        else:
            rboxes = np.array( rboxes )

        # vis( img_info["file_name"], rboxes )

        # img = cv2.imread( original_img )
        img = np.array( original_img.convert( "RGB" ) )[:, :, ::-1].copy()
        cv2.polylines( img, rboxes.reshape(-1, 4, 2).astype( np.int32 ), True, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA )
        filename = img_path.split( "/" )[-1]
        cv2.imwrite( "{}/{}".format( res_dir, filename ), img )
        # cv2.imwrite( "{}/{:0>3d}.png".format( res_dir, i ), img )




if __name__ == "__main__":
    main()
