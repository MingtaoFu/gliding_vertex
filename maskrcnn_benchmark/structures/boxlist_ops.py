# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

import math

from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms

def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications

"""
#import maskrcnn_benchmark.utils.cython_bbox as cython_bbox
#bbox_overlaps = cython_bbox.bbox_overlaps
import numpy as np
"""

def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """

    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    # device = box1.device

    # iou = bbox_overlaps( box1.cpu().numpy(), box2.cpu().numpy()).astype( np.float16 )
    # iou = torch.from_numpy( iou )

    #lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    #rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    TO_REMOVE = 1
    BLOCK = 1073741824 // 28
    mat_size = M*N*2
    num_block = math.ceil(mat_size / BLOCK)
    step = math.ceil(N / num_block)
    iou_block_list = []
    for i in range(num_block):
        start_indx = i*step
        end_indx = min([(i+1)*step,N])
        #lt_block = lt[start_indx:end_indx,:]
        #rb_block = rb[start_indx:end_indx,:]
        lt_block = torch.max(box1[start_indx:end_indx, None, :2], box2[:, :2])
        rb_block = torch.min(box1[start_indx:end_indx, None, 2:], box2[:, 2:])
       
        wh_block = (rb_block - lt_block + TO_REMOVE).clamp(min=0)
        del lt_block, rb_block
        inter_block = wh_block[:, :, 0] * wh_block[:, :, 1]
        del wh_block
        iou_block = inter_block / (area1[start_indx:end_indx, None] + area2 - inter_block)
        del inter_block
        iou_block_list.append(iou_block.detach())

    iou = torch.cat(iou_block_list, dim=0)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
