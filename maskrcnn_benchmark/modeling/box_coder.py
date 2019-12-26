# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch
import cv2


class RatioCoder(object):
    def __init__(self, bbox_xform_clip=math.log(1000. / 16)):
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, rbox):
        def polygon_area( corners ):
            n = len( corners ) # of corners
            area = 0.0
            for i in range( n ):
                j = ( i + 1 ) % n
                area += corners[i][0] * corners[j][1]
                area -= corners[j][0] * corners[i][1]
            area = abs( area ) / 2.0
            return area

        polygons = list( map( lambda x: x.polygons[0], rbox ) )
        rbox = torch.stack( polygons, axis=0 )

        max_x_, max_x_idx = rbox[:,  ::2].max( 1 )
        min_x_, min_x_idx = rbox[:,  ::2].min( 1 )
        max_y_, max_y_idx = rbox[:, 1::2].max( 1 )
        min_y_, min_y_idx = rbox[:, 1::2].min( 1 )

        rbox = rbox.view( (-1, 4, 2) )

        polygon_areas = list( map( polygon_area, rbox ) )
        polygon_areas = torch.stack( polygon_areas )
        hbox_areas = ( max_y_ - min_y_ + 1 ) * ( max_x_ - min_x_ + 1 )
        ratio_gt = polygon_areas / hbox_areas

        ratio_gt = ratio_gt.view( (-1, 1) )
        return ratio_gt

class FixCoder(object):
    def __init__(self, bbox_xform_clip=math.log(1000. / 16)):
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, rbox):
        polygons = list( map( lambda x: x.polygons[0], rbox ) )
        rbox = torch.stack( polygons, axis=0 )

        max_x_, max_x_idx = rbox[:,  ::2].max( 1 )
        min_x_, min_x_idx = rbox[:,  ::2].min( 1 )
        max_y_, max_y_idx = rbox[:, 1::2].max( 1 )
        min_y_, min_y_idx = rbox[:, 1::2].min( 1 )

        x_center = ( max_x_ + min_x_ ) / 2.
        y_center = ( max_y_ + min_y_ ) / 2.

        box = torch.stack( [min_x_, min_y_, max_x_, max_y_ ], axis=0 ).permute( 1, 0 )

        rbox = rbox.view( (-1, 4, 2) )

        rbox_ordered = torch.zeros_like( rbox )
        rbox_ordered[:, 0] = rbox[range(len(rbox)), min_y_idx]
        rbox_ordered[:, 1] = rbox[range(len(rbox)), max_x_idx]
        rbox_ordered[:, 2] = rbox[range(len(rbox)), max_y_idx]
        rbox_ordered[:, 3] = rbox[range(len(rbox)), min_x_idx]

        top   = rbox_ordered[:, 0, 0]
        right = rbox_ordered[:, 1, 1]
        down  = rbox_ordered[:, 2, 0]
        left  = rbox_ordered[:, 3, 1]

        """
        top = torch.min( torch.max( top, box[:, 0] ), box[:, 2] )
        right = torch.min( torch.max( right, box[:, 1] ), box[:, 3] )
        down = torch.min( torch.max( down, box[:, 0] ), box[:, 2] )
        left = torch.min( torch.max( left, box[:, 1] ), box[:, 3] )
        """

        top_gt = (top - box[:, 0]) / (box[:, 2] - box[:, 0])
        right_gt = (right - box[:, 1]) / (box[:, 3] - box[:, 1])
        down_gt = (box[:, 2] - down) / (box[:, 2] - box[:, 0])
        left_gt = (box[:, 3] - left) / (box[:, 3] - box[:, 1])

        hori_box_mask = ((rbox_ordered[:,0,1] - rbox_ordered[:,1,1]) == 0) + ((rbox_ordered[:,1,0] - rbox_ordered[:,2,0]) == 0)

        fix_gt = torch.stack( [top_gt, right_gt, down_gt, left_gt] ).permute( 1, 0 )
        fix_gt = fix_gt.view( (-1, 4) )
        fix_gt[hori_box_mask, :] = 1
        return fix_gt

    def decode(self, box, alphas):
        pred_top = (box[:, 2::4] - box[:, 0::4]) * alphas[:, 0::4] + box[:, 0::4]
        pred_right = (box[:, 3::4] - box[:, 1::4]) * alphas[:, 1::4] + box[:, 1::4]
        pred_down = (box[:, 0::4] - box[:, 2::4]) * alphas[:, 2::4] + box[:, 2::4]
        pred_left = (box[:, 1::4] - box[:, 3::4]) * alphas[:, 3::4] + box[:, 3::4]

        pred_rbox = torch.zeros( (box.shape[0], box.shape[1] * 2) )
        pred_rbox[:, 0::8] = pred_top
        pred_rbox[:, 1::8] = box[:, 1::4]
        pred_rbox[:, 2::8] = box[:, 2::4]
        pred_rbox[:, 3::8] = pred_right
        pred_rbox[:, 4::8] = pred_down
        pred_rbox[:, 5::8] = box[:, 3::4]
        pred_rbox[:, 6::8] = box[:, 0::4]
        pred_rbox[:, 7::8] = pred_left
        return pred_rbox

class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes

