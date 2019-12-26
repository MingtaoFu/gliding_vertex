# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from PIL import Image
import os
import numpy as np
import cv2
from maskrcnn_benchmark.config import cfg


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # remove ignored or crowd box
    anno = [obj for obj in anno if obj["iscrowd"] == 0 and obj["ignore"] == 0 ]
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class DOTADataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(DOTADataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        # NOTE Mingtao: We need an extra attribute `angle`
        # so we cannot simply super it.
        # img, anno = super(TD500Dataset, self).__getitem__(idx)

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        loaded_img = coco.loadImgs(img_id)[0]
        path = loaded_img['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # if "angle" in loaded_img and loaded_img["angle"] is not 0:
        if 'angle' in loaded_img.keys() and loaded_img["angle"] is not 0:
            if loaded_img["angle"] == 90:
                img = img.rotate( 270, expand=True )
            elif loaded_img["angle"] == 180:
                img = img.rotate( 180, expand=True )
            elif loaded_img["angle"] == 270:
                img = img.rotate( 90, expand=True )
            else:
                raise ValueError()

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0 and obj["ignore"] == 0]

        """
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        """

        def to_rrect( x ):
            x = cv2.minAreaRect( x )
            x = cv2.boxPoints( x )
            return x

        # masks = [obj["segmentation"] for obj in anno]
        masks = np.array( [obj["segmentation"] for obj in anno] )

        rrects = list( map( to_rrect, masks.reshape( (-1, 4, 2) ) ) )
        rrects_np = np.array( rrects, dtype=np.float32 ).reshape( (-1, 8) )
        xmins = np.min( rrects_np[:,  ::2], axis=1 )
        ymins = np.min( rrects_np[:, 1::2], axis=1 )
        xmaxs = np.max( rrects_np[:,  ::2], axis=1 )
        ymaxs = np.max( rrects_np[:, 1::2], axis=1 )
        xyxy = np.vstack( [xmins, ymins, xmaxs, ymaxs] ).transpose()
        boxes = torch.from_numpy( xyxy ).reshape(-1, 4)  # guard against no boxes
        target = BoxList( boxes, img.size, mode="xyxy" )

        masks = SegmentationMask( rrects_np.reshape( (-1, 1, 8)).tolist(), img.size, mode='poly' )
        target.add_field( "masks", masks )

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # NOTE Qimeng: close it for getting correct alpha
        #target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
