# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
from shapely.geometry import Polygon
import math


class Compose(object):
    def __init__(self, transforms, training):
        self.transforms = transforms
        self.training = training

    def __call__(self, image, target):
        for t in self.transforms:
            if self.training:
                image, target = t(image, target)
            else:
                image = t(image)

        """
        if self.training:
            return image, target
        else:
            return image
        """
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class RandomRotate90( object ):
    def __init__( self, prob=0.5 ):
        self.prob = prob

    def __call__( self, image, target=None ):
        if target is None:
            return image
        if random.random() < self.prob:
            image = F.rotate( image, 90, expand=True )
            target = target.rotate90()
        return image, target

# (0, 90, 180, or 270)
class RandomRotateAug( object ):
    def __init__(self, random_rotate_on):
        self.random_rotate_on = random_rotate_on

    def __call__( self, image, target=None ):
        if target is None:
            return image
        if not self.random_rotate_on:
            return image, target
        indx = int(random.random() * 100) // 25
        image = F.rotate( image, 90 * indx, expand=True )
        for _ in range(indx):
            target = target.rotate90()
        return image, target


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        # NOTE Mingtao
        if w <= h:
          size = np.clip( size, int(w / 1.5), int(w * 1.5) )
        else:
          size = np.clip( size, int(h / 1.5), int(h * 1.5) )

        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target

# The image must be a square as we do not expand the image
class RandomSquareRotate( object ):
    def __init__( self, do=True ):
        self.do = do

    def __call__( self, image, target=None ):
        if target == None:
            return image
        if not self.do:
            return image, target

        w, h = image.size
        assert w == h

        cx = w / 2
        cy = cx

        degree = random.uniform(0, 360)
        radian = degree * math.pi / 180

        new_image = image.rotate( -degree )

        sin = math.sin( radian )
        cos = math.cos( radian )

        masks = target.get_field( "masks" )
        polygons = list( map( lambda x: x.polygons[0], masks.instances.polygons ) )
        polygons = torch.stack( polygons, 0 ).reshape( (-1, 2) ).t()

        M = torch.Tensor([[cos, -sin], [sin, cos]])
        b = torch.Tensor([[(1 - cos) * cx + cy * sin], [(1 - cos) * cy - cx * sin]])
        new_points = M.mm( polygons ) + b
        new_points = new_points.t().reshape( (-1, 8) )
        xmins, _ = torch.min( new_points[:,  ::2], 1 )
        ymins, _ = torch.min( new_points[:, 1::2], 1 )
        xmaxs, _ = torch.max( new_points[:,  ::2], 1 )
        ymaxs, _ = torch.max( new_points[:, 1::2], 1 )
        boxes = torch.stack( [xmins, ymins, xmaxs, ymaxs], 1 ).reshape((-1, 4))

        new_target = BoxList( boxes, image.size, mode="xyxy" )
        new_target._copy_extra_fields( target )
        new_masks = SegmentationMask( new_points.reshape((-1, 1, 8)).tolist(), image.size, mode='poly' )
        new_target.add_field( "masks", new_masks )

        return new_image, new_target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if target is None:
            return image
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if target is None:
            return image
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target=None):
        if target == None:
            return image
        image = self.color_jitter(image)
        return image, target

from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.bounding_box import BoxList
class ToRect( object ):
    def __init__( self, do=True ):
        self.do=do

    @staticmethod
    def _to_rrect( x ):
        x = cv2.minAreaRect( x )
        x = cv2.boxPoints( x )
        return x

    def __call__( self, image, target=None ):
        if target is None:
            return image

        if not self.do:
            return image, target

        masks = target.get_field( "masks" )
        polygons = list( map( lambda x: x.polygons[0].numpy(), masks.instances.polygons ) )
        polygons = np.stack( polygons, axis=0 ).reshape( (-1, 4, 2) )
        rrects = list( map( self._to_rrect, polygons ) )

        rrects_np = np.array( rrects, dtype=np.float32 ).reshape( (-1, 8) )
        xmins = np.min( rrects_np[:,  ::2], axis=1 )
        ymins = np.min( rrects_np[:, 1::2], axis=1 )
        xmaxs = np.max( rrects_np[:,  ::2], axis=1 )
        ymaxs = np.max( rrects_np[:, 1::2], axis=1 )
        xyxy = np.vstack( [xmins, ymins, xmaxs, ymaxs] ).transpose()
        boxes = torch.from_numpy( xyxy ).reshape(-1, 4)  # guard against no boxes

        new_target = BoxList( boxes, image.size, mode="xyxy" )
        new_target._copy_extra_fields( target )
        new_masks = SegmentationMask( rrects_np.reshape( (-1, 1, 8)).tolist(), image.size, mode='poly' )
        new_target.add_field( "masks", new_masks )

        return image, new_target

class SortForQuad( object ):
    def __init__( self, do=True ):
        self.do=do

    @staticmethod
    def choose_best_pointorder_fit_another(poly1):
        x1 = poly1[0]
        y1 = poly1[1]
        x2 = poly1[2]
        y2 = poly1[3]
        x3 = poly1[4]
        y3 = poly1[5]
        x4 = poly1[6]
        y4 = poly1[7]

        xmin = min( x1, x2, x3, x4 )
        ymin = min( y1, y2, y3, y4 )
        xmax = max( x1, x2, x3, x4 )
        ymax = max( y1, y2, y3, y4 )
        poly2 = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

        combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                     np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
        dst_coordinate = np.array(poly2)
        distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
        sorted = distances.argsort()
        return combinate[sorted[0]].tolist()

    def __call__( self, image, target=None ):
        if target == None:
            return image
        if not self.do:
            return image, target

        masks = target.get_field( "masks" )
        polygons = list( map( lambda x: x.polygons[0].numpy(), masks.instances.polygons ) )
        polygons = np.stack( polygons, axis=0 ).reshape( (-1, 8) )

        new_polygons = []
        for polygon in polygons:
            new_polygon = self.choose_best_pointorder_fit_another( polygon )
            new_polygons.append( [new_polygon] )

        new_masks = SegmentationMask( new_polygons, image.size, mode='poly' )
        target.add_field( "masks", new_masks )

        return image, target


class RandomCrop( object ):
    def __init__( self, size ):
        self.crop_size = size

    def __call__( self, image, target ):
        width, height = image.size
        i = random.choice( np.arange( 0, height - self.crop_size[0] ) )
        j = random.choice( np.arange( 0, width - self.crop_size[1] ) )
        image = F.crop( image, i, j, self.crop_size[0], self.crop_size[1] ) #image[i:i+width,j+height,:]
        target_ = target.crop( ( j, i, j + self.crop_size[1], i + self.crop_size[0]) )
        return image, target_

class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is None:
            return image
        return image, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target
