# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        crop = cfg.INPUT.CROP_TRAIN
        rect = cfg.INPUT.RECT_TRAIN
        square_rotate = cfg.INPUT.SQUARE_ROTATE_TRAIN
        sort_vertices = cfg.INPUT.SORT_VERTICES
        rotate90_prob = cfg.INPUT.ROTATE90_PROB_TRAIN
        random_rotate_on = cfg.INPUT.RANDOM_ROTATE_ON
        flip_horizontal_prob = cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        crop = False
        rect = False
        square_rotate = False
        sort_vertices = False
        sort_quadio_vertices = False
        random_rotate_on = False
        rotate90_prob = 0.0
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            # NOTE Mingtao
            # T.RandomSampleCrop( crop ),
            T.ToRect( rect ),
            T.RandomSquareRotate( square_rotate ),
            T.RandomRotate90( rotate90_prob ),
            T.RandomRotateAug(random_rotate_on),
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ],
        is_train
    )
    return transform
