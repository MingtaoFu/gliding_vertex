# encoding=utf-8
"""
@author Mingtao Fu
"""

import os
import json
from PIL import Image
import numpy as np
import random

np.random.seed( 3 )
random.seed( 3 )

categories = [
    {
        "supercategory": "none",
        "id": 1,
        "name": "plane"
    }, {
        "supercategory": "none",
        "id": 2,
        "name": "baseball-diamond"
    }, {
        "supercategory": "none",
        "id": 3,
        "name": "bridge"
    }, {
        "supercategory": "none",
        "id": 4,
        "name": "ground-track-field"
    }, {
        "supercategory": "none",
        "id": 5,
        "name": "small-vehicle"
    }, {
        "supercategory": "none",
        "id": 6,
        "name": "large-vehicle"
    }, {
        "supercategory": "none",
        "id": 7,
        "name": "ship"
    }, {
        "supercategory": "none",
        "id": 8,
        "name": "tennis-court"
    }, {
        "supercategory": "none",
        "id": 9,
        "name": "basketball-court"
    }, {
        "supercategory": "none",
        "id": 10,
        "name": "storage-tank"
    }, {
        "supercategory": "none",
        "id": 11,
        "name": "soccer-ball-field"
    }, {
        "supercategory": "none",
        "id": 12,
        "name": "roundabout"
    }, {
        "supercategory": "none",
        "id": 13,
        "name": "harbor"
    }, {
        "supercategory": "none",
        "id": 14,
        "name": "swimming-pool"
    }, {
        "supercategory": "none",
        "id": 15,
        "name": "helicopter"
    }
]

class_dict = {
        'plane': 1,
        'baseball-diamond': 2,
        'bridge': 3,
        'ground-track-field': 4,
        'small-vehicle': 5,
        'large-vehicle': 6,
        'ship': 7,
        'tennis-court': 8,
        'basketball-court': 9,
        'storage-tank': 10,
        'soccer-ball-field': 11,
        'roundabout': 12,
        'harbor': 13,
        'swimming-pool': 14,
        'helicopter': 15
        }


def polygon_area( corners ):
    n = len( corners ) # of corners
    area = 0.0
    for i in range( n ):
        j = ( i + 1 ) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs( area ) / 2.0
    return area

_image_id = 0
_gt_id = 0

def txt2json( src, key, obj ):
    global _image_id
    global _gt_id

    img_src = os.path.join( src, "images", key + ".png" )
    width, height = Image.open( img_src ).size
    file_name = os.path.basename( img_src )
    _image_id += 1

    obj["images"].append({
        "file_name": file_name,
        "width": width,
        "height": height,
        "id": _image_id
    })

    txt_src = os.path.join( src, "labelTxt", key + ".txt" )

    if not os.path.exists( txt_src):
        return 

    with open( txt_src ) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            splits = line.split( " " )
            x1 = int( float( splits[0] ) )
            y1 = int( float( splits[1] ) )
            x2 = int( float( splits[2] ) )
            y2 = int( float( splits[3] ) )
            x3 = int( float( splits[4] ) )
            y3 = int( float( splits[5] ) )
            x4 = int( float( splits[6] ) )
            y4 = int( float( splits[7] ) )
            xmin = min( x1, x2, x3, x4 )
            xmax = max( x1, x2, x3, x4 )
            ymin = min( y1, y2, y3, y4 )
            ymax = max( y1, y2, y3, y4 )
            name = splits[8]
            diff = int( splits[9] )

            category_id = class_dict[name]
            _gt_id += 1
            segmentation = [x1, y1, x2, y2, x3, y3, x4, y4]
            corners = [( x1, y1 ), ( x4, y4 ), ( x3, y3 ), ( x2, y2 )]
            gt = {
                "ignore": diff,
                "segmentation": [segmentation],
                "area": polygon_area( corners ),
                "iscrowd": 0,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "image_id": _image_id,
                "category_id": category_id,
                "id": _gt_id
            }

            obj["annotations"].append( gt )


def convert( imagelist, src, dst ):
    obj = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": categories
    }

    for k in imagelist:
        for key in imagelist[k]:
            txt2json( src, key, obj )
    with open( dst, "w" ) as f:
        json.dump( obj, f )

def collect_unaug_dataset( txtdir ):
    txts = os.listdir( txtdir )

    img_dic = {}
    for cls in class_dict:
        img_dic[cls] = []

    for txt in txts:
        dic = {}
        for cls in class_dict:
            dic[cls] = False

        with open( os.path.join( txtdir, txt ) ) as f:
            lines = f.readlines()
            for line in lines:
                cls = line.split( " " )[-2]
                dic[cls] = True
            for key in dic:
                if dic[key]:
                    img_dic[key].append( txt[:-4] )
    return img_dic

