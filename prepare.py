import os
import json
from DOTA_devkit.ImgSplit_multi_process import splitbase
from DOTA_devkit.SplitOnlyImage_multi_process import splitbase as splitbase_onlyimg
from txt2json import collect_unaug_dataset, convert

with open( "config.json" ) as config_f:
    CONFIG = json.load( config_f )
O_ROOT = CONFIG["original_root_dir"]
T_ROOT = CONFIG["target_root_dir"]
SETs = CONFIG["sets"]

# split
for SET in SETs:
    if SET["only_img"]:
        if not os.path.exists( os.path.join( T_ROOT, SET["name"] + "_cut", "images" ) ):
            os.makedirs( os.path.join( T_ROOT, SET["name"] + "_cut", "images" ) )
        split = splitbase_onlyimg( os.path.join( O_ROOT, SET["name"], "images" ), os.path.join( T_ROOT, SET["name"] + "_cut", "images" ), gap=SET["gap"], subsize=1024, num_process=8, padding=False )
    else:
        split = splitbase( os.path.join( O_ROOT, SET["name"] ), os.path.join( T_ROOT, SET["name"] + "_cut" ), gap=SET["gap"], subsize=1024, num_process=8, padding=False )
    split.splitdata( 1 )
    split.splitdata( 0.5 )

# class balancing of training set
# NOTE This may be not optimal, and you can choose a different strategy.
img_dic = collect_unaug_dataset( os.path.join( T_ROOT, "trainval_cut", "labelTxt" ) )
img_dic["storage-tank"] = img_dic["storage-tank"] + img_dic["storage-tank"][:526]
img_dic["baseball-diamond"] = img_dic["baseball-diamond"] * 2 + img_dic["baseball-diamond"][:202]
img_dic["ground-track-field"] = img_dic["ground-track-field"] + img_dic["ground-track-field"][:575]
img_dic["swimming-pool"] = img_dic["swimming-pool"] * 2 + img_dic["swimming-pool"][:104]
img_dic["soccer-ball-field"] = img_dic["soccer-ball-field"] + img_dic["soccer-ball-field"][:962]
img_dic["roundabout"] = img_dic["roundabout"] + img_dic["roundabout"][:711]
img_dic["tennis-court"] = img_dic["tennis-court"] + img_dic["tennis-court"][:655]

img_dic["basketball-court"] = img_dic["basketball-court"] * 4
img_dic["helicopter"] = img_dic["helicopter"] * 8

convert( img_dic, os.path.join( T_ROOT, "trainval_cut" ),  os.path.join( T_ROOT, "trainval_cut", "trainval_cut.json" ) )
img_dic_test = {"all": [i[:-4] for i in os.listdir( os.path.join( T_ROOT, "test_cut", "images" ) )]}
convert( img_dic_test, os.path.join( T_ROOT, "test_cut" ),  os.path.join( T_ROOT, "test_cut", "test_cut.json" ) )
