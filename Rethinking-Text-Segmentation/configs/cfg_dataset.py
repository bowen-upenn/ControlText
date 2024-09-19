import os
import os.path as osp
import numpy as np
import copy

from easydict import EasyDict as edict

cfg = edict()
cfg.DATASET_MODE = None
cfg.LOADER_PIPELINE = []
cfg.LOAD_BACKEND_IMAGE = 'pil'
cfg.LOAD_IS_MC_IMAGE = False
cfg.TRANS_PIPELINE = []
cfg.NUM_WORKERS_PER_GPU = None
cfg.NUM_WORKERS = None
cfg.TRY_SAMPLE = None


#################################
#####      wukong_1of5      #####
#################################

cfg_wukong_1of5 = copy.deepcopy(cfg)
cfg_wukong_1of5.DATASET_NAME = 'wukong_1of5'
cfg_wukong_1of5.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/wukong_1of5/imgs'
cfg_wukong_1of5.IM_MEAN = [0.485, 0.456, 0.406]
cfg_wukong_1of5.IM_STD = [0.229, 0.224, 0.225]
cfg_wukong_1of5.SEGLABEL_IGNORE_LABEL = 999
cfg_wukong_1of5.CLASS_NUM = 2


#################################
#####      wukong_2of5      #####
#################################

cfg_wukong_2of5 = copy.deepcopy(cfg)
cfg_wukong_2of5.DATASET_NAME = 'wukong_2of5'
cfg_wukong_2of5.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/wukong_2of5/imgs'
cfg_wukong_2of5.IM_MEAN = [0.485, 0.456, 0.406]
cfg_wukong_2of5.IM_STD = [0.229, 0.224, 0.225]
cfg_wukong_2of5.SEGLABEL_IGNORE_LABEL = 999
cfg_wukong_2of5.CLASS_NUM = 2

#################################
#####      wukong_3of5      #####
#################################

cfg_wukong_3of5 = copy.deepcopy(cfg)
cfg_wukong_3of5.DATASET_NAME = 'wukong_3of5'
cfg_wukong_3of5.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/wukong_3of5/imgs'
cfg_wukong_3of5.IM_MEAN = [0.485, 0.456, 0.406]
cfg_wukong_3of5.IM_STD = [0.229, 0.224, 0.225]
cfg_wukong_3of5.SEGLABEL_IGNORE_LABEL = 999
cfg_wukong_3of5.CLASS_NUM = 2


#################################
#####      wukong_4of5      #####
#################################

cfg_wukong_4of5 = copy.deepcopy(cfg)
cfg_wukong_4of5.DATASET_NAME = 'wukong_4of5'
cfg_wukong_4of5.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/wukong_4of5/imgs'
cfg_wukong_4of5.IM_MEAN = [0.485, 0.456, 0.406]
cfg_wukong_4of5.IM_STD = [0.229, 0.224, 0.225]
cfg_wukong_4of5.SEGLABEL_IGNORE_LABEL = 999
cfg_wukong_4of5.CLASS_NUM = 2


#################################
#####      wukong_5of5      #####
#################################

cfg_wukong_5of5 = copy.deepcopy(cfg)
cfg_wukong_5of5.DATASET_NAME = 'wukong_5of5'
cfg_wukong_5of5.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/wukong_5of5/imgs'
cfg_wukong_5of5.IM_MEAN = [0.485, 0.456, 0.406]
cfg_wukong_5of5.IM_STD = [0.229, 0.224, 0.225]
cfg_wukong_5of5.SEGLABEL_IGNORE_LABEL = 999
cfg_wukong_5of5.CLASS_NUM = 2


#################################
#####      controltext      #####
#################################

cfg_controltext = copy.deepcopy(cfg)
cfg_controltext.DATASET_NAME = 'controltext'
cfg_controltext.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'anytext_laion_generated'
))
cfg_controltext.IM_MEAN = [0.485, 0.456, 0.406]
cfg_controltext.IM_STD = [0.229, 0.224, 0.225]
cfg_controltext.SEGLABEL_IGNORE_LABEL = 999
cfg_controltext.CLASS_NUM = 2


##############################
#####      Laion_p1      #####
##############################

cfg_laion_p1 = copy.deepcopy(cfg)
cfg_laion_p1.DATASET_NAME = 'laion_p1'
cfg_laion_p1.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/laion/laion_p1/imgs'
cfg_laion_p1.IM_MEAN = [0.485, 0.456, 0.406]
cfg_laion_p1.IM_STD = [0.229, 0.224, 0.225]
cfg_laion_p1.SEGLABEL_IGNORE_LABEL = 999
cfg_laion_p1.CLASS_NUM = 2

##############################
#####      Laion_p2      #####
##############################

cfg_laion_p2 = copy.deepcopy(cfg)
cfg_laion_p2.DATASET_NAME = 'laion_p2'
cfg_laion_p2.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/laion/laion_p2/imgs'
cfg_laion_p2.IM_MEAN = [0.485, 0.456, 0.406]
cfg_laion_p2.IM_STD = [0.229, 0.224, 0.225]
cfg_laion_p2.SEGLABEL_IGNORE_LABEL = 999
cfg_laion_p2.CLASS_NUM = 2


##############################
#####      Laion_p3      #####
##############################

cfg_laion_p3 = copy.deepcopy(cfg)
cfg_laion_p3.DATASET_NAME = 'laion_p3'
cfg_laion_p3.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/laion/laion_p3/imgs'
cfg_laion_p3.IM_MEAN = [0.485, 0.456, 0.406]
cfg_laion_p3.IM_STD = [0.229, 0.224, 0.225]
cfg_laion_p3.SEGLABEL_IGNORE_LABEL = 999
cfg_laion_p3.CLASS_NUM = 2


##############################
#####      Laion_p4      #####
##############################

cfg_laion_p4 = copy.deepcopy(cfg)
cfg_laion_p4.DATASET_NAME = 'laion_p4'
cfg_laion_p4.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/laion/laion_p4/imgs'
cfg_laion_p4.IM_MEAN = [0.485, 0.456, 0.406]
cfg_laion_p4.IM_STD = [0.229, 0.224, 0.225]
cfg_laion_p4.SEGLABEL_IGNORE_LABEL = 999
cfg_laion_p4.CLASS_NUM = 2


##############################
#####      Laion_p5      #####
##############################

cfg_laion_p5 = copy.deepcopy(cfg)
cfg_laion_p5.DATASET_NAME = 'laion_p5'
cfg_laion_p5.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/laion/laion_p5/imgs'
cfg_laion_p5.IM_MEAN = [0.485, 0.456, 0.406]
cfg_laion_p5.IM_STD = [0.229, 0.224, 0.225]
cfg_laion_p5.SEGLABEL_IGNORE_LABEL = 999
cfg_laion_p5.CLASS_NUM = 2

##############################
#####      imagenet      #####
##############################

cfg_imagenet = copy.deepcopy(cfg)
cfg_imagenet.DATASET_NAME = 'imagenet'
cfg_imagenet.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data',
    'ImageNet', 'ILSVRC2012'))
cfg_imagenet.CLASS_INFO_JSON = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data',
    'ImageNet', 'addon', 'ILSVRC2012', '1000nids.json'))
cfg_imagenet.IM_MEAN = [0.485, 0.456, 0.406]
cfg_imagenet.IM_STD = [0.229, 0.224, 0.225]
cfg_imagenet.CLASS_NUM = 1000

#############################
#####      textseg      #####
#############################

cfg_textseg = copy.deepcopy(cfg)
cfg_textseg.DATASET_NAME = 'textseg'
cfg_textseg.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'TextSeg'))
cfg_textseg.CLASS_NUM = 2
cfg_textseg.CLASS_NAME = [
    'background', 
    'text']
cfg_textseg.SEGLABEL_IGNORE_LABEL = 999
cfg_textseg.SEMANTIC_PICK_CLASS = 'all'
cfg_textseg.IM_MEAN = [0.485, 0.456, 0.406]
cfg_textseg.IM_STD = [0.229, 0.224, 0.225]
cfg_textseg.LOAD_IS_MC_SEGLABEL = True

##########################
#####    cocotext    #####
##########################

cfg_cocotext = copy.deepcopy(cfg)
cfg_cocotext.DATASET_NAME = 'coco_text'
cfg_cocotext.ROOT_DIR = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/other_anytext_data/COCO_Text/images'
cfg_cocotext.IM_MEAN = [0.485, 0.456, 0.406]
cfg_cocotext.IM_STD = [0.229, 0.224, 0.225]
cfg_cocotext.SEGLABEL_IGNORE_LABEL = 999
cfg_cocotext.CLASS_NUM = 2


##########################
#####    MTWI2018    #####
##########################

cfg_MTWI2018 = copy.deepcopy(cfg)
cfg_MTWI2018.DATASET_NAME = 'MTWI2018'
cfg_MTWI2018.ROOT_DIR = '/pool/bwjiang/datasets/AnyWord-3M/link_download/ocr_data/MTWI2018/images'
cfg_MTWI2018.IM_MEAN = [0.485, 0.456, 0.406]
cfg_MTWI2018.IM_STD = [0.229, 0.224, 0.225]
cfg_MTWI2018.SEGLABEL_IGNORE_LABEL = 999
cfg_MTWI2018.CLASS_NUM = 2


##########################
#####     ReCTS      #####
##########################

cfg_ReCTS = copy.deepcopy(cfg)
cfg_ReCTS.DATASET_NAME = 'ReCTS'
cfg_ReCTS.ROOT_DIR = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/other_anytext_data/ReCTS/images'
cfg_ReCTS.IM_MEAN = [0.485, 0.456, 0.406]
cfg_ReCTS.IM_STD = [0.229, 0.224, 0.225]
cfg_ReCTS.SEGLABEL_IGNORE_LABEL = 999
cfg_ReCTS.CLASS_NUM = 2

##########################
#####  icdar2017rctw #####
##########################

cfg_icdar2017rctw = copy.deepcopy(cfg)
cfg_icdar2017rctw.DATASET_NAME = 'icdar2017rctw'
cfg_icdar2017rctw.ROOT_DIR = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/other_anytext_data/icdar2017rctw/images'
cfg_icdar2017rctw.IM_MEAN = [0.485, 0.456, 0.406]
cfg_icdar2017rctw.IM_STD = [0.229, 0.224, 0.225]
cfg_icdar2017rctw.SEGLABEL_IGNORE_LABEL = 999
cfg_icdar2017rctw.CLASS_NUM = 2

##########################
#####      Art       #####
##########################

cfg_art = copy.deepcopy(cfg)
cfg_art.DATASET_NAME = 'art'
cfg_art.ROOT_DIR = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/other_anytext_data/Art/images'
cfg_art.IM_MEAN = [0.485, 0.456, 0.406]
cfg_art.IM_STD = [0.229, 0.224, 0.225]
cfg_art.SEGLABEL_IGNORE_LABEL = 999
cfg_art.CLASS_NUM = 2

##########################
#####      LSVT      #####
##########################

cfg_lsvt = copy.deepcopy(cfg)
cfg_lsvt.DATASET_NAME = 'LSVT'
cfg_lsvt.ROOT_DIR = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/other_anytext_data/LSVT/images'
cfg_lsvt.IM_MEAN = [0.485, 0.456, 0.406]
cfg_lsvt.IM_STD = [0.229, 0.224, 0.225]
cfg_lsvt.SEGLABEL_IGNORE_LABEL = 999
cfg_lsvt.CLASS_NUM = 2


##########################
#####     mlt2019    #####
##########################

cfg_mlt2019 = copy.deepcopy(cfg)
cfg_mlt2019.DATASET_NAME = 'mlt2019'
cfg_mlt2019.ROOT_DIR = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/other_anytext_data/mlt2019/images'
cfg_mlt2019.IM_MEAN = [0.485, 0.456, 0.406]
cfg_mlt2019.IM_STD = [0.229, 0.224, 0.225]
cfg_mlt2019.SEGLABEL_IGNORE_LABEL = 999
cfg_mlt2019.CLASS_NUM = 2


########################
#####    cocots    #####
########################

cfg_cocots = copy.deepcopy(cfg)
cfg_cocots.DATASET_NAME = 'cocots'
cfg_cocots.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'COCO'))
cfg_cocots.IM_MEAN = [0.485, 0.456, 0.406]
cfg_cocots.IM_STD = [0.229, 0.224, 0.225]
cfg_cocots.CLASS_NUM = 2
cfg_cocots.SEGLABEL_IGNORE_LABEL = 255
cfg_cocots.LOAD_BACKEND_SEGLABEL = 'pil'
cfg_cocots.LOAD_IS_MC_SEGLABEL = False

#####################
#####    mlt    #####
#####################

cfg_mlt = copy.deepcopy(cfg)
cfg_mlt.DATASET_NAME = 'mlt'
cfg_mlt.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'ICDAR17', 'challenge8'))
cfg_mlt.IM_MEAN = [0.485, 0.456, 0.406]
cfg_mlt.IM_STD = [0.229, 0.224, 0.225]
cfg_mlt.CLASS_NUM = 2
cfg_mlt.SEGLABEL_IGNORE_LABEL = 255

#######################
#####   icdar13   #####
#######################

cfg_icdar13 = copy.deepcopy(cfg)
cfg_icdar13.DATASET_NAME = 'icdar13'
cfg_icdar13.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'ICDAR13'))
cfg_icdar13.CLASS_NUM = 2
cfg_icdar13.CLASS_NAME = [
    'background', 
    'text']
cfg_icdar13.SEGLABEL_IGNORE_LABEL = 999
cfg_icdar13.SEMANTIC_PICK_CLASS = 'all'
cfg_icdar13.IM_MEAN = [0.485, 0.456, 0.406]
cfg_icdar13.IM_STD = [0.229, 0.224, 0.225]
cfg_icdar13.LOAD_BACKEND_SEGLABEL = 'pil'
cfg_icdar13.LOAD_IS_MC_SEGLABEL = False
cfg_icdar13.FROM_SOURCE = 'addon'
cfg_icdar13.USE_CACHE = False

#########################
#####   totaltext   #####
#########################

cfg_totaltext = copy.deepcopy(cfg)
cfg_totaltext.DATASET_NAME = 'totaltext'
cfg_totaltext.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'TotalText'))
cfg_totaltext.CLASS_NUM = 2
cfg_totaltext.CLASS_NAME = [
    'background', 
    'text']
cfg_totaltext.SEGLABEL_IGNORE_LABEL = 999 
# dummy, totaltext pixel level anno has no ignore label
cfg_totaltext.IM_MEAN = [0.485, 0.456, 0.406]
cfg_totaltext.IM_STD = [0.229, 0.224, 0.225]

#######################
#####   textssc   #####
#######################
# text semantic segmentation composed

cfg_textssc = copy.deepcopy(cfg)
cfg_textssc.DATASET_NAME = 'textssc'
cfg_textssc.ROOT_DIR = osp.abspath(osp.join(
    osp.dirname(__file__), '..', 'data', 'TextSSC'))
cfg_textssc.CLASS_NUM = 2
cfg_textssc.CLASS_NAME = [
    'background', 
    'text']
cfg_textssc.SEGLABEL_IGNORE_LABEL = 999 
cfg_textssc.IM_MEAN = [0.485, 0.456, 0.406]
cfg_textssc.IM_STD = [0.229, 0.224, 0.225]
cfg_textssc.LOAD_BACKEND_SEGLABEL = 'pil'
cfg_textssc.LOAD_IS_MC_SEGLABEL = False
