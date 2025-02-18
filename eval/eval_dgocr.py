import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
from cldm.recognizer import TextRecognizer, crop_image
from easydict import EasyDict as edict
from anytext_singleGPU import load_data, get_item
from tqdm import tqdm
import os
import torch
import Levenshtein
import numpy as np
import math
import json
import argparse
PRINT_DEBUG = False
num_samples = 4
from paddleocr import PaddleOCR, draw_ocr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default='/tmp/datasets/AnyWord-3M/anytext_eval_imgs/anytext_v1.1_laion_generated',
        help='path of generated images for eval'
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default='/tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json',
        help='json path for evaluation dataset'
    )
    parser.add_argument(
        "--glyph_path",
        type=str,
        default='',
        help='json path for the glyphs of the evaluation dataset'
    )
    args = parser.parse_args()
    return args


def get_ld(ls1, ls2):
    # Convert lists of integers to strings
    str1 = ''.join(map(str, ls1))
    str2 = ''.join(map(str, ls2))

    # Calculate Levenshtein distance
    edit_dist = Levenshtein.distance(str1, str2)
    # edit_dist = Levenshtein.distance(ls1, ls2)
    return 1 - edit_dist/(max(len(ls1), len(ls2)) + 1e-5)


def pre_process(img_list, shape):
    numpy_list = []
    img_num = len(img_list)
    assert img_num > 0
    for idx in range(0, img_num):
        # rotate
        img = img_list[idx]
        h, w = img.shape[1:]
        if h > w * 1.2:
            img = torch.transpose(img, 1, 2).flip(dims=[1])
            img_list[idx] = img
            h, w = img.shape[1:]
        # resize
        imgC, imgH, imgW = (int(i) for i in shape.strip().split(','))
        assert imgC == img.shape[0]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(imgH, resized_w),
            mode='bilinear',
            align_corners=True,
        )
        # padding
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        numpy_list += [padding_im.permute(1, 2, 0).cpu().numpy()]  # HWC ,numpy
    return numpy_list


def main():
    args = parse_args()
    img_dir = args.img_dir
    json_path = args.json_path
    glyph_path = args.glyph_path

    if 'wukong' in json_path:
        model_lang = 'ch'
        rec_char_dict_path = os.path.join('./ocr_weights', 'ppocr_keys_v1.txt')
    elif 'laion' in json_path:
        rec_char_dict_path = os.path.join('./ocr_weights', 'en_dict.txt')

    ocr_ch = PaddleOCR(show_log=False, lang="ch")
    ocr_en = PaddleOCR(show_log=False, lang="en")

    predictor = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
    rec_image_shape = "3, 48, 320"
    args = edict()
    args.rec_image_shape = rec_image_shape
    args.rec_char_dict_path = rec_char_dict_path
    args.rec_batch_num = 1
    args.use_fp16 = False
    text_recognizer = TextRecognizer(args, None)

    if len(glyph_path) > 0:
        invalid_json_path = './Rethinking-Text-Segmentation/log/images/ocr_verified/invalid_gly_lines_test.json'
        with open(invalid_json_path, 'r') as f:
            invalid_glyph_lines = json.load(f)
        data_list = load_data(json_path, glyph_path)
    else:
        data_list = load_data(json_path)

    sen_acc = []
    edit_dist = []
    for i in tqdm(range(len(data_list)), desc='evaluate'):
        item_dict = get_item(data_list, i)
        img_name = item_dict['img_name'].split('.')[0]
        n_lines = item_dict['n_lines']

        if 'gly_lines' in img_dir:
            for j in range(num_samples):
                for k in range(n_lines):
                    img_path = os.path.join(img_dir, img_name + f"_{j}_{item_dict['texts'][k]}.jpg")
                    gt_texts = item_dict['texts'][k]

                    # find if image path exists
                    if not os.path.exists(img_path):
                        continue

                    if 'laion' in img_dir:
                        pred_texts = ocr_en.ocr(img_path, det=False, cls=False)[0][0][0]
                    else:
                        pred_texts = ocr_ch.ocr(img_path, det=False, cls=False)[0][0][0]
                    # print('pred_texts', pred_texts, 'gt_texts', gt_texts)

                    if pred_texts == gt_texts:
                        sen_acc += [1]
                    else:
                        sen_acc += [0]

                    gt_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars) - 1) for m in gt_texts]
                    pred_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars) - 1) for m in pred_texts]
                    # print('pred_order', pred_order, 'gt_order', gt_order)
                    edit_dist += [get_ld(pred_order, gt_order)]

        else:
            for j in range(num_samples):
                img_path = os.path.join(img_dir, img_name+f'_{j}.jpg')
                img = cv2.imread(img_path)
                if PRINT_DEBUG:
                    cv2.imwrite(f'{i}_{j}.jpg', img)
                img = torch.from_numpy(img)
                img = img.permute(2, 0, 1).float()  # HWC-->CHW
                gt_texts = []
                pred_texts = []

                if len(glyph_path) > 0:
                    glyph_path = item_dict['glyphs_path']
                for k in range(n_lines):  # line
                    if len(glyph_path) > 0 and glyph_path in invalid_glyph_lines:
                        for element in invalid_glyph_lines[glyph_path]:
                            if item_dict['texts'][k] in element['text']:
                                n_lines -= 1
                                continue

                    gt_texts += [item_dict['texts'][k]]
                    np_pos = (item_dict['positions'][k]*255.).astype(np.uint8)  # 0-1, hwc
                    pred_text = crop_image(img, np_pos)
                    pred_texts += [pred_text]

                if n_lines > 0:
                    pred_texts = pre_process(pred_texts, rec_image_shape)
                    preds_all = []
                    for idx, pt in enumerate(pred_texts):
                        if PRINT_DEBUG:
                            cv2.imwrite(f'{i}_{j}_{idx}.jpg', pt)
                        rst = predictor(pt)
                        preds_all += [rst['text'][0]]
                    for k in range(len(preds_all)):
                        pred_text = preds_all[k]
                        gt_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars)-1) for m in gt_texts[k]]
                        pred_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars)-1) for m in pred_text]
                        # print('gt_texts[k]', gt_texts[k], 'pred_text', pred_text, 'gt_order', gt_order, 'pred_order', pred_order)
                        if pred_text == gt_texts[k]:
                            sen_acc += [1]
                        else:
                            sen_acc += [0]

                        edit_dist += [get_ld(pred_order, gt_order)]
                        if PRINT_DEBUG:
                            print(f'pred/gt="{pred_text}"/"{gt_texts[k]}", ed={edit_dist[-1]:.4f}')
    print(f'Done, lines={len(sen_acc)}, sen_acc={np.array(sen_acc).mean():.4f}, edit_dist={np.array(edit_dist).mean():.4f}')


if __name__ == "__main__":
    main()
