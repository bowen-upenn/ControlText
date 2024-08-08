import os
import numpy as np
import cv2
import random
import math
import time
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from dataset_util import load, show_bbox_on_image
import yaml
import torch
import argparse
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


phrase_list = [
    ', content and position of the texts are ',
    ', textual material depicted in the image are ',
    ', texts that says ',
    ', captions shown in the snapshot are ',
    ', with the words of ',
    ', that reads ',
    ', the written materials on the picture: ',
    ', these texts are written on it: ',
    ', captions are ',
    ', content of the text in the graphic is '
]


def insert_spaces(string, nSpace):
    if nSpace == 0:
        return string
    new_string = ""
    for char in string:
        new_string += char + " " * nSpace
    return new_string[:-nSpace]


def get_caption_pos(ori_caption, pos_idxs, prob=1.0, place_holder='*'):
    idx2pos = {
        0: " top left",
        1: " top",
        2: " top right",
        3: " left",
        4: random.choice([" middle", " center"]),
        5: " right",
        6: " bottom left",
        7: " bottom",
        8: " bottom right"
    }
    new_caption = ori_caption + random.choice(phrase_list)
    pos = ''
    for i in range(len(pos_idxs)):
        if random.random() < prob and pos_idxs[i] > 0:
            pos += place_holder + random.choice([' located', ' placed', ' positioned', '']) + random.choice([' at', ' in', ' on']) + idx2pos[pos_idxs[i]] + ', '
        else:
            pos += place_holder + ' , '
    pos = pos[:-2] + '.'
    new_caption += pos
    return new_caption


def generate_random_rectangles(w, h, box_num):
    rectangles = []
    for i in range(box_num):
        x = random.randint(0, w)
        y = random.randint(0, h)
        w = random.randint(16, 256)
        h = random.randint(16, 96)
        angle = random.randint(-45, 45)
        p1 = (x, y)
        p2 = (x + w, y)
        p3 = (x + w, y + h)
        p4 = (x, y + h)
        center = ((x + x + w) / 2, (y + y + h) / 2)
        p1 = rotate_point(p1, center, angle)
        p2 = rotate_point(p2, center, angle)
        p3 = rotate_point(p3, center, angle)
        p4 = rotate_point(p4, center, angle)
        rectangles.append((p1, p2, p3, p4))
    return rectangles


def rotate_point(point, center, angle):
    # rotation
    angle = math.radians(angle)
    x = point[0] - center[0]
    y = point[1] - center[1]
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = x * math.sin(angle) + y * math.cos(angle)
    x1 += center[0]
    y1 += center[1]
    return int(x1), int(y1)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


class T3DataSet(Dataset):
    def __init__(
            self,
            json_path,
            max_lines=5,
            max_chars=20,
            place_holder='*',
            font_path='./font/Arial_Unicode.ttf',
            caption_pos_prob=1.0,
            mask_pos_prob=1.0,
            mask_img_prob=0.5,
            for_show=False,
            using_dlc=False,
            glyph_scale=1,
            percent=1.0,
            debug=False,
            wm_thresh=1.0,
            ):
        assert isinstance(json_path, (str, list))
        if isinstance(json_path, str):
            json_path = [json_path]
        data_list = []
        self.using_dlc = using_dlc
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.place_holder = place_holder
        self.font = ImageFont.truetype(font_path, size=60)
        self.caption_pos_porb = caption_pos_prob
        self.mask_pos_prob = mask_pos_prob
        self.mask_img_prob = mask_img_prob
        self.for_show = for_show
        self.glyph_scale = glyph_scale
        self.wm_thresh = wm_thresh
        for jp in json_path:
            data_list += self.load_data(jp, percent)
        self.data_list = data_list
        print(f'All dataset loaded, imgs={len(self.data_list)}')
        self.debug = debug
        if self.debug:
            self.tmp_items = [i for i in range(100)]

    def load_data(self, json_path, percent):
        tic = time.time()
        content = load(json_path)
        d = []
        count = 0
        wm_skip = 0
        max_img = len(content['data_list']) * percent
        for gt in content['data_list']:
            if len(d) > max_img:
                break
            if 'wm_score' in gt and gt['wm_score'] > self.wm_thresh:  # wm_score > thresh will be skipped as an img with watermark
                wm_skip += 1
                continue
            data_root = content['data_root']
            if self.using_dlc:
                data_root = data_root.replace('/data/vdb', '/mnt/data', 1)
            img_path = os.path.join(data_root, gt['img_name'])

            info = {}
            info['img_path'] = img_path
            info['caption'] = gt['caption'] if 'caption' in gt else ''
            if self.place_holder in info['caption']:
                count += 1
                info['caption'] = info['caption'].replace(self.place_holder, " ")
            if 'annotations' in gt:
                polygons = []
                invalid_polygons = []
                texts = []
                languages = []
                pos = []
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0:
                        continue
                    if 'valid' in annotation and annotation['valid'] is False:
                        invalid_polygons.append(annotation['polygon'])
                        continue
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    languages.append(annotation['language'])
                    if 'pos' in annotation:
                        pos.append(annotation['pos'])
                info['polygons'] = [np.array(i) for i in polygons]
                info['invalid_polygons'] = [np.array(i) for i in invalid_polygons]
                info['texts'] = texts
                info['language'] = languages
                info['pos'] = pos
            d.append(info)
        print(f'{json_path} loaded, imgs={len(d)}, wm_skip={wm_skip}, time={(time.time()-tic):.2f}s')
        if count > 0:
            print(f"Found {count} image's caption contain placeholder: {self.place_holder}, change to ' '...")
        return d

    def __getitem__(self, item):
        item_dict = {}
        if self.debug:  # sample fixed items
            item = self.tmp_items.pop()
            print(f'item = {item}')
        cur_item = self.data_list[item]
        # img
        target = np.array(Image.open(cur_item['img_path']).convert('RGB'))
        if target.shape[0] != 512 or target.shape[1] != 512:
            target = cv2.resize(target, (512, 512))
        target = (target.astype(np.float32) / 127.5) - 1.0
        item_dict['img'] = target
        # caption
        item_dict['caption'] = cur_item['caption']
        item_dict['glyphs'] = []
        item_dict['gly_line'] = []
        item_dict['positions'] = []
        item_dict['texts'] = []
        item_dict['language'] = []
        item_dict['inv_mask'] = []
        texts = cur_item.get('texts', [])
        if len(texts) > 0:
            idxs = [i for i in range(len(texts))]
            if len(texts) > self.max_lines:
                sel_idxs = random.sample(idxs, self.max_lines)
                unsel_idxs = [i for i in idxs if i not in sel_idxs]
            else:
                sel_idxs = idxs
                unsel_idxs = []
            if len(cur_item['pos']) > 0:
                pos_idxs = [cur_item['pos'][i] for i in sel_idxs]
            else:
                pos_idxs = [-1 for i in sel_idxs]

            item_dict['caption'] = get_caption_pos(item_dict['caption'], pos_idxs, self.caption_pos_porb, self.place_holder)
            item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
            # print('polygons', [len(polygon) for polygon in item_dict['polygons']])
            item_dict['areas'] = [cv2.contourArea(cur_item['polygons'][i]) for i in sel_idxs]

            # mask_pos, only draw the largest polygon
            # find the index of the largest area
            largest_area_idx = item_dict['areas'].index(max(item_dict['areas']))   # the index in sel_idxs
            largest_polygon = item_dict['polygons'][largest_area_idx]
            item_dict['positions'] += [self.draw_pos(largest_polygon, self.mask_pos_prob)]
            # for polygon in item_dict['polygons']:
            #     item_dict['positions'] += [self.draw_pos(polygon, self.mask_pos_prob)]

            # glyphs, only draw the glyphs in the largest polygon
            largest_polygon_mask = self.draw_inv_mask([largest_polygon])[:, :, 0].astype(bool)
            item_dict['largest_polygon_mask'] = largest_polygon_mask
            item_dict['largest_polygon'] = largest_polygon

            # only keep the ones for the largest polygon
            item_dict['texts'] = [cur_item['texts'][i][:self.max_chars] for i in sel_idxs if i == largest_area_idx]
            item_dict['language'] = [cur_item['language'][i] for i in sel_idxs if i == largest_area_idx]

        # inv_mask
        invalid_polygons = cur_item['invalid_polygons'] if 'invalid_polygons' in cur_item else []
        if len(texts) > 0:
            invalid_polygons += [cur_item['polygons'][i] for i in unsel_idxs]
        item_dict['inv_mask'] = self.draw_inv_mask(invalid_polygons)

        item_dict['hint'] = self.get_hint(item_dict['positions'])
        if random.random() < self.mask_img_prob:
            # randomly generate 0~3 masks
            box_num = random.randint(0, 3)
            boxes = generate_random_rectangles(512, 512, box_num)
            boxes = np.array(boxes)
            pos_list = item_dict['positions'].copy()
            for i in range(box_num):
                pos_list += [self.draw_pos(boxes[i], self.mask_pos_prob)]
            mask = self.get_hint(pos_list)
            masked_img = target*(1-mask)
        else:
            masked_img = np.zeros_like(target)
        item_dict['masked_img'] = masked_img

        if self.for_show:
            item_dict['img_name'] = os.path.split(cur_item['img_path'])[-1]
            return item_dict
        if len(texts) > 0:
            del item_dict['polygons']
        # padding
        n_lines = min(len(texts), self.max_lines)
        item_dict['n_lines'] = n_lines
        n_pad = self.max_lines - n_lines
        if n_pad > 0:
            item_dict['glyphs'] += [np.zeros((512*self.glyph_scale, 512*self.glyph_scale, 1))] * n_pad
            item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
            item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
            item_dict['texts'] += [' '] * n_pad
            item_dict['language'] += [' '] * n_pad

        return item_dict

    def __len__(self):
        return len(self.data_list)

    def draw_inv_mask(self, polygons):
        img = np.zeros((512, 512))
        for p in polygons:
            pts = p.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
        img = img[..., None]
        return img/255.

    def draw_pos(self, ploygon, prob=1.0):
        img = np.zeros((512, 512))
        rect = cv2.minAreaRect(ploygon)
        w, h = rect[1]
        small = False
        if w < 20 or h < 20:
            small = True
        if random.random() < prob:
            pts = ploygon.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
            # 10% dilate / 10% erode / 5% dilatex2  5% erodex2
            random_value = random.random()
            kernel = np.ones((3, 3), dtype=np.uint8)
            if random_value < 0.7:
                pass
            elif random_value < 0.8:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.9 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.95:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=2)
            elif random_value < 1.0 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=2)
        img = img[..., None]
        return img/255.

    def get_hint(self, positions):
        if len(positions) == 0:
            return np.zeros((512, 512, 1))
        return np.sum(positions, axis=0).clip(0, 1)


if __name__ == '__main__':
    '''
    Run this script to show details of your dataset, such as ocr annotations, glyphs, prompts, etc.
    '''
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import shutil

    if torch.distributed.is_initialized():
        current_rank = torch.distributed.get_rank()
    else:
        current_rank = 0  # Assuming it's the primary device or not using distributed training

    imgs_dir = 'condition_images'
    if os.path.exists(imgs_dir):
        shutil.rmtree(imgs_dir)
    os.makedirs(imgs_dir)

    json_paths = [
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/ArT/data.json',
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/Art/data.json',
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/Art/data.json',
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/COCO_Text/data.json',
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/icdar2017rctw/data.json',
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/LSVT/data.json',
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/mlt2019/data.json',
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/MTWI2018/data.json',
        # r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/ReCTS/data.json',
        '/tmp/datasets/AnyWord-3M/link_download/laion/test_data_v1.1.json',
        # '/tmp/datasets/AnyWord-3M/link_download/wukong_1of5/data_v1.1.json',
        # '/tmp/datasets/AnyWord-3M/link_download/wukong_2of5/data_v1.1.json',
        # '/tmp/datasets/AnyWord-3M/link_download/wukong_3of5/data_v1.1.json',
        # '/tmp/datasets/AnyWord-3M/link_download/wukong_4of5/data_v1.1.json',
        # '/tmp/datasets/AnyWord-3M/link_download/wukong_5of5/data_v1.1.json',
    ]

    dataset = T3DataSet(json_paths, for_show=True, max_lines=20, glyph_scale=2, mask_img_prob=1.0, caption_pos_prob=0.0)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    transform = None

    for i, data in tqdm(enumerate(train_loader)):
        # if data['img_name'][0][:-4] not in ['000149739']:
        #     continue
        # print("img_name:", data['img_name'][0])

        img = ((data['img'][0].numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        mask = data['largest_polygon_mask']
        largest_polygon = data['largest_polygon'][0].numpy()

        # Extract texts from the background image
        img[~mask[0], :] = 0.0
        # cv2.imwrite(f"test_imgs/{data['img_name'][0][:-4]}_img.jpg", img)

        rect = cv2.minAreaRect(largest_polygon)  # a bounding rectangle that also considers the rotation
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        # w, h = rect[1]
        # angle = rect[2]
        # print(f'angle={angle}, w={w}, h={h}, box={box}')

        # Draw and save the bounding box image
        img_with_box = img.copy()
        img_with_box = cv2.drawContours(img_with_box, [box], 0, (255, 0, 0), 2)
        cv2.imwrite(f"test_imgs/{data['img_name'][0][:-4]}_box.jpg", img_with_box)

        # # Draw and save the polygon image
        # img_with_polygon = img.copy()
        # img_with_polygon = cv2.polylines(img_with_polygon, [largest_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        # cv2.imwrite(f"test_imgs/{data['img_name'][0][:-4]}_poly.jpg", img_with_polygon)

        # Draw and save the position image
        positions = data['positions'][0].numpy()[0, :, :, 0]
        # print(positions.shape, positions.max(), positions.min(), f"test_imgs/{data['img_name'][0][:-4]}_pos.jpg")
        img_with_pos = (positions * 255).astype(np.uint8)
        cv2.imwrite(f"test_imgs/{data['img_name'][0][:-4]}_pos.jpg", img_with_pos)

        # Calculate the width and height of the box
        box = order_points(box.astype(np.float32))
        w = np.linalg.norm(box[0] - box[1])
        h = np.linalg.norm(box[1] - box[2])

        # Calculate the new rectangle dimensions
        aspect_ratio = w / h
        max_side_length = int(0.8 * 512)
        if w > h:
            new_width = max_side_length
            new_height = int(max_side_length / aspect_ratio)
        else:
            new_height = max_side_length
            new_width = int(max_side_length / aspect_ratio)
        # print('w', w, 'h', h, 'box', box, 'new_width:', new_width, 'new_height:', new_height, 'aspect_ratio', aspect_ratio, 'name', data['img_name'][0])

        # Center the new rectangle
        center_x, center_y = 256, 256  # Image center for 512x512 image
        new_rect = np.array([
            [center_x - new_width // 2, center_y - new_height // 2],
            [center_x + new_width // 2, center_y - new_height // 2],
            [center_x + new_width // 2, center_y + new_height // 2],
            [center_x - new_width // 2, center_y + new_height // 2]
        ], dtype=np.float32)

        # Calculate the perspective transform matrix
        old_rect_ordered = box
        new_rect_ordered = new_rect.astype(np.float32)
        M = cv2.getPerspectiveTransform(old_rect_ordered, new_rect_ordered)

        # Apply the perspective transformation
        transformed_img = cv2.warpPerspective(img, M, (512, 512))

        # Draw the new rect on the transformed image
        img_with_new_rect = transformed_img.copy()
        img_with_new_rect = cv2.drawContours(img_with_new_rect, [new_rect.astype(np.int32)], 0, (0, 255, 0), 2)

        # Save the projected image with the new rect
        cv2.imwrite(f"test_imgs/{data['img_name'][0][:-4]}_projected.jpg", img_with_new_rect)

    # if i > 10:
        #     break

