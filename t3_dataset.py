import os
import numpy as np
import cv2
import random
import math
import time
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from dataset_util import load, show_bbox_on_image
import time
import argparse
from shapely.geometry import Polygon
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import paddle
from paddleocr import PaddleOCR, draw_ocr
from difflib import SequenceMatcher
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from filelock import FileLock


# Initialize distributed process group for multiple GPUs
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def collate_fn(batch):
    # Filter out dictionaries that contain None in any of their values
    valid_batch = [item for item in batch if all(v is not None for v in item.values())]

    # Check if the filtered batch is empty
    if len(valid_batch) == 0:
        return []

    return torch.utils.data.dataloader.default_collate(valid_batch)
    
    
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


def draw_glyph(font, text):
    g_size = 50
    W, H = (512, 80)
    new_font = font.font_variant(size=g_size)
    img = Image.new(mode='1', size=(W, H), color=0)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = new_font.getbbox(text)
    text_width = max(right - left, 5)
    text_height = max(bottom - top, 5)
    ratio = min(W * 0.9 / text_width, H * 0.9 / text_height)
    new_font = font.font_variant(size=int(g_size * ratio))

    text_width, text_height = new_font.getsize(text)
    offset_x, offset_y = new_font.getoffset(text)
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2 - offset_y // 2
    draw.text((x, y), text, font=new_font, fill='white')
    img = np.expand_dims(np.array(img), axis=2).astype(np.float64)
    return img


def draw_glyph2(font, text, polygon, vertAng=10, scale=1, width=512, height=512, add_space=True):
    enlarge_polygon = polygon * scale
    rect = cv2.minAreaRect(enlarge_polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if (abs(angle) % 90 < vertAng or abs(90 - abs(angle) % 90) % 90 < vertAng):
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    img = np.zeros((height * scale, width * scale, 3), np.uint8)
    img = Image.fromarray(img)

    # infer font size
    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h):
        # add space
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_space = insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = insert_spaces(text, i - 1)
        font_size = min(w, h) * 0.80
    else:
        shrink = 0.75 if vert else 0.85
        font_size = min(w, h) / (text_w / max(w, h)) * shrink
    new_font = font.font_variant(size=int(font_size))

    left, top, right, bottom = new_font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if not vert:
        draw.text((rect[0][0] - text_width // 2, rect[0][1] - text_height // 2 - top), text, font=new_font, fill=(255, 255, 255, 255))
    else:
        x_s = min(box[:, 0]) + _w // 2 - text_height // 2
        y_s = min(box[:, 1])
        for c in text:
            draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))

    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
    img = np.expand_dims(np.array(img.convert('1')), axis=2).astype(np.float64)
    return img


def load_all_glyphs(glyph_path):
    # load the jpg image
    # print('img_path', img_path)
    img = cv2.imread(glyph_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
        img[img < 0.5] = 0
        img[img >= 0.5] = 1
    return img


def find_glyph(glyph_img, polygon, scale=1):
    # start_time = time.time()
    if scale != 1:
        new_size = (int(glyph_img.shape[1] / scale), int(glyph_img.shape[0] / scale))  # (width, height)
        glyph_img = cv2.resize(glyph_img, new_size, interpolation=cv2.INTER_LINEAR)

    W, H = (512, 80)

    rect = cv2.minAreaRect(polygon)  # a bounding rectangle that also considers the rotation
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Calculate the width and height of the box
    box = order_points(box.astype(np.float32))
    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])

    # Calculate the new rectangle dimensions
    if h != 0:
        aspect_ratio = w / h
    else:
        aspect_ratio = 0

    if aspect_ratio == 0:
        # If the box is degenerate (e.g., a triangle or line), use boundingRect as a fallback
        x, y, w, h = cv2.boundingRect(polygon)
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        aspect_ratio = w / (h + 1e-5)

    new_height = int(0.8 * H)
    new_width = int(new_height * aspect_ratio)

    # aspect_ratio = w / h
    # max_side_length = int(0.8 * 512)
    # if w > h:
    #     new_width = max_side_length
    #     new_height = int(max_side_length / aspect_ratio)
    # else:
    #     new_height = max_side_length
    #     new_width = int(max_side_length / aspect_ratio)
    # print('w', w, 'h', h, 'box', box, 'new_width:', new_width, 'new_height:', new_height, 'aspect_ratio', aspect_ratio, 'name', data['img_name'][0])

    # Center the new rectangle
    center_x, center_y = W // 2, H // 2  # Image center for 512x80 image
    new_rect = np.array([
        [center_x - new_width // 2, center_y - new_height // 2],
        [center_x + new_width // 2, center_y - new_height // 2],
        [center_x + new_width // 2, center_y + new_height // 2],
        [center_x - new_width // 2, center_y + new_height // 2]
    ], dtype=np.float32)

    # end_time = time.time()  # Record the end time
    # execution_time = end_time - start_time  # Calculate the time difference
    # print(f"Execution time 1: {execution_time} seconds")
    # start_time = time.time()

    # Calculate the perspective transform matrix
    old_rect_ordered = box
    new_rect_ordered = new_rect.astype(np.float32)
    M = cv2.getPerspectiveTransform(old_rect_ordered, new_rect_ordered)

    # Apply the perspective transformation
    transformed_img = cv2.warpPerspective(glyph_img, M, (W, H))
    transformed_img = np.expand_dims(transformed_img, axis=2).astype(np.float64)

    # end_time = time.time()  # Record the end time
    # execution_time = end_time - start_time  # Calculate the time difference
    # print(f"Execution time 2: {execution_time} seconds")
    return transformed_img, aspect_ratio


def find_glyph2(img, position, scale=1, add_perturbation=True, max_offset=16):
    # start_time = time.time()
    # Convert the image to a numpy array
    img = np.array(img)
    # print(np.min(img), np.max(img))
    # print(np.min(position), np.max(position))
    # img[img < 150] = 0.0
    rows, cols = img.shape

    position = position.squeeze(-1)
    raw_position = position.copy()

    if scale != 1:
        # Scale the image using interpolation
        new_size = (int(position.shape[1] * scale), int(position.shape[0] * scale))  # (width, height)
        # if img.shape != new_size:
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
        position = cv2.resize(position, new_size, interpolation=cv2.INTER_NEAREST)

    # Apply the mask to the image, making pixels outside the polygon black
    img = img * position

    # Undo the scaling of the position mask for downstream processing
    position = raw_position

    # end_time = time.time()  # Record the end time
    # execution_time = end_time - start_time  # Calculate the time difference
    # print(f"Execution time 3: {execution_time} seconds")
    # start_time = time.time()

    if add_perturbation:
        # Add slight random perspective transformations to the image and the position mask
        pts1 = np.float32([[0, 0], [cols * scale, 0], [0, rows * scale], [cols * scale, rows * scale]])
        pts2 = pts1 + np.float32([
            [random.uniform(-max_offset * scale, 0), random.uniform(-max_offset * scale, 0)],  # Top-left
            [random.uniform(0, max_offset * scale), random.uniform(-max_offset * scale, 0)],  # Top-right
            [random.uniform(-max_offset * scale, 0), random.uniform(0, max_offset * scale)],  # Bottom-left
            [random.uniform(0, max_offset * scale), random.uniform(0, max_offset * scale)]  # Bottom-right
        ])
        M_perspective_img = cv2.getPerspectiveTransform(pts1, pts2)

        pts3 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        pts4 = pts3 + np.float32([
            [random.uniform(-max_offset, 0), random.uniform(-max_offset, 0)],  # Top-left
            [random.uniform(0, max_offset), random.uniform(-max_offset, 0)],  # Top-right
            [random.uniform(-max_offset, 0), random.uniform(0, max_offset)],  # Bottom-left
            [random.uniform(0, max_offset), random.uniform(0, max_offset)]  # Bottom-right
        ])
        M_perspective_pos = cv2.getPerspectiveTransform(pts3, pts4)

        # Apply the perspective transformation to both the image and the position mask
        img_perturbed = cv2.warpPerspective(img, M_perspective_img, (cols * scale, rows * scale))
        position_perturbed = cv2.warpPerspective(position, M_perspective_pos, (cols, rows))

        img_perturbed = np.expand_dims(img_perturbed, axis=2).astype(np.float64)
        position_perturbed = np.expand_dims(position_perturbed, axis=-1)

        # end_time = time.time()  # Record the end time
        # execution_time = end_time - start_time  # Calculate the time difference
        # print(f"Execution time 4: {execution_time} seconds")

        return img_perturbed, img, position_perturbed
    else:
        img = np.expand_dims(img, axis=2).astype(np.float64)
        return img, None, None


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


def calculate_iou(polygon1, polygon2):
    """Calculate Intersection over Union (IoU) between two polygons."""
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union


def find_nearest_polygon(detected_box, polygons):
    """Find the polygon from the list of polygons that has the highest IoU with the detected box."""
    best_iou = 0
    best_polygon = None
    best_idx = -1
    for idx, polygon in enumerate(polygons):
        iou = calculate_iou(detected_box, polygon)
        if iou > best_iou:
            best_iou = iou
            best_polygon = polygon
            best_idx = idx
    return best_polygon, best_idx


# Function to append invalid gly_lines for an image to the JSON file
def append_invalid_gly_lines_to_file(invalid_json_path, glyphs_path, invalid_gly_lines_curr_image):
    # Write back the updated content to the JSON file
    lock_path = invalid_json_path + ".lock"
    with FileLock(lock_path):  # Lock the file while reading/writing
        # Read existing content
        with open(invalid_json_path, 'r') as f:
            existing_data = json.load(f)

        # Add new entry for this glyph image
        existing_data[glyphs_path] = invalid_gly_lines_curr_image

        with open(invalid_json_path, 'w') as f:
            json.dump(existing_data, f, indent=4)


class T3DataSet(Dataset):
    def __init__(
            self,
            json_path,
            glyph_path,
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
            step='training',
            invalid_json_path='./Rethinking-Text-Segmentation/log/images/ocr_verified/invalid_gly_lines.json',
    ):
        assert isinstance(json_path, (str, list))
        if isinstance(json_path, str):
            json_path = [json_path]
        assert isinstance(glyph_path, (str, list))
        if isinstance(glyph_path, str):
            glyph_path = [glyph_path]
        assert len(json_path) == len(glyph_path)

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

        self.step = step
        self.invalid_json_path = invalid_json_path
        if self.step == 'training':
            with open(self.invalid_json_path, 'r') as f:
                self.invalid_glyph_lines = json.load(f)

        for jp, gp in zip(json_path, glyph_path):
            data_list += self.load_data(jp, gp, percent)
        self.data_list = data_list
        print(f'All dataset loaded, imgs={len(self.data_list)}')

        self.debug = debug
        if self.debug:
            self.tmp_items = [i for i in range(100)]

        self.num_missing_glyphs = 0
        self.num_invalid_glyph_lines = 0
        self.num_total_glyph_lines = 0

        self.ocr_ch = PaddleOCR(use_angle_cls=True, show_log=False, lang="ch")  # need to run only once to download and load model into memory
        self.ocr_en = PaddleOCR(use_angle_cls=True, show_log=False, lang="en")


    def load_data(self, json_path, glyph_path, percent):
        tic = time.time()
        print('json_path', json_path)
        content = load(json_path)
        d = []
        count = 0
        wm_skip = 0
        max_img = len(content['data_list']) * percent
        for gt in content['data_list']:
            if len(d) > max_img:
                break
            if 'wm_score' in gt and gt['wm_score'] > self.wm_thresh:  # wm_score > thresh will be skiped as an img with watermark
                wm_skip += 1
                continue
            data_root = content['data_root']
            if self.using_dlc:
                data_root = data_root.replace('/data/vdb', '/mnt/data', 1)
            data_root = data_root.replace('/pool/bwjiang/', '/tmp/')

            img_path = os.path.join(data_root, gt['img_name'])

            info = {}
            info['img_path'] = img_path
            info['img_name'] = gt['img_name']

            # # Replace the double occurrence with a single '.jpg'
            # if '.jpg.jpg' in gt['img_name']:
            #     gt['img_name'] = gt['img_name'].replace('.jpg.jpg', '.jpg')
            glyphs_path = os.path.join(glyph_path, gt['img_name'])
            info['glyphs_path'] = glyphs_path
            if self.step == 'training':
                if not os.path.exists(glyphs_path):
                    continue

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
                    # print('glyphs_path', glyphs_path, 'annotation', annotation)

                    # filter out low-quality texts and their polygons
                    if self.step == 'training' and glyphs_path in self.invalid_glyph_lines:
                        if annotation['polygon'] in self.invalid_glyph_lines[glyphs_path]:
                            invalid_polygons.append(annotation['polygon'])
                            continue

                    if 'valid' in annotation and annotation['valid'] is False:
                        invalid_polygons.append(annotation['polygon'])
                        continue

                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    languages.append(annotation['language'])
                    if 'pos' in annotation:
                        pos.append(annotation['pos'])

                if len(polygons) == 0:
                    continue

                info['polygons'] = [np.array(i) for i in polygons]
                info['invalid_polygons'] = [np.array(i) for i in invalid_polygons]
                if len(texts) == 0:
                    continue
                info['texts'] = texts
                info['language'] = languages
                info['pos'] = pos

            d.append(info)

        print(f'{json_path} loaded, imgs={len(d)}, wm_skip={wm_skip}, time={(time.time() - tic):.2f}s')
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
        item_dict['all_glyphs_from_segmentation'] = None
        item_dict['polygons'] = None
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

            item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
            item_dict['caption'] = get_caption_pos(item_dict['caption'], pos_idxs, self.caption_pos_porb, self.place_holder)
            item_dict['texts'] = [cur_item['texts'][i][:self.max_chars] for i in sel_idxs]
            item_dict['language'] = [cur_item['language'][i] for i in sel_idxs]

            # mask_pos
            for polygon in item_dict['polygons']:
                item_dict['positions'] += [self.draw_pos(polygon, self.mask_pos_prob)]

            # glyphs
            all_glyphs_from_segmentation = load_all_glyphs(cur_item['glyphs_path'])
            item_dict['all_glyphs_from_segmentation'] = all_glyphs_from_segmentation
            item_dict['glyphs_path'] = cur_item['glyphs_path']

            if all_glyphs_from_segmentation is not None:
                for idx, text in enumerate(item_dict['texts']):
                    glyphs, glyphs_raw, position = find_glyph2(all_glyphs_from_segmentation, item_dict['positions'][idx], scale=self.glyph_scale)
                    if glyphs_raw is not None:
                        gly_line, aspect_ratio = find_glyph(glyphs_raw, item_dict['polygons'][idx], scale=self.glyph_scale)
                        item_dict['positions'][idx] = position
                    else:
                        gly_line, aspect_ratio = find_glyph(glyphs, item_dict['polygons'][idx], scale=self.glyph_scale)

                    item_dict['glyphs'] += [glyphs]
                    item_dict['gly_line'] += [gly_line]

                    if aspect_ratio == 0:
                        print('Zero box height')
                        # self.num_invalid_glyph_line += 1
                    # self.num_total_glyph_lines += 1
            else:
                # just in case if a sample has no pre-processed glyph image
                # self.num_missing_glyphs += 1
                print('Missing glyph images')
                for idx, text in enumerate(item_dict['texts']):
                    gly_line = draw_glyph(self.font, text)
                    glyphs = draw_glyph2(self.font, text, item_dict['polygons'][idx], scale=self.glyph_scale)
                    item_dict['glyphs'] += [glyphs]
                    item_dict['gly_line'] += [gly_line]
            # # glyphs
            # for idx, text in enumerate(item_dict['texts']):
            #     gly_line = draw_glyph(self.font, text)
            #     glyphs = draw_glyph2(self.font, text, item_dict['polygons'][idx], scale=self.glyph_scale)
            #     item_dict['glyphs'] += [glyphs]
            #     item_dict['gly_line'] += [gly_line]

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
            masked_img = target * (1 - mask)
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
            item_dict['glyphs'] += [np.zeros((512 * self.glyph_scale, 512 * self.glyph_scale, 1))] * n_pad
            item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
            item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
            item_dict['texts'] += [' '] * n_pad
            item_dict['language'] += [' '] * n_pad

        return item_dict


    def __len__(self):
        return len(self.data_list)


    def load_glyline_and_ocr(self, gly_line, text, language, glyph_path, verbose=False):
        # load the jpg image
        if verbose:
            print('target text', text, 'gly_line', gly_line.shape, 'glyph_path', glyph_path)

        # Permute to get it to (C, H, W) format and repeat to make it RGB
        gly_line = (gly_line - torch.min(gly_line)) / (torch.max(gly_line) - torch.min(gly_line) + 1e-6) * 255
        gly_line = gly_line.permute(2, 0, 1).repeat(3, 1, 1)  # Now shape is (3, 80, 512)
        gly_line = gly_line.numpy().astype(np.uint8)  # Convert to NumPy and ensure it's uint8
        gly_line = np.transpose(gly_line, (1, 2, 0))  # Transpose to (H, W, C) format for PIL

        # Convert to PIL image required by OCR
        gly_line = Image.fromarray(gly_line).convert("RGB")

        if language == 'Latin':
            ocr_result = self.ocr_en.ocr(np.asarray(gly_line), cls=True)
        else:  # chinese
            ocr_result = self.ocr_ch.ocr(np.asarray(gly_line), cls=True)

        if ocr_result is not None and ocr_result[0] is not None:
            ocr_result = ocr_result[0][0]
            detected_box = ocr_result[0]  # Bounding box of the detected text
            detected_text = ocr_result[1][0]  # Detected text itself
            confidence = ocr_result[1][1]  # Confidence score of the detected text
            if verbose:
                print('detected_text', detected_text, 'confidence', confidence, 'detected_box', detected_box)

            # Mark high quality detections
            detected_text = detected_text.lower().replace(' ', '')
            text = text.lower().replace(' ', '')
            if detected_text == text:   # Shortcut
                if verbose:
                    print("Matched texts.\n")
                return True

            if confidence > 0.8:
                # Check if the edit distance between the texts is less than 2
                edit_distance = SequenceMatcher(None, detected_text, text).ratio()
                if edit_distance > 0.8:  # If texts are within 1 edit distance
                    if verbose:
                        print(f"High quality detection: Close match (edit distance ratio: {edit_distance}).\n")
                    return True
                else:
                    if verbose:
                        print(f"Low quality detection: Edit distance ratio: {edit_distance}.\n")
            else:
                if verbose:
                    print('Low quality detection\n')
        else:
            if verbose:
                print('No text detected\n')
        return False # Return None if no high quality detection is found


    def draw_inv_mask(self, polygons):
        img = np.zeros((512, 512))
        for p in polygons:
            pts = p.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
        img = img[..., None]
        return img / 255.


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
        return img / 255.


    def get_hint(self, positions):
        if len(positions) == 0:
            return np.zeros((512, 512, 1))
        return np.sum(positions, axis=0).clip(0, 1)


def run_inference(rank, world_size, json_paths, glyph_paths, glyph_scale, show_count, dataset_percent, step, invalid_json_path):
    # Setup distributed environment
    setup(rank, world_size)

    dataset = T3DataSet(json_paths, glyph_paths, for_show=True, max_lines=20, glyph_scale=glyph_scale, percent=dataset_percent, mask_img_prob=1.0, caption_pos_prob=0.0, step=step, invalid_json_path=invalid_json_path)
    if world_size == 1:
        train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    else:
        # Use DistributedSampler for multi-GPU training
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, sampler=sampler, collate_fn=collate_fn)

    pbar = tqdm(total=show_count if show_count != -1 else len(train_loader))
    for i, data in enumerate(train_loader):
        if not data:    # Skip empty batches
            continue

        if show_count != -1 and i == show_count:
            break

        if step == 'show_results':
            img = ((data['img'][0].numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
            masked_img = ((data['masked_img'][0].numpy() + 1.0) / 2.0 * 255)[..., ::-1].astype(np.uint8)
            img_name = data['img_name'][0][:-4]
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{img_name}_masked.jpg'), masked_img)
            if 'texts' in data and len(data['texts']) > 0:
                texts = [x[0] for x in data['texts']]
                img = show_bbox_on_image(Image.fromarray(img), data['polygons'], texts)
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{img_name}.jpg'), np.array(img)[..., ::-1])
            with open(os.path.join(show_imgs_dir, f'plots_{img_name}.txt'), 'w') as fin:
                fin.writelines([data['caption'][0]])
            all_glyphs = []
            for k, glyphs in enumerate(data['glyphs']):
                cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{img_name}_glyph_{k}.jpg'), glyphs[0].numpy().astype(np.int32) * 255)
                all_glyphs += [glyphs[0].numpy().astype(np.int32) * 255]
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{img_name}_allglyphs.jpg'), np.sum(all_glyphs, axis=0))
            for k, gly_line in enumerate(data['gly_line']):
                cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{img_name}_gly_line_{k}.jpg'), gly_line[0].numpy().astype(np.int32) * 255)
            for k, position in enumerate(data['positions']):
                if position is not None:
                    cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{img_name}_pos_{k}.jpg'), position[0].numpy().astype(np.int32) * 255)
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{img_name}_hint.jpg'), data['hint'][0].numpy().astype(np.int32) * 255)
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{img_name}_inv_mask.jpg'), np.array(img)[..., ::-1] * (1 - data['inv_mask'][0].numpy().astype(np.int32)))

        else:
            has_good_quality = False
            filtered_glyphs = data['all_glyphs_from_segmentation'][0]
            if filtered_glyphs is None:
                invalid_gly_lines_curr_image = [{'polygon': [], 'text': ''}]
            else:
                filtered_glyphs = data['all_glyphs_from_segmentation'][0].unsqueeze(-1)
                invalid_gly_lines_curr_image = []

                for k in range(len(data['gly_line'])):
                    good_quality = dataset.load_glyline_and_ocr(data['gly_line'][k][0], data['texts'][k][0], data['language'][k], data['glyphs_path'][0])
                    if good_quality:
                        has_good_quality = True
                    else:
                        poor_positions = data['positions'][k][0]
                        poor_positions = 1 - poor_positions

                        # remove contents in the poor positions
                        filtered_glyphs[poor_positions < 0.5] = 0

                        # If the gly_line is of poor quality, store relevant information
                        invalid_entry = {
                            'polygon': data['polygons'][k].tolist(),  # Store the polygon for this invalid gly_line
                            'text': data['texts'][k][0],  # Store the text for this invalid gly_line
                        }
                        invalid_gly_lines_curr_image.append(invalid_entry)  # Add to the list of invalid gly_lines

                if has_good_quality:
                    # save filtered glyphs
                    # if glyph_scale != 1:
                    #     new_size = (int(filtered_glyphs.shape[1] * glyph_scale), int(filtered_glyphs.shape[0] * glyph_scale))  # (width, height)
                    #     filtered_glyphs = cv2.resize(filtered_glyphs.numpy(), new_size, interpolation=cv2.INTER_AREA) * 255

                    saved_file_name = data['glyphs_path'][0].replace('/output/', '/ocr_verified/')
                    # print('saved_file_name', saved_file_name)
                    cv2.imwrite(saved_file_name, filtered_glyphs.numpy() * 255)

            if len(invalid_gly_lines_curr_image) > 0:
                # Append the invalid gly_lines for this image to the JSON file
                append_invalid_gly_lines_to_file(invalid_json_path, data['glyphs_path'][0], invalid_gly_lines_curr_image)

        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    '''
    Run this script to show details of your dataset, such as ocr annotations, glyphs, prompts, etc.
    '''
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import shutil

    paddle.set_device('cpu')
    print('paddle.is_compiled_with_cuda()', paddle.is_compiled_with_cuda())
    print('paddle.device.get_device()', paddle.device.get_device())

    world_size = torch.cuda.device_count()
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--step', type=str, default="show_results", help='show_results or process_segmentations')
    cmd_args = parser.parse_args()
    step = cmd_args.step
    if step == 'show_results':
        assert world_size == 1

    show_imgs_dir = 'show_results'
    invalid_json_path = './Rethinking-Text-Segmentation/log/images/ocr_verified/invalid_gly_lines.json'

    show_count = -1
    glyph_scale = 2
    dataset_percent = 0.0566   # 1.0 use full datasets, 0.0566 use ~200k images for ablation study
    if os.path.exists(show_imgs_dir):
        shutil.rmtree(show_imgs_dir)
    os.makedirs(show_imgs_dir)
    plt.rcParams['axes.unicode_minus'] = False

    if step == 'show_results':
        json_paths = [
            r'/tmp/datasets/AnyWord-3M/link_download/laion/test_data_v1.1.json',
        ]
        glyph_paths = [
            r'./Rethinking-Text-Segmentation/log/images/output/laion_test',
        ]
    else:
        json_paths = [
            # r'/tmp/datasets/AnyWord-3M/link_download/laion/test_data_v1.1.json',
            r'/tmp/datasets/AnyWord-3M/link_download/laion/data_v1.1.json',
            r'/tmp/datasets/AnyWord-3M/link_download/wukong_1of5/data_v1.1.json',
            r'/tmp/datasets/AnyWord-3M/link_download/wukong_2of5/data_v1.1.json',
            r'/tmp/datasets/AnyWord-3M/link_download/wukong_3of5/data_v1.1.json',
            r'/tmp/datasets/AnyWord-3M/link_download/wukong_4of5/data_v1.1.json',
            r'/tmp/datasets/AnyWord-3M/link_download/wukong_5of5/data_v1.1.json',
            r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/Art/data.json',
            r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/COCO_Text/data.json',
            r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/icdar2017rctw/data.json',
            r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/LSVT/data.json',
            r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/mlt2019/data.json',
            r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/MTWI2018/data.json',
            r'/tmp/datasets/AnyWord-3M/link_download/ocr_data/ReCTS/data.json'
        ]
        glyph_paths = [
            # r'./Rethinking-Text-Segmentation/log/images/output/laion_test',
            r'./Rethinking-Text-Segmentation/log/images/output/laion',
            r'./Rethinking-Text-Segmentation/log/images/output/wukong_1of5',
            r'./Rethinking-Text-Segmentation/log/images/output/wukong_2of5',
            r'./Rethinking-Text-Segmentation/log/images/output/wukong_3of5',
            r'./Rethinking-Text-Segmentation/log/images/output/wukong_4of5',
            r'./Rethinking-Text-Segmentation/log/images/output/wukong_5of5',
            r'./Rethinking-Text-Segmentation/log/images/output/Art',
            r'./Rethinking-Text-Segmentation/log/images/output/COCO_Text',
            r'./Rethinking-Text-Segmentation/log/images/output/icdar2017rctw',
            r'./Rethinking-Text-Segmentation/log/images/output/LSVT',
            r'./Rethinking-Text-Segmentation/log/images/output/mlt2019',
            r'./Rethinking-Text-Segmentation/log/images/output/MTWI2018',
            r'./Rethinking-Text-Segmentation/log/images/output/ReCTS'
        ]

        # Check if the file exists
        # if not os.path.exists(invalid_json_path):
        # If the file does not exist, create it with an empty dictionary
        with open(invalid_json_path, 'w') as f:
            json.dump({}, f)

    mp.spawn(run_inference, nprocs=world_size, args=(world_size, json_paths, glyph_paths, glyph_scale, show_count, dataset_percent, step, invalid_json_path))
    cleanup()
