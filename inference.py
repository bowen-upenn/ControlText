import os
import torch
from torch.utils.data import DataLoader
from t3_dataset import T3DataSet
from cldm.model import create_model, load_state_dict
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import langid
import argparse
from PIL import Image, ImageFont

import util
from t3_dataset import draw_glyph, draw_glyph2


# Configurations for inference
batch_size = 1  # Can adjust based on available VRAM
resume_path = './models/lightning_logs/version_2/checkpoints/last.ckpt'  # './models/anytext_v1.1.ckpt'
# resume_path = './models-oct-12/lightning_logs/version_0/checkpoints/last.ckpt'  # Path to the trained model checkpoint
model_config = './models_yaml/anytext_sd15.yaml'  # Model configuration
mask_ratio = 1  # Inference setting, set 0 to disable masking
wm_thresh = 0.5  # Watermark threshold (adjust based on the inference dataset)
dataset_percent = 1.0  # Use the full dataset for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
single_custom_image = True # Set to True to process a single custom image


def prepare_custom_inputs():
    # This is a test case
    item_dict = {}
    item_dict["img_path"] = "./show_results/plots_000001710.jpg"
    item_dict["caption"] = "human country jukebox logo on a clean background"
    item_dict["texts"] = ['JUK', 'EBOX', 'Country', 'HUMAN']
    font_paths = ["./fonts/BigCaslon.ttf" for _ in range(len(item_dict["texts"]))]
    fonts = [ImageFont.truetype(font_paths[i], size=60) for i in range(len(item_dict["texts"]))]
    item_dict["language"] = []
    for text in item_dict["texts"]:
        lang, _ = langid.classify(text)
        item_dict["language"].append(lang)

    item_dict["polygons"] = [np.array([[[102, 397],
                                        [235, 366],
                                        [240, 393],
                                        [107, 424]]]),
                             np.array([[[231, 392],
                                        [235, 363],
                                        [405, 392],
                                        [401, 421]]]),
                             np.array([[[ 61, 352],
                                        [ 73, 219],
                                        [446, 259],
                                        [434, 393]]]),
                             np.array([[[118, 185],
                                        [396, 175],
                                        [398, 250],
                                        [120, 261]]])]

    all_glyphs = np.zeros((1024, 1024))
    item_dict["glyphs"] = []
    item_dict["gly_line"] = []
    glyph_scale = 2

    for idx, (text, font) in enumerate(zip(item_dict["texts"], fonts)):
        gly_line = draw_glyph(font, text)
        item_dict["gly_line"] += [gly_line]
        glyphs = draw_glyph2(font, text, item_dict["polygons"][idx], scale=glyph_scale)
        item_dict["glyphs"] += [glyphs]
        all_glyphs += glyphs.squeeze(-1)

    all_glyphs[all_glyphs > 0] = 1
    all_glyphs = all_glyphs.astype(np.bool).astype(np.uint8) * 255

    save_path = item_dict['img_path'].replace(".jpg", "_allglyphs.jpg")
    save_path = os.path.join('./inference_output', save_path)
    cv2.imwrite(save_path, all_glyphs)

    return item_dict


def inference(model, dataloader):
    # Inference loop
    print('Starting inference...')

    # Define some constants for sampling
    ddim_steps = 200  # Number of steps in DDIM sampling
    ddim_eta = 0.0  # Sampling noise eta

    for batch in tqdm(dataloader):
        # Prepare inputs using get_input (z, cond, etc.)
        z, cond = model.get_input(batch, model.first_stage_key, bs=batch_size)

        # Process conditioning if required
        if model.cond_stage_trainable:
            with torch.no_grad():
                cond = model.get_learned_conditioning(cond)

        # Prepare the conditioning inputs
        c_crossattn = cond["c_crossattn"][0][:batch_size]  # Cross-attention conditioning
        c_cat = cond["c_concat"][0][:batch_size]  # Concatenated conditioning
        text_info = cond["text_info"]
        text_info['glyphs'] = [i[:batch_size] for i in text_info['glyphs']]
        text_info['gly_line'] = [i[:batch_size] for i in text_info['gly_line']]
        text_info['positions'] = [i[:batch_size] for i in text_info['positions']]
        text_info['n_lines'] = text_info['n_lines'][:batch_size]
        text_info['masked_x'] = text_info['masked_x'][:batch_size]
        text_info['img'] = text_info['img'][:batch_size]
        text_info['img_path'] = text_info['img_path'][:batch_size]

        # Perform inference
        with torch.no_grad():
            samples, _ = model.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c_crossattn], "text_info": text_info},
                batch_size=batch_size,
                ddim=True,  # Use DDIM
                ddim_steps=ddim_steps,
                eta=ddim_eta
            )

        # Decode the latent samples into images
        decoded_samples = model.decode_first_stage(samples)

        # Save the decoded samples as images
        for i in range(batch_size):
            full_path = Path(text_info['img_path'][i])
            extracted_name = full_path.parent.name + '/' + full_path.name
            save_path = os.path.join('./inference_output', extracted_name)
            save_path_no_blend = os.path.join('./inference_output_no_blend', extracted_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            os.makedirs(os.path.dirname(save_path_no_blend), exist_ok=True)

            # Normalize and prepare the decoded sample
            decoded_sample = decoded_samples[i].float().cpu().numpy().transpose(1, 2, 0)
            decoded_sample = (decoded_sample - decoded_sample.min()) / (decoded_sample.max() - decoded_sample.min())
            decoded_sample = (decoded_sample * 255).astype(np.uint8)
            image = text_info['img'][i].float().cpu().numpy()
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)

            # Blend each of the masked region
            for j in range(len(text_info['positions'])):
                region = text_info['positions'][j][i].cpu().numpy().squeeze(0)
                region = (region * 255).astype(np.uint8)
                if torch.max(text_info['glyphs'][j][i]) == 0:
                    continue

                # Find the center of the mask for seamless cloning
                y_indices, x_indices = np.where(region > 0)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    # Skip if no valid region is found
                    continue

                center_x = (x_indices.min() + x_indices.max()) // 2
                center_y = (y_indices.min() + y_indices.max()) // 2
                center_x = util.clamp(center_x, 0, image.shape[0] - 1)
                center_y = util.clamp(center_y, 0, image.shape[1] - 1)
                center = (center_x, center_y)

                # Perform Poisson blending
                blended_image = cv2.seamlessClone(decoded_sample, image, region, center, cv2.NORMAL_CLONE)
                image = blended_image

            # Save the blended result
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path_no_blend, cv2.cvtColor(decoded_sample, cv2.COLOR_RGB2BGR))

        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--option', type=str, default="single", help='single or dataset')
    cmd_args = parser.parse_args()
    single_custom_image = True if cmd_args.option == "single" else False

    # Load the pre-trained model
    model = create_model(model_config)  # Load model configuration
    model.load_state_dict(load_state_dict(resume_path))  # Load trained weights
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    if single_custom_image:
        world_size = torch.cuda.device_count()
        assert world_size == 1

        item_dict = prepare_custom_inputs()
        batch_size = 1
        dataset = T3DataSet(
            item_dict["img_path"], max_lines=5, max_chars=20, caption_pos_prob=0.0,
            mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=2,
            percent=dataset_percent, debug=False, using_dlc=False, wm_thresh=wm_thresh,
            single_custom_image=True, custom_inputs=item_dict
        )
    else:
        # Define paths to data
        json_paths = [
            r'/tmp/datasets/AnyWord-3M/link_download/laion/test_data_v1.1.json',
        ]
        glyph_paths = [
            r'./Rethinking-Text-Segmentation/log/images/ocr_verified/laion_test',
        ]

        # Load the dataset for inference
        dataset = T3DataSet(
            json_paths, glyph_paths, max_lines=5, max_chars=20, caption_pos_prob=0.0,
            mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=2,
            percent=dataset_percent, debug=False, using_dlc=False, wm_thresh=wm_thresh
        )

    # Create a DataLoader for inference
    dataloader = DataLoader(dataset, num_workers=8, persistent_workers=True, batch_size=batch_size, shuffle=False)

    # Run inference
    inference(model, dataloader)
