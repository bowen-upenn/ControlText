import os
import torch
from torch.utils.data import DataLoader
from t3_dataset import T3DataSet
from cldm.model import create_model, load_state_dict
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm


# Configurations for inference
batch_size = 6  # Can adjust based on available VRAM
resume_path = './models-oct-12/lightning_logs/version_0/checkpoints/last.ckpt'  # Path to the trained model checkpoint
model_config = './models_yaml/anytext_sd15.yaml'  # Model configuration
mask_ratio = 1  # Inference setting, set 0 to disable masking
wm_thresh = 0.5  # Watermark threshold (adjust based on the inference dataset)
dataset_percent = 1.0  # Use the full dataset for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load the pre-trained model
    model = create_model(model_config)  # Load model configuration
    model.load_state_dict(load_state_dict(resume_path))  # Load trained weights
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

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
        percent=dataset_percent, debug=False, using_dlc=False, wm_thresh=wm_thresh,
        invalid_json_path=None  # Not needed for inference
    )

    # Create a DataLoader for inference
    dataloader = DataLoader(dataset, num_workers=8, persistent_workers=True, batch_size=batch_size, shuffle=False)

    # Inference loop
    print('Starting inference...')

    # Define some constants for sampling
    ddim_steps = 20  # Number of steps in DDIM sampling
    ddim_eta = 0.0  # Sampling noise eta
    unconditional_guidance_scale = 9.0  # Guidance scale

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
        for i in range(decoded_samples.shape[0]):
            full_path = Path(text_info['img_path'][i])
            extracted_name = full_path.parent.name + '/' + full_path.name
            save_path = os.path.join('./inference_output', extracted_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Only replace the regions within the positions, and keep the masked_x regions
            print("decoded_samples[i]", decoded_samples[i].shape, "text_info['positions'][i]", text_info['positions'][i].shape,
                  "text_info['img'][i]", text_info['img'][i].shape, "masked_x[i]", text_info['masked_x'][i].shape)
            decoded_samples[i] = decoded_samples[i] * text_info['positions'][i] + text_info['img'][i] * (1 - text_info['positions'][i])

            decoded_sample = decoded_samples[i]
            decoded_sample = (decoded_sample - torch.min(decoded_sample)) / (torch.max(decoded_sample) - torch.min(decoded_sample))
            decoded_sample = decoded_sample.cpu().numpy().transpose(1, 2, 0) * 255
            cv2.imwrite(save_path, decoded_sample)

            break


# from modelscope.pipelines import pipeline
# from util import save_images
#
# pipe = pipeline('my-anytext-task', model='/models-oct-12/lightning_logs/version_0/checkpoints/last.ckpt', model_revision='v1.1.3')
# img_save_folder = "SaveImages"
# params = {
#     "show_debug": True,
#     "image_count": 2,
#     "ddim_steps": 20,
# }
#
# # 1. text generation
# mode = 'text-generation'
# input_data = {
#     "prompt": 'photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream',
#     "seed": 66273235,
#     "draw_pos": 'example_images/gen9.png'
# }
# results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
# if rtn_code >= 0:
#     save_images(results, img_save_folder)
#     print(f'Done, result images are saved in: {img_save_folder}')
# if rtn_warning:
#     print(rtn_warning)
#
# # 2. text editing
# mode = 'text-editing'
# input_data = {
#     "prompt": 'A cake with colorful characters that reads "EVERYDAY"',
#     "seed": 8943410,
#     "draw_pos": 'example_images/edit7.png',
#     "ori_image": 'example_images/ref7.jpg'
# }
# results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
# if rtn_code >= 0:
#     save_images(results, img_save_folder)
#     print(f'Done, result images are saved in: {img_save_folder}')
# if rtn_warning:
#     print(rtn_warning)
