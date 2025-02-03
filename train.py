import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from t3_dataset import T3DataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

NUM_NODES = 1
# Configs
batch_size = 1  # default 6
grad_accum = 6  # enable perceptual loss may cost a lot of VRAM, you can set a smaller batch_size and make sure grad_accum * batch_size = 6
ckpt_path = './models/lightning_logs/version_5/checkpoints/epoch=11-step=849999.ckpt'  # if not None, load ckpt_path and continue training task, will not load "resume_path"
resume_path = None # './models/anytext_v1.1.ckpt' # './models/anytext_sd15_scratch.ckpt'  # finetune from scratch
model_config = './models_yaml/anytext_sd15_perloss.yaml'  # use anytext_sd15_perloss.yaml to enable perceptual loss
invalid_json_path = './Rethinking-Text-Segmentation/log/images/ocr_verified/invalid_gly_lines.json'
logger_freq = 5000
learning_rate = 2e-5  # default 2e-5
mask_ratio = 1  # default 0.5, ratio of mask for inpainting(text editing task), set 0 to disable
wm_thresh = 0.5  # set 0.5 to skip watermark imgs from training(ch:~25%, en:~8%, @Precision93.67%+Recall88.80%), 1.0 not skip
root_dir = './models'  # path for save checkpoints
dataset_percent = 1.0  # 1.0 use full datasets, 0.0566 use ~200k images for ablation study
save_steps = 5000  # step frequency of saving checkpoints
save_epochs = None  # epoch frequency of saving checkpoints
max_epochs = 50  # default 60
assert (save_steps is None) != (save_epochs is None)


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    log_img = os.path.join(root_dir, 'image_log/train')
    if os.path.exists(log_img):
        try:
            shutil.rmtree(log_img)
        except OSError:
            pass
    # model = create_model(model_config).cpu()
    model = create_model(model_config)
    if ckpt_path is None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = True
    model.only_mid_control = False
    model.unlockKV = False

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=save_steps,
        every_n_epochs=save_epochs,
        save_top_k=-1,
        save_last=True,
        monitor="global_step",
        mode="max",
    )
    json_paths = [
        # r'/tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json',
        # r'/tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json',
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
        # r'./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/laion_word',
        # r'./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/wukong_word',
        # r'./Rethinking-Text-Segmentation/log/images/ocr_verified/laion_test',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/laion',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/wukong_1of5',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/wukong_2of5',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/wukong_3of5',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/wukong_4of5',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/wukong_5of5',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/Art',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/COCO_Text',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/icdar2017rctw',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/LSVT',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/mlt2019',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/MTWI2018',
        r'./Rethinking-Text-Segmentation/log/images/ocr_verified/ReCTS'
    ]
    dataset = T3DataSet(json_paths, glyph_paths, max_lines=5, max_chars=20, caption_pos_prob=0.0, mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=2, percent=dataset_percent, debug=False, using_dlc=False, wm_thresh=wm_thresh, invalid_json_path=invalid_json_path)
    # sampler = DistributedSampler(dataset)
    # dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, sampler=sampler)
    dataloader = DataLoader(dataset, num_workers=8, persistent_workers=True, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=-1, precision=32, max_epochs=max_epochs, num_nodes=NUM_NODES, accumulate_grad_batches=grad_accum, callbacks=[logger, checkpoint_callback], default_root_dir=root_dir, strategy='ddp')
    print('Start training...')
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)
