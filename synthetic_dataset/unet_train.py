import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Subset
import yaml
import os
import json
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import shutil
from PIL import Image

from unet_models import ModifiedUNet
from unet_dataloader import SyntheticDataset


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


class CombinedLoss(nn.Module):
    def __init__(self, rank, bce_pos_weight, dice_smooth=1.0, bce_weight=0.3, dice_weight=0.7):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight]).to(rank))
        self.dice_smooth = dice_smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def dice_loss(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get [0,1] range predictions
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.dice_smooth) / (inputs.sum() + targets.sum() + self.dice_smooth)
        return 1 - dice_coeff

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def training(gpu, args, train_subset, val_subset):
    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    writer = None
    if rank == 0:
        log_dir = 'runs/train_unet'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)  # remove the old log directory if it exists
        writer = SummaryWriter(log_dir)

    # Set up the dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args['training']['batch_size'], shuffle=False, num_workers=4, drop_last=True, sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, num_replicas=world_size, rank=rank)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args['training']['batch_size'], shuffle=False, num_workers=4, drop_last=True, sampler=val_sampler)
    print("Finished loading the datasets...")

    # Build the UNet model
    unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                in_channels=3, out_channels=args['model']['out_channels'], init_features=32, pretrained=False)
    base_model_final_channels = args['model']['out_channels'] if args['training']['step'] == 'extract' else args['model']['bottleneck_out_dim']
    unet_model = ModifiedUNet(unet_model, base_model_final_channels=base_model_final_channels, step=args['training']['step'], deformable=False)
    unet_model = DDP(unet_model).to(rank)
    unet_model.train()

    map_location = {'cuda:%d' % rank: 'cuda:%d' % 0}
    if args['training']['continue_train']:
        unet_model.load_state_dict(torch.load(args['training']['checkpoint_path'] + '/unet_model_' + str(args['training']['start_epoch'] - 1) +
                                              '_' + args['training']['step'] + '.pth', map_location=map_location))
        print('Loaded checkpoint from %s' % args['training']['checkpoint_path'])

    # Set up optimizer, scheduler, and loss functions
    optimizer = optim.AdamW(unet_model.parameters(), lr=args['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args['training']['patience'])
    criterion = CombinedLoss(rank, bce_pos_weight=args['training']['bce_pos_weight'], bce_weight=args['training']['bce_weight'], dice_weight=args['training']['dice_weight'])
    criterion_mse = nn.MSELoss()

    # Training loop
    running_loss_total, running_loss_coef, running_loss_reg = 0.0, 0.0, 0.0
    for epoch in range(args['training']['start_epoch'], args['training']['num_epochs']):
        print('Start Training... EPOCH %d / %d\n' % (epoch, args['training']['num_epochs']))

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            if args['training']['step'] == 'extract':
                sources, targets = batch
                sources = sources.to(rank)
                targets = targets.to(rank)
            else:
                sources, target_corners, target_midlines = batch
                sources = sources.to(rank)
                target_corners = target_corners.to(rank)
                target_midlines = target_midlines.to(rank)

            # Forward pass
            outputs = unet_model(sources)

            # Compute loss
            predictions = outputs.squeeze(dim=1)
            if args['training']['step'] == 'extract':
                loss = criterion.forward(predictions, targets)
                running_loss_total += loss.item()
            else:
                pred_corners = predictions[:, 0]
                pred_midline = predictions[:, 1]

                loss = criterion.forward(pred_corners, target_corners) + criterion.forward(pred_midline, target_midlines)
                running_loss_total += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the training loss
            global_step = batch_idx + len(train_loader) * epoch
            if rank == 0 and running_loss_total / (global_step + 1e-10) < 10:   # avoid recording the first few large losses for better visualization
                writer.add_scalar('Loss/train_total', running_loss_total / (global_step + 1e-10), global_step)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if batch_idx % args['training']['print_every'] == 0 or batch_idx + 1 == len(train_loader):
                print('Training Rank %d, epoch %d, batch %d, lr %.6f, loss %.6f' % (rank, epoch, batch_idx, optimizer.param_groups[0]['lr'], running_loss_total / (global_step + 1e-10)))

            if batch_idx > 0 and batch_idx % args['training']['val_every'] == 0:
                # Validation
                val_loss = validation(args, unet_model, val_loader, criterion, epoch, writer, rank, batch_idx, step=args['training']['step'])

                # Aggregate validation loss across all GPUs for consistent learning rate scheduler update
                torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
                global_val_loss = val_loss / torch.distributed.get_world_size()
                if rank == 0:
                    scheduler.step(global_val_loss)
                dist.monitored_barrier()

        # Save model checkpoint every epoch
        if rank == 0:
            torch.save(unet_model.state_dict(), args['training']['checkpoint_path'] + '/unet_model_' + args['training']['step'] + '_' + str(epoch) + '_' + args['training']['step'] + '.pth')
        dist.monitored_barrier()

    dist.destroy_process_group()  # clean up
    if rank == 0:
        writer.close()
    print('FINISHED TRAINING\n')


def validation(args, unet_model, val_loader, criterion, epoch, writer, rank, batch_idx, step):
    print('Start Validation... EPOCH %d / %d\n' % (epoch, args['training']['num_epochs']))
    unet_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(tqdm(val_loader)):
            if args['training']['step'] == 'extract':
                val_sources, val_targets = batch
                val_sources = val_sources.to(rank)
                val_targets = val_targets.to(rank)
            else:
                val_sources, val_target_corners, val_target_midlines = batch
                val_sources = val_sources.to(rank)
                val_target_corners = val_target_corners.to(rank)
                val_target_midlines = val_target_midlines.to(rank)

            # Forward pass
            val_outputs = unet_model(val_sources)

            # Compute loss
            val_predictions = val_outputs.squeeze(dim=1)
            if args['training']['step'] == 'extract':
                loss = criterion.forward(val_predictions, val_targets)
            else:
                val_pred_corners = val_predictions[:, 0]
                val_pred_midline = val_predictions[:, 1]

                val_loss += criterion.forward(val_pred_corners, val_target_corners) + criterion.forward(val_pred_midline, val_target_midlines)

            if val_batch_idx == 0 and rank == 0:
                if args['training']['step'] == 'extract':
                    if epoch == 0:
                        save_image(val_sources[0].cpu(), 'outputs/' + args['training']['step'] + '/source_' + step + '_' + str(rank) + '.png')
                        save_image(val_targets[0].cpu(), 'outputs/' + args['training']['step'] + '/target_' + step + '_' + str(rank) + '.png')

                    save_image(val_predictions[0].cpu(), 'outputs/predicted_' + str(epoch) + '_' + str(batch_idx) + '_' + step + '_' + str(rank) + '.png')
                else:
                    if epoch == 0:
                        save_image(val_sources[0].cpu(), 'outputs/' + args['training']['step'] + '/source_' + step + '_' + str(rank) + '.png')
                        save_image(val_target_corners[0].cpu(), 'outputs/' + args['training']['step'] + '/target_corners_' + step + '_' + str(rank) + '.png')
                        save_image(val_target_midlines[0].cpu(), 'outputs/' + args['training']['step'] + '/target_midlines_' + step + '_' + str(rank) + '.png')

                    save_image(val_pred_corners[0].cpu(), 'outputs/predicted_corners_' + str(epoch) + '_' + str(batch_idx) + '_' + step + '_' + str(rank) + '.png')
                    save_image(val_pred_midline[0].cpu(), 'outputs/predicted_midline_' + str(epoch) + '_' + str(batch_idx) + '_' + step + '_' + str(rank) + '.png')

    val_loss /= len(val_loader)

    # Record the validation loss
    global_step = len(val_loader) * epoch
    if rank == 0:
        writer.add_scalar('Loss/val', val_loss, global_step)
    print('Validation Rank %d, epoch %d, val_loss %.3f' % (rank, epoch, val_loss))
    unet_model.train()

    print('FINISHED VALIDATION\n')
    return val_loss


if __name__ == "__main__":
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # load hyperparameters
    try:
        with open('unet_train_config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--step', type=str, default="extract", help='Set training step for the UNet on the synthetic dataset (extract, rectify)')
    cmd_args = parser.parse_args()
    args['training']['step'] = cmd_args.step if cmd_args.step is not None else args['training']['step']

    # Load dataset
    full_dataset = SyntheticDataset(args['dataset']['images_dir'], args['dataset']['targets_dir'], args['dataset']['targets_curved_dir'],
                                    args['dataset']['coeffs_dir'], args['dataset']['image_size'], args['training']['step'])
    train_subset_idx = torch.randperm(len(full_dataset))[:int(args['dataset']['percent_train'] * args['dataset']['train_val_split'] * len(full_dataset))]
    val_subset_idx = torch.randperm(len(full_dataset))[:int(args['dataset']['percent_valid'] * (1 - args['dataset']['train_val_split']) * len(full_dataset))]
    train_subset = Subset(full_dataset, train_subset_idx)
    val_subset = Subset(full_dataset, val_subset_idx)
    print('full_dataset', len(full_dataset))
    print('num of training and validating images:', len(train_subset), len(val_subset))

    print(args)
    # select training or evaluation
    if args['training']['run_mode'] == 'train':
         mp.spawn(training, nprocs=world_size, args=(args, train_subset, val_subset))
    else:
        print('Invalid arguments.')
