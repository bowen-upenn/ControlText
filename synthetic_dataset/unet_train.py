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
from generate_text_transformation_pairs import recover_from_perspective_torch, recover_from_curvature_torch, \
    recover_from_perspective, recover_from_curvature, recover_from_perspective_torch_with_flow, recover_from_curvature_torch_with_flow, apply_optical_flow_to_recover


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
            # sources, targets = batch
            # sources = sources.to(rank)
            # targets = targets.to(rank)
            if args['training']['step'] == 'extract':
                sources, targets = batch
                sources = sources.to(rank)
                targets = targets.to(rank)
            elif args['training']['step'] == 'rectify':
                sources, target_imgs, coeffs = batch
                sources = sources.to(rank)
                targets = coeffs.to(rank)
            else:
                assert False, "Invalid training step."

            # Forward pass
            outputs = unet_model(sources)

            # Compute loss
            predictions = outputs.squeeze(dim=1)

            loss = criterion.forward(predictions, targets)
            running_loss_total += loss.item()

            if args['training']['step'] == 'rectify':
                perspective_coefficients = predictions[:, 3:]
                identity_matrices = torch.eye(3).unsqueeze(0).repeat(perspective_coefficients.size(0), 1, 1).to(rank).flatten(1, 2)
                loss_coef = criterion_mse.forward(predictions, targets)
                loss_reg = args['training']['lambda_reg'] * criterion.forward(perspective_coefficients, identity_matrices)
                loss = loss_coef + loss_reg
                running_loss_coef += loss_coef.item()
                running_loss_reg += loss_reg.item()

            #     curvature_coefficients = targets[1][:3]
            #     perspective_coefficients = targets[1][3:].reshape(3, 3)
            #     # recovered_img, recovered_flow = recover_from_perspective_torch_with_flow(sources[1], perspective_coefficients)
            #     # print('recovered_flow', recovered_flow.shape)
            #     recovered_flow = torch.zeros(512, 512, 2).to(rank)
            #     recovered_img, recovered_flow = recover_from_curvature_torch_with_flow(sources[1], curvature_coefficients, recovered_flow)
            #     save_image(sources[1], 'toy_examples/test_flow_source_img.png')
            #     save_image(recovered_img, 'toy_examples/test_flow_recovered_img.png')
            #     save_image(recovered_flow[:, :, 0], 'toy_examples/test_flow_recovered_flow_x.png')
            #     save_image(recovered_flow[:, :, 1], 'toy_examples/test_flow_recovered_flow_y.png')
            #
            #     print('recovered_flow', recovered_flow.shape, 'sources', sources[1].shape)
            #     recovered_img_from_flow = apply_optical_flow_to_recover(images[1], recovered_flow)
            #     save_image(recovered_img_from_flow, 'toy_examples/test_flow_recovered_img_from_flow.png')
            else:
                loss = criterion.forward(predictions, targets)

            # if args['training']['step'] == 'rectify':
            #     # Add recovery loss
            #     # curvature_coefficients = predictions[:, :3]
            #     # perspective_coefficients = predictions[:, 3:]#.reshape(3, 3)
            #     curvature_coefficients = targets[:, :3]
            #     perspective_coefficients = targets[:, 3:]#.reshape(3, 3)
            #     # print('curvature_coefficients', curvature_coefficients, 'perspective_coefficients', perspective_coefficients)
            #
            #     recovered_imgs = recover_from_perspective_torch(sources, perspective_coefficients)
            #     recovered_imgs = recover_from_curvature_torch(recovered_imgs, curvature_coefficients)
            #
            #     # save_image(recovered_imgs[0], 'outputs/test_recovered_0.png')
            #     # save_image(sources[0], 'outputs/test_source_0.png')
            #     # save_image(target_imgs[0], 'outputs/test_target_0.png')
            #     # save_image(recovered_imgs[1], 'outputs/test_recovered_1.png')
            #     # save_image(sources[1], 'outputs/test_source_1.png')
            #     # save_image(target_imgs[1], 'outputs/test_target_1.png')

            # # Treat the rendered texts as a binary mask
            # if args['training']['step'] == 'extract':
            #     loss = criterion_imgs.forward(predictions, targets)
            # elif args['training']['step'] == 'rectify':
            #     # loss = criterion_coeffs.forward(predictions, targets)
            #     loss_recovery = criterion_imgs.forward(recovered_imgs, sources)
            #     loss_coeffs = criterion_coeffs.forward(predictions, targets)
            #     loss = loss_recovery + loss_coeffs
            #     running_loss_recovery += loss_recovery.item()
            #     running_loss_coeffs += loss_coeffs.item()
            # else:
            #     assert False, "Invalid training step."

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
                if args['training']['step'] == 'extract':
                    print('Training Rank %d, epoch %d, batch %d, lr %.6f, loss %.6f' % (rank, epoch, batch_idx, optimizer.param_groups[0]['lr'], running_loss_total / (global_step + 1e-10)))
                elif args['training']['step'] == 'rectify':
                    print('Training Rank %d, epoch %d, batch %d, lr %.6f, loss %.6f, loss_coef %.6f, loss_reg %.6f' % (rank, epoch, batch_idx, optimizer.param_groups[0]['lr'],
                          running_loss_total / (global_step + 1e-10), running_loss_coef / (global_step + 1e-10), running_loss_reg / (global_step + 1e-10)))
                else:
                    assert False, "Invalid training step."

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
            torch.save(unet_model.state_dict(), args['training']['checkpoint_path'] + '/unet_model_' + str(epoch) + '_' + args['training']['step'] + '.pth')
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
                val_sources, val_targets = val_batch
                val_sources = val_sources.to(rank)
                val_targets = val_targets.to(rank)
                # criterion = criterion_imgs
            else:
                val_sources, val_target_imgs, val_coeffs = val_batch
                val_sources = val_sources.to(rank)
                val_targets = val_coeffs.to(rank)
                # criterion = criterion_coeffs

            # Forward pass
            val_outputs = unet_model(val_sources)

            # Compute loss
            val_predictions = val_outputs.squeeze(dim=1)
            val_loss += criterion.forward(val_predictions, val_targets)

            if val_batch_idx == 0 and rank == 0:
                if args['training']['step'] == 'extract':
                    if epoch == 0:
                        save_image(val_sources[0].cpu(), 'outputs/source_' + step + '_' + str(rank) + '.png')
                        save_image(val_targets[0].cpu(), 'outputs/target_' + step + '_' + str(rank) + '.png')

                    save_image(val_predictions[0].cpu(), 'outputs/predicted_' + str(epoch) + '_' + str(batch_idx) + '_' + step + '_' + str(rank) + '.png')
                else:
                    if epoch == 0:
                        save_image(val_sources[0].cpu(), 'outputs/source_' + step + '_' + str(rank) + '.png')
                        save_image(val_target_imgs[0].cpu(), 'outputs/target_' + step + '_' + str(rank) + '.png')

                    # Transform the source image back to canonical image using the predicted coefficients
                    curvature_coefficients = val_predictions[0][:3]
                    perspective_coefficients = val_predictions[0][3:].reshape(3, 3)
                    # assert np.array_equal(perspective_coefficients.flatten(), val_predictions[0][3:].cpu().numpy()), "Reshape is incorrect."

                    recovered_img = recover_from_perspective_torch(val_sources[0], perspective_coefficients, batch=False)
                    recovered_img = recover_from_curvature_torch(recovered_img, curvature_coefficients, batch=False)
                    # print('curvature_coefficients', curvature_coefficients)
                    # print('filename', 'outputs/predicted_' + str(epoch) + '_' + str(batch_idx) + '_' + step + '.png')
                    save_image(recovered_img, 'outputs/predicted_' + str(epoch) + '_' + str(batch_idx) + '_' + step + '_' + str(rank) + '.png')

                    coeffs_dict = {'predict': val_predictions[0].cpu().numpy().tolist(), 'target': val_coeffs[0].cpu().numpy().tolist()}
                    # print('coeffs_dict', coeffs_dict)
                    with open('outputs/coeffs_' + str(epoch) + '_' + str(batch_idx) + '_' + step + '_' + str(rank) + '.json', 'w') as f:
                        json.dump(coeffs_dict, f)

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
