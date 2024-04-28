import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import yaml

from unet_models import ModifiedUNet


def load_checkpoint(model, checkpoint_path, device):
    if device.type == 'cuda':
        device_index = device.index if device.index is not None else 0
    else:
        device_index = 'cpu'
    map_location = lambda storage, loc: storage.cuda(device_index)

    # Load the checkpoint on CPU
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Adjust the keys: remove 'module.' prefix
    adjusted_checkpoint = {}
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` when not using DDP in model inference
        adjusted_checkpoint[name] = v

    # Load the weights into the model
    model.load_state_dict(adjusted_checkpoint)
    model.to(device)
    return model


def inference(args, model_path, image_path, output_path, device):
    # Load the model
    unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=args['model']['out_channels'], init_features=32, pretrained=False)
    unet_model = ModifiedUNet(unet_model, base_model_final_channels=args['model']['out_channels'])
    # print(unet_model)

    unet_model = load_checkpoint(unet_model, model_path, device)
    unet_model.eval()

    # Load and transform the image
    transform = transforms.Compose([
        transforms.Resize((args['dataset']['image_size'], args['dataset']['image_size'])),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = unet_model(image)
        binary_mask = torch.sigmoid(outputs['binary_mask']).squeeze(0).cpu()
        color_map = outputs['color_map'].squeeze(0).cpu()

    # Process output
    binary_mask = (binary_mask > 0.5).float()  # Thresholding the binary mask
    colored_text = color_map * binary_mask.expand_as(color_map)  # Apply color map to the text

    # Convert to PIL Image and save
    binary_mask = transforms.ToPILImage()(binary_mask)
    binary_mask.save(output_path + '_binary_mask.png')
    colored_text = transforms.ToPILImage()(colored_text)
    colored_text.save(output_path + '_colored_text.png')


if __name__ == "__main__":
    # This function performs single-image inference
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # load hyperparameters
    try:
        with open('unet_train_config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file', e)

    model_path = 'unet_ckpts/unet_model_' + str(args['training']['test_epoch']) + '.pth'
    image_path = 'toy_examples/noisy_wrapped_text_2.png'
    output_path = 'toy_examples/recovered_text_2'

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())

    print(args)
    inference(args, model_path, image_path, output_path, device)
