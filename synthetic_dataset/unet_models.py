import torch
import torch.nn as nn
import torchvision.ops as ops


class DeformConvSequential(nn.Module):
    def __init__(self, offset_conv, deform_conv):
        super(DeformConvSequential, self).__init__()
        self.offset_conv = offset_conv
        self.deform_conv = deform_conv

    def forward(self, x):
        # Move the entire module to the device of x
        self.to(x.device)

        # the output from the offset_conv layer is passed as the offset parameter to the deform_conv layer.
        offset = self.offset_conv(x)
        return self.deform_conv(input=x, offset=offset)


class ModifiedUNet(nn.Module):
    """The U-Net model is a fully convolutional neural network and the modified network replaces all the Conv2d layers with DeformConv2d layers.
    In addition, the final layer of the U-Net needs to be modified to produce three outputs:
    - Binary Mask: A single-channel output with a sigmoid activation for the binary mask.
    - Color Map: A three-channel output (assuming RGB) for the color map.
    - Feature Map for Text Perceptual Loss: Depending on the design, this could be a feature map from one of the intermediate layers.
    """
    def __init__(self, base_model, base_model_final_channels, step, deformable=False):
        super(ModifiedUNet, self).__init__()
        self.step = step
        self.base_model = base_model
        self.base_model_final_channels = base_model_final_channels

        if step == 'extract':
            self.prediction_head = nn.Conv2d(in_channels=base_model_final_channels, out_channels=1, kernel_size=1)  # sigmoid is in nn.BCEWithLogitsLoss
        elif step == 'rectify':
            self.prediction_head = nn.Conv2d(in_channels=base_model_final_channels, out_channels=2, kernel_size=1)
        else:
            raise ValueError('step must be either "extract" or "rectify"')
        # self.color_map_head = nn.Conv2d(in_channels=self.base_model_final_channels, out_channels=3, kernel_size=1)

        if deformable:
            # Replace Conv2d with DeformConv2d
            self.replace_conv2d_with_deformconv2d(self.base_model)

    def replace_conv2d_in_sequential(self, sequential_module):
        """Replace all the Conv2d layers with DeformConv2d layers in a Sequential module.
        Example: (encoder1): Sequential(
                    (enc1conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (enc1norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (enc1relu1): ReLU(inplace=True)
                    (enc1conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (enc1norm2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (enc1relu2): ReLU(inplace=True)
                )
        becomes
                (encoder1): Sequential(
                    (enc1conv1): DeformConvSequential(
                      (offset_conv): Conv2d(3, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                      (deform_conv): DeformConv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    )
                    (enc1norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (enc1relu1): ReLU(inplace=True)
                    (enc1conv2): DeformConvSequential(
                      (offset_conv): Conv2d(32, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                      (deform_conv): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    )
                    (enc1norm2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (enc1relu2): ReLU(inplace=True)
                  )
        """
        new_sequential = nn.Sequential()
        for layer_name, layer in sequential_module.named_children():
            if isinstance(layer, nn.Conv2d):
                # Define the new DeformConv2d layer
                deform_conv = ops.DeformConv2d(layer.in_channels, layer.out_channels,
                                               kernel_size=layer.kernel_size, stride=layer.stride,
                                               padding=layer.padding, dilation=layer.dilation)

                # Define the offset convolution layer
                offset_channels = 2 * layer.kernel_size[0] * layer.kernel_size[1]
                offset_conv = nn.Conv2d(layer.in_channels, offset_channels, kernel_size=layer.kernel_size,
                                        stride=layer.stride, padding=layer.padding)

                # Replace the Conv2d layer with a sequential container of offset_conv and deform_conv
                new_layer = DeformConvSequential(offset_conv, deform_conv)
                new_sequential.add_module(layer_name, new_layer)
            else:
                new_sequential.add_module(layer_name, layer)

        return new_sequential

    def replace_conv2d_with_deformconv2d(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                # Replace Conv2d layers in the Sequential module
                new_sequential = self.replace_conv2d_in_sequential(module)
                setattr(model, name, new_sequential)


    def forward(self, x):
        # Forward pass through the base model
        base_output = self.base_model(x)
        output = self.prediction_head(base_output)
        # color_map = self.color_map_head(base_output)

        # Extract the feature map for perceptual loss
        # perceptual_feature_map = base_output  # or some intermediate layer

        return output #{'binary_mask': binary_mask, 'color_map': color_map}


# # Example usage:
# # Load the U-Net model from https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
# # set the out_channels from 1 to 16 and attach it to our ModifiedUNet's output layers
# unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#                             in_channels=3, out_channels=16, init_features=32, pretrained=False)
#
# unet_model = ModifiedUNet(unet_model, base_model_final_channels=16)  # the out_channels of the U-Net model
# print(unet_model)
