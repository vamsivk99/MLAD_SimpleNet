import torch
from torch.nn import functional as F 
from torchvision import models  


# Define a custom convolutional layer with batch normalization and activation
class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1):
        super().__init__()
        # Initialize convolutional layer with specified parameters
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2, bias=False)
        # Initialize batch normalization
        self.norm = torch.nn.BatchNorm2d(out_ch)
        # Set activation function
        self.relu = activation

    def forward(self, x):
        # Apply convolution, normalization, and activation in sequence
        return self.relu(self.norm(self.conv(x)))


# Define a residual block as used in ResNet architectures
class Residual(torch.nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        # Determine if a matching layer is needed
        self.add_m = s != 1 or in_ch != self.expansion * out_ch

        # Initialize convolutional layers
        self.conv1 = Conv(in_ch, out_ch, torch.nn.ReLU(inplace=True), 3, s)
        self.conv2 = Conv(out_ch, out_ch, torch.nn.Identity(), 3, 1)

        # Initialize a matching layer if needed
        if self.add_m:
            self.conv3 = Conv(in_ch, out_ch, torch.nn.Identity(), s=s)

    def zero_init(self):
        # Zero-initialize the batch normalization weights
        torch.nn.init.zeros_(self.conv2.norm.weight)

    def forward(self, x):
        # Apply convolutions and residual connection
        y = self.conv1(x)
        y = self.conv2(y)

        if self.add_m:
            x = self.conv3(x)

        return self.relu(x + y)


# Define a bottleneck block for deeper ResNet architectures
class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.add_m = s != 1 or in_ch != self.expansion * out_ch

        # Initialize convolutional layers
        self.conv1 = Conv(in_ch, out_ch, torch.nn.ReLU(inplace=True))
        self.conv2 = Conv(out_ch, out_ch, torch.nn.ReLU(inplace=True), 3, s)
        self.conv3 = Conv(out_ch, out_ch * self.expansion, torch.nn.Identity())

        # Initialize a matching layer if needed
        if self.add_m:
            self.conv4 = Conv(in_ch, self.expansion * out_ch, torch.nn.Identity(), s=s)

    def zero_init(self):
        # Zero-initialize the batch normalization weights
        torch.nn.init.zeros_(self.conv3.norm.weight)

    def forward(self, x):
        # Apply convolutions and residual connection
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        if self.add_m:
            x = self.conv4(x)

        return self.relu(x + y)


# Define the custom ResNet architecture
class ResNet(torch.nn.Module):
    def __init__(self, block, depth):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.fn = block
        filters = [3, 64, 128, 256]

        # p1/2 - Initial convolution and pooling layers
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU(inplace=True), 7, 2))
        # p2/4 - Layer 1
        for i in range(depth[0]):
            if i == 0:
                self.p2.append(torch.nn.MaxPool2d(3, 2, 1))
                self.p2.append(self.fn(filters[1], filters[1], 1))
            else:
                self.p2.append(self.fn(self.fn.expansion * filters[1], filters[1]))
        # p3/8 - Layer 2
        for i in range(depth[1]):
            if i == 0:
                self.p3.append(self.fn(self.fn.expansion * filters[1], filters[2], 2))
            else:
                self.p3.append(self.fn(self.fn.expansion * filters[2], filters[2], 1))
        # p4/16 - Layer 3
        for i in range(depth[2]):
            if i == 0:
                self.p4.append(self.fn(self.fn.expansion * filters[2], filters[3], 2))
            else:
                self.p4.append(self.fn(self.fn.expansion * filters[3], filters[3], 1))

        # Convert lists of layers into Sequential models
        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)

    def forward(self, x):
        # Forward pass through the ResNet
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        return [p3, p4]  # Return feature maps from the last two stages


# Define functions to create specific ResNet models
def resnet18():
    return ResNet(Residual, [2, 2, 2, 2])

def resnet34():
    return ResNet(Residual, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# Function to load a pre-trained ResNet backbone
def load_backbone(name):
    backbones = {
        "resnet50": models.resnet50(pretrained=True),
        "resnet101": models.resnet101(pretrained=True),
        "resnet152": models.resnet152(pretrained=True)
    }
    return backbones[name]


# Define a forward hook to extract intermediate layer outputs
class ForwardHook(object):
    def __init__(self, hook_dict, layer_name, last_layer_to_extract):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = (layer_name == last_layer_to_extract)

    def __call__(self, module, x, y):
        self.hook_dict[self.layer_name] = y
        return None


# Custom exception for handling layer extraction
class LayerException(Exception):
    pass


# Define a backbone class to wrap a pre-trained ResNet and extract features
class Backbone(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.layers = ['layer2', 'layer3']
        self.backbone = load_backbone('resnet50')

        # Remove existing hooks if any
        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        # Register forward hooks on the specified layers
        for layer in self.layers:
            forward_hook = ForwardHook(self.outputs, layer, self.layers[-1])
            block = self._get_block(layer)
            if isinstance(block, torch.nn.Sequential):
                self.backbone.hook_handles.append(block[-1].register_forward_hook(forward_hook))
            else:
                self.backbone.hook_handles.append(block.register_forward_hook(forward_hook))
        self.to(self.device)

    def _get_block(self, layer):
        # Helper function to get the block corresponding to the layer name
        if "." in layer:
            block, index = layer.split(".")
            block = self.backbone.__dict__["_modules"][block]
            index = int(index) if index.isnumeric() else block.__dict__["_modules"][index]
        else:
            block = self.backbone.__dict__["_modules"][layer]
        return block

    def forward(self, images):
        # Forward pass to extract features
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LayerException:
                pass
        return [self.outputs[layer] for layer in self.layers]


# Define a generator class (potentially for a GAN)
class Generator(torch.nn.Module):
    def __init__(self, args, in_ch, out_ch):
        super().__init__()
        self.args = args
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.linear = torch.nn.Linear(in_ch, out_ch)

        # Initialize weights of linear layers
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    @staticmethod
    def patchify(features):
        # Extract patches from the input features
        k, p, s = 3, 1, 1
        unfolded_features = F.unfold(features, kernel_size=k, padding=p, stride=s)
        unfolded_features = unfolded_features.reshape(*features.shape[:2], k, k, -1)
        return unfolded_features.permute(0, 4, 1, 2, 3)

    def forward(self, x):
        # Forward pass for the generator
        x = [self.patchify(i) for i in x]
        shapes = self.args.input_size // 8, self.args.input_size // 16

        for i in range(1, len(x)):
            y = x[i]
            y = y.reshape(y.shape[0], shapes[i], shapes[i], *y.shape[2:])
            y = y.permute(0, -3, -2, -1, 1, 2)
            y = y.reshape(-1, *y.shape[-2:])
            y = F.interpolate(y.unsqueeze(1), size=(shapes[0], shapes[0]), mode="bilinear", align_corners=False)
            y = y.squeeze(1)
            y = y.reshape(*perm_base_shape[:-2], shapes[0], shapes[0])
            y = y.permute(0, -2, -1, 1, 2, 3)
            y = y.reshape(len(y), -1, *y.shape[-3:])
            x[i] = y
        x = [x.reshape(-1, *x.shape[-3:]) for x in x]

        pool_features = []
        for y in x:
            y = y.reshape(len(y), 1, -1)
            pool_features.append(F.adaptive_avg_pool1d(y, self.in_ch).squeeze(1))
        x = torch.stack(pool_features, dim=1)
        x = x.reshape(len(x), 1, -1)
        x = F.adaptive_avg_pool1d(x, self.in_ch)
        x = x.reshape(len(x), -1)
        return self.linear(x)


# Define a discriminator class (potentially for a GAN)
class Discriminator(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_ch, out_ch),
            torch.nn.BatchNorm1d(out_ch),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(out_ch, 1, bias=False)
        )

        # Initialize weights of linear layers
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # Forward pass for the discriminator
        return self.fc(x)
