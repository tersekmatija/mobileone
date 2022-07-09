import os
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import glob
import matplotlib.pyplot as plt

from mobileone import MobileOne

_IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
_IMAGENETTE_CLASSES = ["tench", "English springer", "cassette player", "chain saw", "church", "French horn",
                        "garbage truck", "gas pump", "golf ball", "parachute"]


# parse args
parser = argparse.ArgumentParser(description='MobileOne demo inference')
parser.add_argument('-w', '--weights', type=str, default="mobile-one-s0.pth.tar", help='Path to weights')
parser.add_argument('-s', '--source', type=str, required=True, help="Path to image or directory")
args = parser.parse_args()

# define model
model = MobileOne()
state_dict = torch.load(args.weights, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.eval()
model.switch_to_deploy()

# define transforms
transforms = torch.nn.Sequential(
    transforms.Normalize(_IMAGENET_DEFAULT_MEAN, _IMAGENET_DEFAULT_STD),
    transforms.Resize((224, 224))
)
softmax = torch.nn.Softmax(dim=0)

# get files
files = []
if os.path.isdir(args.source):
    ext = ['png', 'jpg', 'gif', 'jpeg']    # Add image formats here
    files = []
    for e in ext:
        files.extend(glob.glob(args.source + '*.' + e))
else:
    files.append(args.source)

for fn in files:
    img_ = torchvision.io.read_image(fn)
    img = img_.type(torch.float32).unsqueeze(0) / 255  # [3, H, W] uint8 --> [1, 3, H, W] float32
    img = transforms(img)                                                       # [3, H, W] --> [3, 224, 224]

    output = model(img)

    # parse
    idx = torch.argmax(output[0])
    probs = softmax(output[0])

    plt.imshow(img_.numpy().transpose(1,2,0))
    plt.title(f"{_IMAGENETTE_CLASSES[idx]}: {probs[idx]}")
    plt.show()

    print(f"{_IMAGENETTE_CLASSES[idx]}: {probs[idx]}")
    

