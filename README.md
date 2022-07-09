# MobileOne - An Improved One millisecond Mobile Backbone Pytorch implementation

<img src="https://user-images.githubusercontent.com/56075061/178083730-4cdcc34d-2ba2-4f56-a97e-c73aa24f545d.png" alt="MobileOne block" width="300"/>

This is an unofficial Pytorch implementation of [MobileOne](https://arxiv.org/pdf/2206.04040.pdf) from Apple. Similar to [RepVGG](https://github.com/DingXiaoH/RepVGG), they introduce a block with reparametrizable convolutions called MobileOne block.

This implementation is based on [RepVGG](https://github.com/DingXiaoH/RepVGG), with Stack and Sum operations suitable for edge devices.

Currently, S0 is implemented by default, but other backbones can be trivially configured. You can find the model trained on Imagenette in the releases (or click [here to download](https://github.com/tersekmatija/mobileone/releases/download/0.0.1/mobile-one-s0.pth.tar)).

## Setup

```
pip3 install -r requirements.txt
```

## Demo

Download weights from the demo and some images with Imagenette classes. Call
```
python demo.py -s path/to/image/or/dir -w path/to/weights
```

You will see a matplotlib window with image and inferred class.

## Compare and time measure

Call `python compare.py` to verify that the model returns the same values in deploy mode. Code also contains script for measuring inference speed on CPU.

| Backbone    | Normal (ms) |  Deploy (ms)| 
| ----------- | ----------- | ----------- |
| MobileOne-S0| 0.0664      | 0.0077      |


## TODO
- [ ] Add support for other MobileOne backbones
- [ ] Add export options
- [ ] Provide training scripts

## Contributions

All contributions and improvements through PRs welcome!