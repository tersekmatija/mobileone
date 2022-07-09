import torch
import torch.nn as nn

from layers import MobileOneBlock

class MobileOne(nn.Module):

    def __init__(self, num_classes=10, a = [0.75,0.75,1,1,1,2,2], k = [4,4,4,4,4,4]):

        super(MobileOne, self).__init__()

        ch = [int(x * y) for x,y in zip([64, 64, 128, 256, 256, 512], a)]

        self.block1 = MobileOneBlock(3, ch[0], k[0], stride=2)

        self.block2 = nn.Sequential(
                MobileOneBlock(ch[0], ch[1], k[1], stride=2), 
                MobileOneBlock(ch[1], ch[1], k[1])
            )

        self.block3 = nn.Sequential(
                MobileOneBlock(ch[1], ch[2], k[2], stride=2), 
                *[MobileOneBlock(ch[2], ch[2], k[2]) for _ in range(7)]
            )

        self.block4 = nn.Sequential(
                MobileOneBlock(ch[2], ch[3], k[3], stride=2),
                *[MobileOneBlock(ch[3], ch[3], k[3]) for _ in range(4)]
            )
            
        self.block5 = nn.Sequential(
                MobileOneBlock(ch[3], ch[4], k[4], stride=2),
                *[MobileOneBlock(ch[4], ch[4], k[4]) for _ in range(4)]
            )

        self.block6 = MobileOneBlock(ch[4], ch[5], k[5], stride=2)
        self.linear = nn.Linear(ch[5], num_classes)

    def switch_to_deploy(self):

        self.block1.switch_to_deploy()
        for b in self.block2:
            b.switch_to_deploy()
        for b in self.block3:
            b.switch_to_deploy()
        for b in self.block4:
            b.switch_to_deploy()
        for b in self.block5:
            b.switch_to_deploy()
        self.block6.switch_to_deploy()


    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = x.mean([2,3])

        x = self.linear(x)

        return x

