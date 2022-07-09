import time
import torch
from layers import MobileOneBlock
from mobileone import MobileOne

torch.manual_seed(0)

# options
options = [
    [3, 128, 3, 1],
    [3, 128, 3, 2],
    [3, 128, 1, 1],
]

print(f"--- Comparing blocks ---")

# compare
img = torch.randn(1, 3, 224, 224)
for opt in options:

    print(f"Comparing for {opt}")

    m = MobileOneBlock(opt[0], opt[1], k=opt[2], stride=opt[3])
    m.eval()

    out1 = m(img)
    m.switch_to_deploy()
    out2 = m(img)

    print(f"Difference: {torch.sum((out1-out2) ** 2)}")


print(f"--- Comparing backbones ---")

m = MobileOne()

t = time.monotonic()
out1 = m(img)
print(f"Inference took {time.monotonic()-t}s")

m.switch_to_deploy()

t = time.monotonic()
out2 = m(img)
print(f"Inference took {time.monotonic()-t}s")




