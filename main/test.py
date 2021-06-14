import torch
import torchvision.transforms as T

from dataloading.datasets import RGZ20k
from dataloading.utils import Circle_Crop
from paths import Path_Handler

paths = Path_Handler()
path_dict = paths._dict()

rescaling = lambda x: (x - 0.5) * 2.0
trans = T.Compose([T.RandomRotation(180), T.ToTensor(), rescaling, Circle_Crop()])
trans = T.Compose([T.RandomRotation(180), T.ToTensor(), Circle_Crop()])

train_loader = torch.utils.data.DataLoader(
    RGZ20k(path_dict["rgz"], download=True, train=True, transform=trans),
    batch_size=50,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    RGZ20k(path_dict["rgz"], download=True, train=False, transform=trans),
    batch_size=50,
    shuffle=True,
)

x, y = next(iter(train_loader))
print(torch.max(x), torch.min(x))


print(next(iter(test_loader)))
print(next(iter(train_loader)))
