from torch.utils.data import Dataset
import json
from PIL import Image
import torch
from utils import transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CityScape(Dataset):
    def __init__(self, filename, size = (512, 256),transform=None):
        super(CityScape, self).__init__()
        f = open(filename)
        self.data = json.load(f)
        self.transform = transform
        self.size = size
        self.id_to_cat = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 1,
                    8: 1,
                    9: 1,
                    10: 1,
                    11: 2,
                    12: 2,
                    13: 2,
                    14: 2,
                    15: 2,
                    16: 2,
                    17: 3,
                    18: 3,
                    19: 3,
                    20: 3,
                    21: 4,
                    22: 4,
                    23: 5,
                    24: 6,
                    25: 6,
                    26: 7,
                    27: 7,
                    28: 7,
                    29: 7,
                    30: 7,
                    31: 7,
                    32: 7,
                    33: 7,
                    -1: 7
                }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, mask_path = self.data[index]
        image = Image.open(image_path).convert("RGB").resize(self.size)
        mask = Image.open(mask_path).convert('L').resize(self.size, Image.Resampling.NEAREST)
        if self.transform:
            image, mask = self.transform(image, mask)
        image = image/255.

        mask = mask.apply_(self.id_to_cat.get)
        # print(torch.unique(mask))
        mask_with_channels = torch.zeros((8, self.size[1], self.size[0]))
        for i in range(mask_with_channels.shape[0]):
            mask_with_channels[i] = torch.as_tensor((mask == i), dtype= torch.int8)


        image_stack = torch.zeros(2, 3, self.size[1], self.size[1])
        mask_stack = torch.zeros(2, 8, self.size[1], self.size[1])

        image_stack[0] = image[:, :, :self.size[1]]
        image_stack[1] = image[:, :, self.size[1]:]
        # image_stack[2] = image[:, self.size[1]//2:, :self.size[0]//2]
        # image_stack[3] = image[:, self.size[1]//2:, self.size[0]//2:]

        mask_stack[0] = mask_with_channels[:, :, :self.size[1]]
        mask_stack[1] = mask_with_channels[:, :, self.size[1]:]
        # mask_stack[2] = mask_with_channels[:, self.size[1]//2:, :self.size[0]//2]
        # mask_stack[3] = mask_with_channels[:, self.size[1]//2:, self.size[0]//2:]

        return image_stack.to(torch.float32), mask_stack.to(torch.float32)

    def collate_fn(self, batch):
        images = []
        masks = []

        for b in batch:
            images.append(b[0])
            masks.append(b[1])

        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)

        return images, masks


# Testing:
if __name__ == "__main__":
    train_file = "train.json"
    train_dataset = CityScape(train_file, (512, 256), transform)
    train_gen = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    for i, (image, mask) in enumerate(train_gen):
        print(image.shape, mask.shape)
        break
        # plt.imshow(image[0].permute(1, 2, 0))
        # plt.show()
        # plt.imshow(mask)
        # plt.show()