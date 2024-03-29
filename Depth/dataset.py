from torch.utils.data import Dataset
import json
from PIL import Image
import torch
from utils import transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CityScapeDepth(Dataset):
    def __init__(self, filename, size = (512, 256),transform=None):
        super(CityScapeDepth, self).__init__()
        f = open(filename)
        self.data = json.load(f)
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, mask_path = self.data[index]
        image = Image.open(image_path).convert("RGB").resize(self.size)
        mask = Image.open(mask_path).convert('L').resize(self.size, Image.Resampling.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)
        image = image / 255.
        mask = mask / 255.

        # image_stack = torch.zeros(4, 3, self.size[1]//2, self.size[0]//2)
        # mask_stack = torch.zeros(4, 1, self.size[1]//2, self.size[0]//2)

        # image_stack[0] = image[:, :self.size[1]//2, :self.size[0]//2]
        # image_stack[1] = image[:, :self.size[1]//2, self.size[0]//2:]
        # image_stack[2] = image[:, self.size[1]//2:, :self.size[0]//2]
        # image_stack[3] = image[:, self.size[1]//2:, self.size[0]//2:]

        # mask_stack[0] = mask[:, :self.size[1]//2, :self.size[0]//2]
        # mask_stack[1] = mask[:, :self.size[1]//2, self.size[0]//2:]
        # mask_stack[2] = mask[:, self.size[1]//2:, :self.size[0]//2]
        # mask_stack[3] = mask[:, self.size[1]//2:, self.size[0]//2:]

        image_stack = torch.zeros(2, 3, self.size[1], self.size[0]//2)
        mask_stack = torch.zeros(2, 1, self.size[1], self.size[0]//2)

        image_stack[0] = image[:, :, :self.size[0]//2]
        image_stack[1] = image[:, :, self.size[0]//2:]

        mask_stack[0] = mask[:, :, :self.size[0]//2]
        mask_stack[1] = mask[:, :, self.size[0]//2:]

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
    train_dataset = CityScapeDepth(train_file, (512, 256), transform)
    train_gen = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    for i, (image, mask) in enumerate(train_gen):
        print(image.shape, mask.shape)
        plt.imshow(image[0].permute(1, 2, 0))
        plt.show()
        plt.imshow(mask[0].permute(1, 2, 0), cmap="Greys")
        plt.show()
        break
