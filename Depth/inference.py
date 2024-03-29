import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model_v2 import *
from dataset import CityScapeDepth
from utils import transform

@torch.no_grad()
def inference(input, model):
    pred = model(input).cpu()
    pred = pred.permute(0, 2, 3, 1).detach().numpy()
    return pred

if __name__ == "__main__":
    checkpoint = "Depth_v9.pth.tar"
    test_file = "val.json"
    size = (512, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f, arr = plt.subplots(4, 2, figsize=(8, 8))
    model = torch.load(checkpoint)['model'].to(device)
    test_dataset = CityScapeDepth(test_file, size, transform)
    val_gen = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        num_workers=8,
        collate_fn=test_dataset.collate_fn
    )
    for image, mask in val_gen:
        print(image.shape)
        image, mask = image.to(device), mask
        pred = inference(image, model)
        mask = mask.permute(0, 2, 3, 1).detach().numpy()
        for i in range(pred.shape[0]):
            arr[i][0].imshow(pred[i])
            arr[i][1].imshow(mask[i])
        plt.show()
        print(mask.shape)
        break
