from dataset import CityScape
from model import *
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
from utils import transform, add_result, save_checkpoint
import os


def train(checkpoint):
    if checkpoint == None:
        model = UNET(in_c=3, out_c=8)
        start_epoch = 0
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']

    model = model.to(device=device)
    criterion = SegmentationLoss(num_classes=num_classes, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f" -- Initiating the Training Process -- ")
    print(f"Epoch: {start_epoch}: ")

    for epoch in range(start_epoch, epochs + 1):
        average_loss = 0
        model.train()
        for i, (image, mask) in enumerate(train_gen):
            image = image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            pred_mask = model(image)
            print(pred_mask.shape)
            loss = criterion(pred_mask, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            average_loss = average_loss + loss.item()
            torch.cuda.empty_cache()

            if i%500 == 0:
                print("=", end="")

        model.eval()
        validation_loss = 0
        for j, (image, mask) in enumerate(val_gen):
            image = image.to(device)
            mask = mask.to(device)
            pred_mask = model(image)
            loss = criterion(pred_mask, mask)

            validation_loss = validation_loss + loss.item()
            torch.cuda.empty_cache()

        save_checkpoint(epoch=epoch, model=model, version=version)
        add_result(f"Epoch: {epoch} | Average Loss: {average_loss/(i + 1)} | Val Loss: {validation_loss/(j + 1)}", version)
        print(f"   Epoch: {epoch} | Average Loss: {average_loss/(i + 1)} | Val Loss: {validation_loss/(j + 1)}")


if __name__ == "__main__":
    version = 1.1
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if f"Segmentation_v{version}.pth.tar" in os.listdir():
        checkpoint = f"Segmentation_v{version}.pth.tar"
    else:
        checkpoint = None
    num_classes = 8
    batch_size = 2
    workers = 8
    epochs = 3000
    lr = 1e-5
    train_file = "train.json"
    val_file = "val.json"
    size = (512, 256) # (256, 512)

    train_dataset = CityScape(train_file, size, transform)
    val_dataset = CityScape(val_file, size, transform)

    train_gen = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    val_gen = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=val_dataset.collate_fn
    )

    train(checkpoint)
