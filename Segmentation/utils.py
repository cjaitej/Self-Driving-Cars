import os
import json
import random
from torchvision import transforms
import torch

def create_json_dataset(image_dir, mask_dir):
    for folder_name in os.listdir(image_dir):
        image_path = image_dir + "/" + folder_name
        mask_path = mask_dir + "/" + folder_name
        data = []
        for sub_folder in os.listdir(image_path):
            for image in os.listdir(image_path + "/" + sub_folder):
                image_name = image_path + "/" + sub_folder + "/" + image
                mask_name = mask_path + "/" + sub_folder + "/" + image[:-15] + "gtFine_labelIds.png"
                data.append([image_name, mask_name])

        with open(f'{folder_name}.json', "w", encoding='utf-8') as f:
            json.dump(data, f)


def transform(image, mask):
    hflip = transforms.RandomHorizontalFlip(p=1)
    vflip = transforms.RandomVerticalFlip(p=1)
    totensor = transforms.PILToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    if random.random() > 0.5:
        image = hflip(image)
        mask = hflip(mask)

    # #Vertical Flipping
    if random.random() > 0.5:
        image = vflip(image)
        mask = vflip(mask)

    image = totensor(image)
    mask = totensor(mask)

    # image = normalize(image/1.)

    return image, mask

def add_result(result, version):
    with open(f'results_v{version}.txt', 'a') as f:
        f.write(result + "\n")
    f.close()

def save_checkpoint(epoch, model, version):
    state = {'epoch': epoch,
             'model': model}
    filename = f'Segmentation_v{version}.pth.tar'
    torch.save(state, filename)