import os
import json
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from urllib.request import urlopen


def create_json_dataset(image_dir, mask_dir):
    for folder_name in os.listdir(image_dir):
        image_path = image_dir + "/" + folder_name
        mask_path = mask_dir + "/" + folder_name
        data = []
        for sub_folder in os.listdir(image_path):
            for image in os.listdir(image_path + "/" + sub_folder):
                image_name = image_path + "/" + sub_folder + "/" + image
                mask_name = mask_path + "/" + sub_folder + "/" + image[:-15] + "depth.png"
                data.append([image_name, mask_name])

        with open(f'{folder_name}.json', "w", encoding='utf-8') as f:
            json.dump(data, f)

def transform(image, mask):
    hflip = transforms.RandomHorizontalFlip(p=1)
    vflip = transforms.RandomVerticalFlip(p=1)
    totensor = transforms.PILToTensor()

    if random.random() > 0.5:
        image = hflip(image)
        mask = hflip(mask)

    # #Vertical Flipping
    if random.random() > 0.5:
        image = vflip(image)
        mask = vflip(mask)

    image = totensor(image)
    mask = totensor(mask)

    return image, mask

def transform_pil_to_image(image, mask):
    totensor = transforms.PILToTensor()
    return totensor(image), totensor(mask)

def add_result(result, version):
    with open(f'results_v{version}.txt', 'a') as f:
        f.write(result + "\n")
    f.close()

def save_checkpoint(epoch, model, version):
    state = {'epoch': epoch,
             'model': model}
    filename = f'Depth_v{version}.pth.tar'
    torch.save(state, filename)

def draw_loss_graph(file_name, save_name, from_epoch = 0):
    if file_name.startswith("http"):
        f = urlopen(file_name).read().decode('utf-8').split("\n")[:-1]
    else:
        f = open(file_name, 'r')
        f = f.readlines()
    x = []
    y = []
    y_val = []
    for i in f[from_epoch:]:
        temp = i.split(" ")
        x.append(int(temp[1]))
        y.append(float(temp[5]))
        y_val.append(float(temp[-1].replace("\n", "")))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x, y, color='c', label='Train Loss')
    plt.plot(x, y_val, color='orange',label='Val Loss')
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.legend()
    plt.savefig(save_name)
    plt.close()

# draw_loss_graph('results_v6.txt', "loss.png")