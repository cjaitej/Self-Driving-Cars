from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

import torch
import numpy as np
from utils import transform_pil_to_image

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def surface():
    def create_rgb_surface(rgb_img, depth_img, **kwargs):
        rgb_img = rgb_img.swapaxes(0, 1)[:, ::-1]
        depth_img = depth_img.swapaxes(0, 1)[:, ::-1]
        eight_bit_img = Image.fromarray(rgb_img).convert('P', palette='ADAPTIVE', dither=None)
        idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
        colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
        depth_map = depth_img.copy().astype('float')

        return go.Surface(
            z=depth_map,
            surfacecolor=np.array(eight_bit_img),
            cmin=0,
            cmax=255,
            colorscale=colorscale,
            showscale=False,
            **kwargs
        )



    d = np.flipud(depth)
    img = np.flipud(image)


    fig = go.Figure(
        data=[create_rgb_surface(img,
                                d,
                                contours_z=dict(show=True, project_z=True, highlightcolor="white"),
                                opacity=1.0,
                                ),
                                ],
        layout_title_text="3D Surface",
    )
    camera = dict(
        eye=dict(x=0.001, y=0, z=2)
    )

    fig.update_layout(scene_camera=camera)

    fig.update_layout(
        scene = dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
            ),
        )
    fig.show()

def scatter():
    x = []
    y = []
    z = []
    colors = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x.append(i)
            y.append(j)
            z.append(depth[i][j])
            colors.append(f"rgb{tuple(image[i][j])}")

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
        ))])
    camera = dict(
        eye=dict(x=0.001, y=0, z=2)
    )

    fig.update_layout(scene_camera=camera)
    fig.show()

if __name__ == "__main__":
    x = "D:/Major_Project_Initial/left/leftImg8bit/train/monchengladbach/monchengladbach_000000_017950_leftImg8bit.png"
    y = "darmstadt/darmstadt_000002_000019_depth.png"

    image_path, depth_path = [f"{x}", f"test.png"]
    size = (512, 256)

    image = Image.open(image_path).convert("RGB").resize(size)
    depth = Image.open(depth_path).convert('L').resize(size, Image.Resampling.NEAREST)


    image, depth = transform_pil_to_image(image, depth)
    depth = inverse_sigmoid(depth).squeeze(0).numpy()*100.0
    image = image.permute(1, 2, 0).numpy()

    surface()