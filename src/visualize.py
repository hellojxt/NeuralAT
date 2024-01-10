import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import torch

from PIL import Image


def torch_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return np.array(tensor)
    else:
        return tensor


def combine_images(image_paths, output_path):
    """
    Combines multiple images into a single image.

    :param image_paths: A 2D list of image paths, where each row contains paths for a row of images.
    :param output_path: Path to save the combined image.
    """
    # Determine the number of rows and columns
    num_rows = len(image_paths)
    num_cols = len(image_paths[0])

    # Open the first image to get the dimensions
    with Image.open(image_paths[0][0]) as img:
        width, height = img.size

    # Create a new image with the combined dimensions
    combined_img = Image.new("RGB", (num_cols * width, num_rows * height))

    # Place each image in the correct position
    for row in range(num_rows):
        for col in range(num_cols):
            with Image.open(image_paths[row][col]) as img:
                combined_img.paste(img, (col * width, row * height))

    # Save the combined image
    combined_img.save(output_path)
    print(f"Combined image saved to {output_path}")


def crop_center(image_path, crop_width, crop_height):
    """
    Crops the image at the center with the given width and height.

    :param image_path: Path to the input image.
    :param crop_width: Width of the crop area.
    :param crop_height: Height of the crop area.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        # Calculate the top, left, right, and bottom coordinates for the crop
        left = (width - crop_width) / 2
        top = (height - crop_height) / 2
        right = (width + crop_width) / 2
        bottom = (height + crop_height) / 2

        # Perform the crop
        cropped_img = img.crop((left, top, right, bottom))

        # Save the cropped image
        cropped_img.save(image_path)


def plot_mesh(
    vertices,
    triangles,
    data=None,
    names=None,
    cmin=None,
    cmax=None,
):
    """
    vertices: (N, 3)
    triangles: (M, 3)
    data: (N, K) or (M, K)
    names: (K)
    """
    if data is None:
        data = np.zeros((vertices.shape[0], 1))
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if names is None:
        names = [str(i) for i in range(data.shape[1])]

    vertices, triangles, data = [torch_to_numpy(x) for x in [vertices, triangles, data]]
    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    fig = go.Figure()
    # Add traces, one for each slider step
    for i in range(data.shape[1]):
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                colorscale="Viridis",
                intensity=data[:, i],
                intensitymode="cell"
                if data.shape[0] == triangles.shape[0]
                else "vertex",
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                name=names[i],
                visible=False,
            )
        )

    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": "Switched to feature: " + str(i)},
            ],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(
            active=10, currentvalue={"prefix": "Feature: "}, pad={"t": 50}, steps=steps
        )
    ]
    fig.update_layout(sliders=sliders)

    return fig


def plot_point_cloud(
    vertices,
    triangles,
    coords,
    data=None,
    point_size=5,
    mesh_opacity=0.2,
    cmin=None,
    cmax=None,
    zoom=1.0,
    background_grid=True,
):
    """
    coords: (N, 3)
    data: (N, K)
    """
    if data is None:
        data = np.ones(len(coords)).reshape(-1, 1)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    vertices, triangles, coords, data = [
        torch_to_numpy(x) for x in [vertices, triangles, coords, data]
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=mesh_opacity,
            cmin=cmin,
            cmax=cmax,
            visible=True,  # Mesh is always visible
            name="",  # Don't show legend for mesh
        )
    )

    # Add traces, one for each slider step
    for i in range(data.shape[1]):
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=data[:, i],
                    colorscale="Viridis",  # choose a colorscale
                    opacity=1.0,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title=""),
                ),
                visible=False,
                name="",  # Don't show legend for mesh
            )
        )

    fig.data[1].visible = True

    # Create and add slider
    steps = []
    for i in range(1, len(fig.data)):  # Skip the first trace (mesh)
        step = dict(
            method="update",
            args=[
                {
                    "visible": [True] + [False] * (len(fig.data) - 1)
                },  # Keep mesh always visible
                {"title": "Switched to feature: " + str(i)},
            ],
        )
        step["args"][0]["visible"][i] = True  # Make corresponding plane trace visible
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "Feature: "}, pad={"t": 50}, steps=steps)
    ]
    fig.update_layout(sliders=sliders)
    if not background_grid:
        fig.update_layout(
            {
                "scene": {
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "zaxis": {"visible": False},
                },
                "scene_aspectmode": "data",
            }
        )
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=0, y=0.5 * zoom, z=1.5 * zoom), up=dict(x=0, y=1, z=0)
        )
    )
    return fig


def plot_mesh_with_plane(
    vertices,
    triangles,
    mesh_data,
    xs,
    ys,
    zs,
    plane_data,
    min_bound=None,
    max_bound=None,
    use_marker=False,
    mesh_opacity=0.2,
    cmin=None,
    cmax=None,
    log_color=False,
):
    """
    vertices: (N, 3) array for mesh vertices
    triangles: (M, 3) array for mesh triangles
    mesh_data: (M) array for mesh color data
    xs, ys, zs: (H, W, K) arrays for plane coordinates
    plane_data: (H, W, K) array for plane color data
    """
    vertices, triangles, mesh_data, xs, ys, zs, plane_data, min_bound, max_bound = [
        torch_to_numpy(x)
        for x in [
            vertices,
            triangles,
            mesh_data,
            xs,
            ys,
            zs,
            plane_data,
            min_bound,
            max_bound,
        ]
    ]
    if log_color:
        plane_data = np.log(plane_data + 1e-6)
        mesh_data = np.log(mesh_data + 1e-6)

    if cmin is None or cmax is None:
        cmin = plane_data.min()
        cmax = plane_data.max()

    mesh_data = (mesh_data - np.min(mesh_data)) / (
        np.max(mesh_data) - np.min(mesh_data) + 1e-6
    ) * (cmax - cmin) + cmin
    print("cmin = ", cmin, "cmax = ", cmax)
    # Create figure
    fig = go.Figure()

    # Add mesh trace
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=mesh_opacity,
            cmin=cmin,
            cmax=cmax,
            visible=True,  # Mesh is always visible
            name="",  # Don't show legend for mesh
        )
    )

    if not (min_bound is None and max_bound is None):
        # plot bbox
        bbox_corners = np.array(
            [
                [min_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
            ]
        )
        # Add lines to connect the corners
        fig.add_trace(
            go.Scatter3d(
                x=bbox_corners[:, 0],
                y=bbox_corners[:, 1],
                z=bbox_corners[:, 2],
                mode="markers",
                marker=dict(size=2),
                opacity=0.3,
                showlegend=False,
            )
        )

    for i in range(plane_data.shape[2]):
        x = xs[:, :, i]
        y = ys[:, :, i]
        z = zs[:, :, i]
        if use_marker:
            fig.add_trace(
                go.Scatter3d(
                    x=x.reshape(-1),
                    y=y.reshape(-1),
                    z=z.reshape(-1),
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=plane_data[:, :, i].reshape(-1),
                        colorscale="Viridis",  # choose a colorscale
                        opacity=1.0,
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=dict(title=""),
                    ),
                    visible=False,
                    name="",  # Don't show legend for mesh
                )
            )

        else:
            fig.add_trace(
                go.Surface(
                    visible=False,
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=plane_data[:, :, i],
                    cmin=cmin,
                    cmax=cmax,
                    colorscale="Viridis",
                    colorbar=dict(title=""),
                    showscale=False,  # Hide color scale for the plane
                    lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0),
                    name="",  # Don't show legend for mesh
                )
            )

    fig.data[2].visible = True  # Make plane trace visible

    # Create and add slider
    steps = []
    for i in range(2, len(fig.data)):  # Skip the first trace (mesh)
        step = dict(
            method="update",
            args=[
                {
                    "visible": [True, True] + [False] * (len(fig.data) - 2)
                },  # Keep mesh always visible
                {"title": "Slice %d" % i},
            ],
        )
        step["args"][0]["visible"][i] = True  # Make corresponding plane trace visible
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Slice: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(
        {
            # "scene": {
            #     "xaxis": {"visible": False},
            #     "yaxis": {"visible": False},
            #     "zaxis": {"visible": False},
            # },
            "scene_aspectmode": "data",
        }
    )
    fig.update_layout(sliders=sliders)

    return fig


class CombinedFig:
    def __init__(self):
        self.fig = go.Figure()

    def add_mesh(
        self, vertices, triangles, data=None, opacity=0.2, cmax=None, cmin=None
    ):
        vertices, triangles, data = [
            torch_to_numpy(x) for x in [vertices, triangles, data]
        ]
        # Add traces, one for each slider step
        if data is None:
            self.fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    opacity=opacity,
                    visible=True,  # Mesh is always visible
                    name="",  # Don't show legend for mesh
                    showlegend=False,
                    showscale=False,
                )
            )
        else:
            if cmax is None or cmin is None:
                cmax = data.max()
                cmin = data.min()
            print("cmin = ", cmin, "cmax = ", cmax)
            self.fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    colorscale="Viridis",
                    intensity=data,
                    intensitymode="cell"
                    if data.shape[0] == triangles.shape[0]
                    else "vertex",
                    name="",
                    opacity=opacity,
                )
            )
        return self

    def add_points(
        self, coords, data=None, point_size=5, showscale=True, cmax=None, cmin=None
    ):
        coords, data = [torch_to_numpy(x) for x in [coords, data]]
        if data is None:
            data = np.ones(len(coords))
        else:
            data = data.reshape(-1)

        coords = coords.reshape(-1, 3)
        if cmax is None or cmin is None:
            cmax = data.max()
            cmin = data.min()
        print("cmin = ", cmin, "cmax = ", cmax)
        self.fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=data,
                    colorscale="Viridis",  # choose a colorscale
                    opacity=1.0,
                    cmax=cmax,
                    cmin=cmin,
                    colorbar=dict(title="") if showscale else None,
                ),
                name="",
            )
        )
        return self

    def show(self, grid=True):
        self.fig.update_layout(
            {
                "scene": {
                    "xaxis": {"visible": grid},
                    "yaxis": {"visible": grid},
                    "zaxis": {"visible": grid},
                },
                "scene_aspectmode": "data",
            }
        )
        self.fig.show()
