import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import torch


def plot_mesh(
    vertices, triangles, data=None, names=None, cmin=None, cmax=None, back_ground=False
):
    """
    vertices: (N, 3)
    triangles: (M, 3)
    data: (N, K)
    names: (K, )
    """
    if data is None:
        data = np.zeros((vertices.shape[0], 1))
    if names is None:
        names = [str(i) for i in range(data.shape[1])]

    rows = data.shape[1] // 3  # Assuming you want 3 columns
    cols = 3
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=names,
        start_cell="top-left",
        specs=[[{"type": "mesh3d"}] * cols for _ in range(rows)],
        horizontal_spacing=0.0,
        vertical_spacing=0.0,
    )
    colorbar_xs = [0.33, 0.66, 0.99]
    colorbar_dy = 1.0 / rows
    colorbar_ys = [0.75 - i * colorbar_dy for i in range(rows)]

    for i in range(data.shape[1]):
        color_data = data[:, i]
        row = (i // cols) + 1
        col = (i % cols) + 1
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        mesh = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=color_data,
            colorscale="Viridis",
            opacity=1,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                thickness=20,
                len=0.8 / rows,  # 设置colorbar的长度
                x=colorbar_xs[col - 1],  # 设置colorbar的位置
                y=colorbar_ys[row - 1],  # 设置colorbar的位置
            ),
            lighting=dict(
                ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0
            ),  # 关闭光照效果
        )

        fig.add_trace(mesh, row=row, col=col)

    for i in range(data.shape[1]):
        if i == 0:
            subname = ""
        else:
            subname = str(i + 1)

        fig.update_layout(
            {
                "scene"
                + subname: {
                    "xaxis": {"visible": back_ground},
                    "yaxis": {"visible": back_ground},
                    "zaxis": {"visible": back_ground},
                },
                "scene" + subname + "_aspectmode": "data",
            }
        )
    fig.update_layout(font_family="Linux Biolinum", font_size=20)
    fig.update_annotations(font_size=30)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig


def plot_point_cloud(
    vertices,
    triangles,
    coords,
    data=None,
    point_size=2,
    mesh_opacity=0.2,
    cmin=None,
    cmax=None,
):
    """
    coords: (N, 3)
    data: (N)
    """
    if data is None:
        data = np.ones(len(coords))
    else:
        data = data.reshape(-1)
    vertices, triangles, coords, data = [
        torch_to_numpy(x) for x in [vertices, triangles, coords, data]
    ]
    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color=data,
            colorscale="Viridis",  # choose a colorscale
            opacity=1.0,
            cmin=cmin,
            cmax=cmax,
        ),
    )

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=-np.zeros(len(vertices)),
        colorscale="Viridis",
        opacity=mesh_opacity,
        cmin=-1,
        cmax=1,
        # lighting=dict(ambient=1.0, diffuse=1.0, specular=1.0, fresnel=0.0),  # 关闭光照效果
    )
    fig = go.Figure(data=[mesh, scatter])
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
        scene_camera=dict(eye=dict(x=0, y=0.5, z=1.5), up=dict(x=0, y=1, z=0))
    )
    return fig


def torch_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


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
    color_data = mesh_data
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=color_data,
            colorscale="Viridis",
            opacity=mesh_opacity,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(thickness=20),
            visible=True,  # Mesh is always visible
            lighting=dict(
                ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0
            ),  # 关闭光照效果
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
