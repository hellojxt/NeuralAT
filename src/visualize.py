import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go


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
    )
    colorbar_xs = [0.28, 0.63, 0.99]
    colorbar_dy = 1.25 / rows
    colorbar_ys = [0.18 + i * colorbar_dy for i in range(rows)]

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
    return fig


def plot_point_cloud(vertices, triangles, coords, data=None):
    """
    coords: (N, 3)
    data: (N)
    """
    if data is None:
        data = np.ones(len(coords))
    else:
        data = data.reshape(-1)

    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=data,
            colorscale="Viridis",  # choose a colorscale
            opacity=1.0,
            cmin=-1,
            cmax=1,
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
        intensity=np.zeros(len(vertices)),
        colorscale="Viridis",
        opacity=1,
        cmin=-1,
        cmax=1,
        # lighting=dict(ambient=1.0, diffuse=1.0, specular=1.0, fresnel=0.0),  # 关闭光照效果
    )
    return go.Figure(data=[mesh, scatter])
