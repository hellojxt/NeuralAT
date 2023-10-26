import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plot_mesh(vertices, triangles, data=None, cmin=None, cmax=None):
    """
    vertices: (N, 3)
    triangles: (M, 3)
    data: (N, M)
    """
    if data is None:
        data = np.zeros((vertices.shape[0], 1))
    # if cmin is None:
    #     cmin = np.min(data) * 1.2
    # if cmax is None:
    #     cmax = np.max(data) / 1.2
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for i in range(data.shape[1]):
        color_data = data[:, i]
        fig.add_trace(
            go.Mesh3d(
                visible=False,
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                intensity=color_data,
                colorscale="Viridis",
                opacity=1,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(thickness=20),
            )
        )

    # Make 10th trace visible
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
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [
        dict(
            active=10, currentvalue={"prefix": "Feature: "}, pad={"t": 50}, steps=steps
        )
    ]

    fig.update_layout(sliders=sliders)

    return fig


def plot_point_cloud(coords, data=None, cmin=None, cmax=None):
    """
    coords: (N, 3)
    data: (N, M)
    """
    if data is None:
        data = np.zeros((coords.shape[0], 1))
    # if cmin is None:
    #     cmin = np.min(data) * 1.2
    # if cmax is None:
    #     cmax = np.max(data) / 1.2
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for i in range(data.shape[1]):
        color_data = data[:, i]
        fig.add_trace(
            go.Scatter3d(
                visible=False,
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    # set color to an array/list of desired values
                    color=color_data,
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(thickness=20),
                ),
            )
        )

    # Make 10th trace visible
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
