import streamlit as st
import numpy as np
import plotly.graph_objects as go

def create_fig(arr):
    arr = np.flipud(arr)
    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=arr,
        colorscale=[[0, 'black'], [1, 'white']],  # 1 = white, 0 = black
        xgap=1,  # Adds gap between pixels
        ygap=1,
        showscale=False  # Hide color bar
    ))

    # Set the size and remove axis labels
    fig.update_layout(
        width=400, height=400,  # Size of the plot
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0)  # Remove plot margins
    )

    return fig