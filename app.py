import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
import time

# Import custom functions
from functions import read_pbm, random_flip, corrupt_crop

# Function to create a plotly figure
def create_fig(data):
    data = np.flipud(data)  # Flip vertically to fix orientation
    fig = go.Figure(data=go.Heatmap(
        z=data.T,
        colorscale=[[0, 'black'], [1, 'white']],
        xgap=1, ygap=1,
        showscale=False
    ))
    fig.update_layout(
        width=150, height=150,
        xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# Initialize session state for selected images and weights
if 'selected_images' not in st.session_state:
    st.session_state.selected_images = []
if 'weights' not in st.session_state:
    st.session_state.weights = None

# Title and sidebar
st.title("Hopfield Network App")

##Sidebar crap
st.sidebar.title("📌 Hopfield Network Overview")

st.sidebar.markdown("""
### 🧠 What is a Hopfield Network?
A Hopfield network is a type of recurrent neural network used for associative memory. It can store patterns and retrieve them even when the input is noisy or incomplete.

### 🌟 How It Works:
1. **Initialization:** Start with a set of binary patterns to store.
2. **Training:** Use the Hebbian rule to compute the weight matrix.
3. **Retrieval:** Update the network state iteratively until convergence.
""")

# Use raw string (r"") to avoid escape issues
st.sidebar.markdown("### 📏 Hebbian Rule:")
st.sidebar.latex(r"W_{ij} = \frac{1}{N} \sum_{\mu=1}^{P} \xi_i^\mu \xi_j^\mu")

st.sidebar.markdown(r"""
where:  
- \( N \) = number of neurons  
- \( P \) = number of patterns  
- \( \xi_i^\mu \) = state of neuron \( i \) in pattern \( \mu \)  

### ✅ Algorithm Steps:
1. Load the input pattern.  
2. Apply Hebbian learning to compute the weight matrix.  
3. Use asynchronous or synchronous updates to converge to a stable state.  
4. Compare final state to the stored pattern.  
""")


# Load PBM files
pbm_files = [f for f in os.listdir("pbm_files") if f.endswith(".pbm")]
train_images = [read_pbm(os.path.join("pbm_files", file)) for file in pbm_files]

# Ensure selected_images state matches number of files
while len(st.session_state.selected_images) < len(train_images):
    st.session_state.selected_images.append(False)

# 🏋️‍♂️ Training Setup
with st.container(border=True):
    st.markdown("## 🏋️‍♂️ Training Setup")

    cols = st.columns(len(train_images))
    for i, col in enumerate(cols):
        with col:
            st.plotly_chart(create_fig(train_images[i]), use_container_width=True, key=f"train_image_{i}")
            st.session_state.selected_images[i] = st.checkbox(f"Image {i + 1}", value=st.session_state.selected_images[i])

    if st.button("🚀 Start Training"):
        selected = [i for i in range(len(train_images)) if st.session_state.selected_images[i]]
        if selected:
            st.success(f"Training started with {len(selected)} selected images!")
            weights = np.zeros((256, 256))
            for idx in selected:
                img_vector = train_images[idx].flatten() * 2 - 1
                weights += np.outer(img_vector, img_vector)
            np.fill_diagonal(weights, 0)
            weights = weights / len(selected)
            st.session_state.weights = weights
            st.write("✅ Training Complete!")
        else:
            st.error("Please select at least one image to train.")

# 📂 Load and Display File
with st.container(border=True):
    st.markdown("## 📂 Load and Display File")

    selected_file = st.selectbox(
        "Select File",
        options=[f"Image {i + 1}" for i in range(len(train_images))],
        index=0,
        key="file_selection"
    )
    selected_index = int(selected_file.split(" ")[1]) - 1

    st.plotly_chart(create_fig(train_images[selected_index]), use_container_width=True, key=f"test_plot_{selected_index}")

    corruption_type = st.selectbox(
        "Choose corruption method",
        ["None", "Random Flip", "Crop"],
        index=0,
        key="corruption_keys"
    )

    if corruption_type == "Random Flip":
        flip_prob = st.slider("Flip Probability", 0.0, 1.0, 0.1)
    elif corruption_type == "Crop":
        crop_size = st.slider("Crop Size", 1, 16, 8)

    if st.button("Apply Corruption"):
        corrupted_image = train_images[selected_index].copy()
        if corruption_type == "Random Flip":
            corrupted_image = random_flip(corrupted_image, flip_prob)
            st.success(f"Applied Random Flip with p={flip_prob:.2f}")
        elif corruption_type == "Crop":
            corrupted_image = corrupt_crop(corrupted_image, crop_size)
            st.success(f"Applied Crop with box size={crop_size}")
        st.session_state["corrupted_image"] = corrupted_image

        # Display corrupted image
        st.empty().plotly_chart(create_fig(st.session_state["corrupted_image"]), use_container_width=True)

# 🧠 Run Hopfield Network
def async_update(image, weights, iterations=1000):
    theta = 0
    state = image.flatten() * 2 - 1

    # Use st.empty() to manage dynamic updates
    plot_container = st.empty()

    for its in range(iterations):
        i = np.random.randint(256)
        sum_input = np.dot(weights[i], state)
        state[i] = 1 if sum_input >= theta else -1

        if its % 20 == 0:  # Only update plot every 20 iterations to reduce flickering
            fig = create_fig((state.reshape(16, 16) + 1) // 2)
            plot_container.plotly_chart(fig, use_container_width=True, key=f"update_plot_{its}")

        time.sleep(0.005)  # Small delay for smoother updates

    return (state.reshape(16, 16) + 1) // 2, its

with st.container(border=True):
    st.markdown("## 🧠 Run Hopfield Network")

    if st.button("Run Hopfield Network", key="run_network"):
        st.info("Running the algorithm...")

        input_image = st.session_state.get("corrupted_image", train_images[selected_index])
        weights = st.session_state.get("weights")

        if weights is not None:
            final_state, num_iters = async_update(input_image, weights)

            # Display final state
            st.plotly_chart(create_fig(final_state), use_container_width=True, key="final_plot")

            st.success(f"Converged in {num_iters} iterations!")
        else:
            st.error("Train the network first.")

