import numpy as np
import os
#from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import time

def read_pbm(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:

        # Read the header
        magic_number = f.readline().strip()
        if magic_number != 'P1':
            raise ValueError("File is not a valid ASCII PBM (P1) file.")
        
        # Skip comments
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        
        # Read dimensions
        width, height = map(int, line.split())
        
        # Read pixel data
        pixels = []
        for line in f:
            pixels.extend(line.split())
        
    # Convert to numpy array and reshape
    image = np.array(pixels, dtype=int).reshape((height, width))
    return image

def write_pbm(image, filename):
    with open(filename, 'w') as f:
        f.write("P1\n")
        f.write("16 16\n")
        for row in image:
            for el in row:
                if el == 0:
                    f.write("0 ")
                else:
                    f.write("1 ")
            f.write("\n")

def display_image(image): #function to display image
    # Reshape to 16x16 for visualization
    image = image.reshape(16, 16)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()

# ✅ Corruption functions
def random_flip(image, p):
    corrupted = image.copy()
    flip_bool = np.random.rand(16, 16) < p
    corrupted[flip_bool] = 1 - corrupted[flip_bool]
    return corrupted

def corrupt_crop(image, box_size):
    corrupted = np.zeros((16, 16))
    start = (16 - box_size) // 2
    corrupted[start:start + box_size, start:start + box_size] = image[start:start + box_size, start:start + box_size]
    return corrupted