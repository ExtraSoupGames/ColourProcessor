from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
def load_images(image_paths):
    #Loads all images from the list of paths
    pixels = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        pixels.append(np.array(img).reshape(-1, 3))
    return np.vstack(pixels)

def extract_palette(image_paths, num_colors=16, resize_factor=0.5):
    #Extract the palette from the images
    all_pixels = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        all_pixels.append(np.array(img).reshape(-1,3))
    
    all_pixels = np.vstack(all_pixels)
    kmeans = KMeans(n_clusters=num_colors, random_state=1)
    kmeans.fit(all_pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def show_palette(colors):
    #Display the extracted palette using pyplot
    plt.figure(figsize=(8,2))
    plt.axis("off")
    plt.imshow([colors])
    plt.show()

if __name__ == "__main__":
    #Get list of files inside Images folder
    folder_path = "ColourProcessor\\Images"
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_names_full = [os.path.join(folder_path, f) for f in file_names]

    #Extract colours
    num_colors = 32
    palette = extract_palette(file_names_full, num_colors=num_colors)
    print("Extracted Colors (RGB):")
    print(palette)

    #Display palette
    show_palette(palette)
