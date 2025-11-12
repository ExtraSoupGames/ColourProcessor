from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import colorsys
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

        #Scale down for speed
        if resize_factor < 1.0:
            img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)),Image.Resampling.LANCZOS)
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

def sort_palette_by_hue(colors):
    # Convert to HSV
    hsv_colors = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r,g,b in colors]
    # Sort by HSV
    sorted_colors = [color for _, color in sorted(zip(hsv_colors, colors))]
    return np.array(sorted_colors)

def save_palette_as_png(colors, filename="palette.png", swatch_size=50):
    #Save the output palette
    num_colors = len(colors)
    palette_img = np.zeros((swatch_size, swatch_size * num_colors, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        palette_img[:, i*swatch_size:(i+1)*swatch_size, :] = color

    img = Image.fromarray(palette_img)
    img.save(filename)
    print(f"Palette saved as {filename}")


def show_palette_interactive(colors, folder_path, swatch_size=50):
    #Display the palette and allow the user to trim unneeded colours
    colors = np.array(colors)
    enabled = [True] * len(colors)

    fig, ax = plt.subplots(figsize=(len(colors), 2))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis('off')

    bars = []
    for i, c in enumerate(colors):
        bar = ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c/255))
        bars.append(bar)

    def onclick(event):
        x = int(event.xdata)
        if 0 <= x < len(colors):
            enabled[x] = not enabled[x]
            bars[x].set_alpha(1.0 if enabled[x] else 0.2)
            fig.canvas.draw()

    def onkey(event):
        if event.key == 's':  # Press 's' to save filtered palette
            filtered = colors[enabled]
            save_palette_as_png(filtered, filename=os.path.join(folder_path, "trimmed_palette.png"))
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    return colors[enabled]




if __name__ == "__main__":
    #Get list of files inside Images folder
    folder_path = "ColourProcessor\\Images"
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_names_full = [os.path.join(folder_path, f) for f in file_names]

    #Extract colours
    num_colors = 64
    palette = extract_palette(file_names_full, num_colors=num_colors, resize_factor=0.1)
    palette = sort_palette_by_hue(palette)
    print(palette)
    palette = show_palette_interactive(palette, folder_path)
    #Save an extra time
    save_palette_as_png(palette)
