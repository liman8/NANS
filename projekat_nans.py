import tkinter as tk
from tkinter import ttk, filedialog
from scipy.interpolate import RectBivariateSpline
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
import numpy as np

# -------------Bilinear interpolation----------------
def bilinear_interpolation_algorithm(image, new_height, new_width):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    scale_y = new_height / height
    scale_x = new_width / width
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            y = i / scale_y
            x = j / scale_x
            y_floor, x_floor = int(np.floor(y)), int(np.floor(x))
            y_ceil, x_ceil = min(y_floor + 1, height - 1), min(x_floor + 1, width - 1)

            top_left = img_array[y_floor, x_floor] * ((y_ceil - y) * (x_ceil - x))
            top_right = img_array[y_floor, x_ceil] * ((y_ceil - y) * (x - x_floor))
            bottom_left = img_array[y_ceil, x_floor] * ((y - y_floor) * (x_ceil - x))
            bottom_right = img_array[y_ceil, x_ceil] * ((y - y_floor) * (x - x_floor))

            new_img_array[i, j] = top_left + top_right + bottom_left + bottom_right

    new_image = Image.fromarray(new_img_array)
    return new_image

# ---------------Cubic interpolation for Bicubic interpolation----------------
def cubic_interp(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))

# ---------------Bicubic interpolation-----------------
def bicubic_interpolation_algorithm(image, new_height, new_width):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    scale_y = new_height / height
    scale_x = new_width / width
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.float32)

    for i in range(new_height):
        for j in range(new_width):
            y = i / scale_y
            x = j / scale_x
            y_vals = np.array([y + 1, y, y - 1, y - 2]) % height
            x_vals = np.array([x + 1, x, x - 1, x - 2]) % width
            cubic_weights_y = cubic_interp(y_vals - y)
            cubic_weights_x = cubic_interp(x_vals - x)

            for channel in range(3):
                patch = img_array[y_vals.astype(int), :, channel]
                patch = patch[:, x_vals.astype(int)]
                new_img_array[i, j, channel] = np.clip(np.sum(cubic_weights_y * patch * cubic_weights_x.T), 0, 255)

    new_image = Image.fromarray(new_img_array.astype(np.uint8))
    return new_image

# --------------- Nearest-neighbor interpolation -----------------
def nearest_neighbor_interpolation_algorithm(image, new_height, new_width):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    scale_y = height / new_height
    scale_x = width / new_width
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            y = int(i * scale_y)
            x = int(j * scale_x)
            new_img_array[i, j] = img_array[y, x]

    new_image = Image.fromarray(new_img_array)
    return new_image

# --------------- B-spline interpolation -----------------
def bspline_interpolation_algorithm(image, new_height, new_width):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    x_vals = np.arange(0, width, 1)
    y_vals = np.arange(0, height, 1)

    bspline_r = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 0])
    bspline_g = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 1])
    bspline_b = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 2])

    new_y_vals = np.linspace(0, height - 1, new_height)
    new_x_vals = np.linspace(0, width - 1, new_width)
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            new_img_array[i, j, 0] = np.clip(int(bspline_r(new_y_vals[i], new_x_vals[j])[0, 0]), 0, 255)
            new_img_array[i, j, 1] = np.clip(int(bspline_g(new_y_vals[i], new_x_vals[j])[0, 0]), 0, 255)
            new_img_array[i, j, 2] = np.clip(int(bspline_b(new_y_vals[i], new_x_vals[j])[0, 0]), 0, 255)

    new_image = Image.fromarray(new_img_array)
    return new_image

# --------------- Lanczos interpolation -----------------
def lanczos_interpolation_algorithm(image, new_height, new_width):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Get the dimensions of the original image
    height, width, _ = img_array.shape

    # Calculate the scaling factors
    scale_y = height / new_height
    scale_x = width / new_width

    # Initialize the new image array
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Create a RectBivariateSpline for each color channel
    x_vals = np.arange(0, width, 1)
    y_vals = np.arange(0, height, 1)
    interp_r = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 0])
    interp_g = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 1])
    interp_b = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 2])

    # Perform Lanczos interpolation
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the coordinates in the original image
            y = i * scale_y
            x = j * scale_x

            # Evaluate Lanczos interpolation for each color channel
            new_img_array[i, j, 0] = np.clip(int(interp_r(y, x)[0, 0]), 0, 255)
            new_img_array[i, j, 1] = np.clip(int(interp_g(y, x)[0, 0]), 0, 255)
            new_img_array[i, j, 2] = np.clip(int(interp_b(y, x)[0, 0]), 0, 255)

    # Convert the NumPy array back to an image
    new_image = Image.fromarray(new_img_array)

    return new_image


class ImageInterpolatorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Mini image interpolation tool")
        self.style = ThemedStyle(master)
        self.style.set_theme("radiance")

        # Variables
        self.input_image = None
        self.output_image = None
        self.previous_image = None
        self.new_height_var = tk.IntVar()
        self.new_width_var = tk.IntVar()
        self.interpolation_method_var = tk.StringVar()
        self.interpolation_methods = ["Bilinear", "Bicubic", "Nearest Neighbor", "B-spline", "Lanczos"]

        # Widgets
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = ttk.Label(self.master, text="Mini image interpolation tool", font=('Baskerville', 24, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="w")

        # Open Image Button
        open_button = ttk.Button(self.master, text="\t\tOpen JPEG\t\t", command=self.open_image)
        open_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Save As Button
        save_as_button = ttk.Button(self.master, text="Save As JPEG", command=self.save_as_jpeg)
        save_as_button.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # Interpolation Method Menu
        method_label = ttk.Label(self.master, text="Interpolation Method:")
        method_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        method_menu = ttk.Combobox(self.master, textvariable=self.interpolation_method_var,
                                   values=self.interpolation_methods)
        self.interpolation_method_var.set(self.interpolation_methods[0])
        method_menu.grid(row=2, column=1, padx=10, pady=5, sticky="w")




        # New Height Entry
        height_label = ttk.Label(self.master, text="New Height:")
        height_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        height_entry = ttk.Entry(self.master, textvariable=self.new_height_var)
        height_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        # New Width Entry
        width_label = ttk.Label(self.master, text="New Width:")
        width_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")

        width_entry = ttk.Entry(self.master, textvariable=self.new_width_var)
        width_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # Interpolate Button
        interpolate_button = ttk.Button(self.master, text="Interpolate", command=self.interpolate_image)
        interpolate_button.grid(row=5, column=0, padx=10, pady=10, sticky="w")

        # Revert Button
        revert_button = ttk.Button(self.master, text="Revert to Previous", command=self.revert_to_previous)
        revert_button.grid(row=5, column=1, padx=10, pady=10, sticky="w")

        # Display Canvas
        self.canvas = tk.Canvas(self.master, bg="white", width=800, height=600, relief="sunken", borderwidth=2)
        self.canvas.grid(row=1, column=2, rowspan=5, padx=10, pady=10, sticky="nsew")

        # Configure Grid Row/Column Weights for Resizability
        self.master.columnconfigure(2, weight=1)
        self.master.rowconfigure(1, weight=1)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.previous_image = self.input_image
            self.input_image = Image.open(file_path)
            self.display_image(self.input_image)

    def display_image(self, image):
        if image:
            tk_image = ImageTk.PhotoImage(image)
            self.canvas.config(width=tk_image.width(), height=tk_image.height())
            self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
            self.canvas.image = tk_image

    def interpolate_image(self):
        if self.input_image:
            try:
                new_height = int(self.new_height_var.get())
                new_width = int(self.new_width_var.get())
                interpolation_method = self.interpolation_method_var.get()

                if interpolation_method == "Bilinear":
                    self.output_image = self.bilinear_interpolation(self.input_image, new_height, new_width)
                elif interpolation_method == "Bicubic":
                    self.output_image = self.bicubic_interpolation(self.input_image, new_height, new_width)
                elif interpolation_method == "Nearest Neighbor":
                    self.output_image = self.nearest_neighbor_interpolation(self.input_image, new_height, new_width)
                elif interpolation_method == "B-spline":
                    self.output_image = self.bspline_interpolation(self.input_image, new_height, new_width)
                elif interpolation_method == "Lanczos":
                    self.output_image = self.lanczos_interpolation(self.input_image, new_height, new_width)

                self.display_image(self.output_image)
            except ValueError:
                pass  # Handle invalid input

    def revert_to_previous(self):
        if self.previous_image:
            self.input_image = self.previous_image
            self.display_image(self.input_image)

    def save_as_jpeg(self):
        if self.output_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpeg", filetypes=[("JPEG files", "*.jpeg")])
            if file_path:
                self.output_image.save(file_path, format="JPEG")

    def bilinear_interpolation(self, image, new_height, new_width):
        img_array = np.array(image)
        img = Image.fromarray(img_array)
        resized_img = bilinear_interpolation_algorithm(img, new_height, new_width)
        return resized_img

    def bicubic_interpolation(self, image, new_height, new_width):
        img_array = np.array(image)
        img = Image.fromarray(img_array)
        resized_img = bicubic_interpolation_algorithm(img, new_height, new_width)
        return resized_img

    def nearest_neighbor_interpolation(self, image, new_height, new_width):
        img_array = np.array(image)
        img = Image.fromarray(img_array)
        resized_img = nearest_neighbor_interpolation_algorithm(img, new_height, new_width)
        return resized_img

    def bspline_interpolation(self, image, new_height, new_width):
        img_array = np.array(image)
        img = Image.fromarray(img_array)
        resized_img = bspline_interpolation_algorithm(img, new_height, new_width)
        return resized_img

    def lanczos_interpolation(self, image, new_height, new_width):
        img_array = np.array(image)
        img = Image.fromarray(img_array)
        resized_img = lanczos_interpolation_algorithm(img, new_height, new_width)
        return resized_img

def main():
    root = tk.Tk()
    app = ImageInterpolatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
