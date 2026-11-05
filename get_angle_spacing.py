"""Interactive Fourier-peak selection using the original angle convention."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class AngleSpacingCalculator:
    """Drag around one sideband peak and press Enter to confirm and close."""

    def __init__(self, image, dx):
        self.image = np.asarray(image)
        if self.image.ndim != 2 or self.image.shape[0] != self.image.shape[1]:
            raise ValueError("The spectrum must be a square 2D image.")
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError("Pixel size must be positive.")
        self.image_size = self.image.shape[0]
        self.dx = dx
        self.xc = self.yc = self.image_size // 2
        self.start_x = self.start_y = None
        self.brightest_pixel = None
        self.fig, self.ax = plt.subplots()
        self.rect = Rectangle((0, 0), 0, 0, fill=False, edgecolor="red", linewidth=2)
        self.ax.imshow(self.image, interpolation="nearest")
        self.ax.add_patch(self.rect)
        self.ax.set_title("Drag around one noncentral peak; Enter to confirm")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self.start_x, self.start_y = event.xdata, event.ydata
        self.brightest_pixel = None
        self.rect.set_xy((self.start_x, self.start_y))
        self.rect.set_width(0)
        self.rect.set_height(0)

    def on_drag(self, event):
        if self.start_x is None or event.inaxes != self.ax or event.button != 1:
            return
        self.rect.set_width(event.xdata - self.start_x)
        self.rect.set_height(event.ydata - self.start_y)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if event.inaxes == self.ax and self.start_x is not None and event.button == 1:
            self.process_selection(event.xdata, event.ydata)

    def process_selection(self, end_x, end_y):
        if self.start_x is None or self.start_y is None:
            return
        x1, x2 = sorted((self.start_x, end_x))
        y1, y2 = sorted((self.start_y, end_y))
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(self.image_size, int(x2)), min(self.image_size, int(y2))
        cropped = self.image[y1:y2, x1:x2]
        if cropped.size:
            peak_y, peak_x = np.unravel_index(np.argmax(cropped), cropped.shape)
            self.brightest_pixel = (int(peak_x + x1), int(peak_y + y1))

    def on_key_press(self, event):
        if event.key == "enter":
            try:
                self.get_ang_spacing()
            except ValueError as error:
                self.ax.set_title(str(error))
                self.fig.canvas.draw_idle()
            else:
                plt.close(self.fig)

    def show(self):
        plt.show(block=True)

    def get_brightest_pixel(self):
        return self.brightest_pixel

    def get_ang_spacing_given_brightest_pixel(self, bp):
        self.brightest_pixel = bp
        return self.get_ang_spacing()

    def get_ang_spacing(self):
        if self.brightest_pixel is None:
            raise ValueError("Select a noncentral frequency peak first.")
        xb, yb = self.brightest_pixel
        if not (np.isfinite(xb) and np.isfinite(yb) and 0 <= xb < self.image_size and 0 <= yb < self.image_size):
            raise ValueError("Selected peak must lie within the image.")
        distance = np.hypot(xb - self.xc, yb - self.yc)
        if distance == 0:
            raise ValueError("Choose a sideband peak away from the image center.")
        angle = -np.arctan2(xb - self.xc, -yb + self.yc)
        spacing = self.image_size * self.dx / distance
        return float(angle), float(spacing)
