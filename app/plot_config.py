import math

class PlotConfig:
    def __init__(self, width=10, height=6, dpi=300, override_dpi=None, display_size=None,
                 display_resolution_width=None, display_resolution_height=None):
        """
        Initializes the plot configuration.

        Args:
            width (float): Width of the plot in inches.
            height (float): Height of the plot in inches.
            dpi (int): Dots per inch (resolution) for the plot.
            override_dpi (int, optional): Override DPI value if provided.
            display_size (float, optional): Diagonal display size in inches.
            display_resolution_width (int, optional): Width of the display in pixels.
            display_resolution_height (int, optional): Height of the display in pixels.
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.override_dpi = override_dpi
        self.display_size = display_size
        self.display_resolution_width = display_resolution_width
        self.display_resolution_height = display_resolution_height

    def calculate_dpi(self, pixel_width, pixel_height, diagonal_size):
        """
        Calculates the DPI (Dots per inch) based on screen resolution and size.

        Args:
            pixel_width (int): Width of the screen in pixels.
            pixel_height (int): Height of the screen in pixels.
            diagonal_size (float): Diagonal size of the screen in inches.

        Returns:
            float: Calculated DPI value.
        """
        aspect_ratio = pixel_width / pixel_height
        physical_width = diagonal_size / math.sqrt(1 + (1 / aspect_ratio) ** 2)
        dpi = min(pixel_width / physical_width, pixel_height / physical_width)
        return dpi

    def get_font_size(self, base_size=14):
        """
        Calculates the font size for plot labels based on plot dimensions.

        Args:
            base_size (int): Base font size for a 10x6 inch plot.

        Returns:
            float: Scaled font size based on the plot's width and height.
        """
        return base_size * min(self.width / 10, self.height / 6)

    def get_tick_size(self, base_size=12):
        """
        Calculates the size of the ticks on the plot axes.

        Args:
            base_size (int): Base tick size for a 10x6 inch plot.

        Returns:
            float: Scaled tick size based on the plot's width and height.
        """
        return base_size * min(self.width / 10, self.height / 6)

    def get_marker_size(self, base_size=5):
        """
        Calculates the size of markers in the plot.

        Args:
            base_size (int): Base marker size for a 10x6 inch plot.

        Returns:
            float: Scaled marker size based on the plot's width and height.
        """
        return base_size * min(self.width / 10, self.height / 6)

    def get_line_width(self, base_size=1):
        """
        Calculates the width of the lines in the plot.

        Args:
            base_size (int): Base line width for a 10x6 inch plot.

        Returns:
            float: Scaled line width based on the plot's width and height.
        """
        return base_size * min(self.width / 10, self.height / 6)

    def get_point_label_size(self, base_label_size=12):
        """
        Calculates the size of the point labels based on plot dimensions.

        Args:
            base_label_size (int): Base label size for a 10x6 inch plot.

        Returns:
            float: Scaled label size based on the plot's width and height.
        """
        width_scale = self.width / 10
        height_scale = self.height / 6
        scale = min(width_scale, height_scale)
        return base_label_size * scale

    def apply_tick_size(self, ax):
        """
        Applies the calculated tick size to the axes of the plot.

        Args:
            ax (matplotlib.axes.Axes): The axes object where tick sizes will be applied.
        """
        tick_size = self.get_tick_size()
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)

    def apply_line_marker_and_label_size(self, ax):
        """
        Applies the calculated line width, marker size, and label font size to the axes of the plot.

        Args:
            ax (matplotlib.axes.Axes): The axes object where line width, marker size, and font size will be applied.
        """
        line_width = self.get_line_width()
        marker_size = self.get_marker_size()
        label_size = self.get_point_label_size()

        for line in ax.get_lines():
            line.set_linewidth(line_width)
            line.set_markersize(marker_size)

        for text in ax.texts:
            text.set_fontsize(label_size)
