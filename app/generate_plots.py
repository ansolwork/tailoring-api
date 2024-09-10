import pandas as pd
import os
import ast
import numpy as np
import seaborn as sns
import logging
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from utils.data_processing_utils import DataProcessingUtils
from app.vertex_plot_data_processor import VertexPlotDataProcessor
from app.plot_config import PlotConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_DPI = 300
HPGL_SCALE_FACTOR = 1.297  # Magic constant for scaling HPGL files
GRID_SIZE = 10
OUTPUT_DIR = "data/output/plots/"

class GeneratePlots:
    def __init__(self, input_table_path: str, input_vertices_path: str, grid: bool = True, 
                 plot_actual_size: bool = False, dpi: int = DEFAULT_DPI,
                 override_dpi: int = None, display_size: int = None, 
                 display_resolution_width: int = None, display_resolution_height: int = None):
        """
        Initialize the visualizer with file paths and configuration options.

        :param input_table_path: Path to the input CSV for alteration data.
        :param input_vertices_path: Path to the input CSV for vertex data.
        :param grid: Whether to show a grid on the plot.
        :param plot_actual_size: Whether to plot at actual size.
        :param dpi: Dots per inch for the plot resolution.
        :param override_dpi: Optional DPI override for plotting.
        :param display_size: The display size in inches.
        :param display_resolution_width: Display resolution width in pixels.
        :param display_resolution_height: Display resolution height in pixels.
        """
        self.data_processing_utils = DataProcessingUtils()

        # Load data
        if input_table_path and input_vertices_path:
            self.df = self.data_processing_utils.load_csv(input_table_path)
            self.vertices_df = self.data_processing_utils.load_csv(input_vertices_path)
            self.piece_name = self.get_piece_name(self.df)
        
            # Instantiate VertexPlotDataProcessor to handle vertex scaling and processing
            self.vertex_processor = VertexPlotDataProcessor(self.data_processing_utils)

        # Plot configuration
        self.grid = grid
        self.plot_actual_size = plot_actual_size
        self.dpi = dpi
        self.override_dpi = override_dpi
        self.plot_config = PlotConfig(
            width=10, height=6, dpi=dpi, override_dpi=override_dpi,
            display_size=display_size, display_resolution_width=display_resolution_width,
            display_resolution_height=display_resolution_height
        )

    def prepare_plot_data(self, output_dir: str = "data/staging_processed/"):
        """
        Prepares the plot data using the VertexPlotDataProcessor.
        """
        self.vertex_processor.prepare_plot_data(self.df, self.vertices_df, output_dir)
        self.plot_df = self.vertex_processor.plot_df

    @staticmethod
    def get_piece_name(df: pd.DataFrame) -> str:
        """
        Extracts and returns the piece name from the 'piece_name' column in the dataframe.

        Raises an exception if more than one unique piece name is found.

        :param df: DataFrame containing the data.
        :return: Extracted piece name from the dataframe.
        """
        unique_pieces = df['piece_name'].unique()

        if len(unique_pieces) > 1:
            raise ValueError(f"More than one unique piece name found: {unique_pieces}")
        
        return unique_pieces[0]

    @staticmethod
    def ensure_directory(dir_path: str):
        """
        Ensures that the specified directory exists by creating it if necessary.

        :param dir_path: Path to the directory to be ensured.
        """
        os.makedirs(dir_path, exist_ok=True)

    def create_and_save_grid(self, filename: str, num_squares_x: int = GRID_SIZE, num_squares_y: int = GRID_SIZE):
        """
        Creates a grid with customizable dimensions, where each square is 1x1 inch, and saves it 
        in multiple formats (PNG, SVG, HPGL, DXF).

        :param filename: The base name for the saved grid files (special characters are removed).
        :param num_squares_x: The number of 1-inch squares along the x-axis.
        :param num_squares_y: The number of 1-inch squares along the y-axis.
        """
        output_dir = os.path.join(OUTPUT_DIR, "calibration")
        self.ensure_directory(output_dir)

        # Clean the filename by removing special characters
        filename = filename.replace('#', '').replace(' ', '_')

        # File paths for various formats
        base_path = os.path.join(output_dir, filename)
        png_path = f"{base_path}.png"
        svg_path = f"{base_path}.svg"
        hpgl_path = f"{base_path}.hpgl"
        dxf_path = f"{base_path}.dxf"

        logging.info(f"Starting to create grid: {filename} ({num_squares_x}x{num_squares_y} squares)")

        # Set up the figure and axis for plotting the grid
        fig, ax = plt.subplots(figsize=(num_squares_x, num_squares_y))
        ax.set_xlim(0, num_squares_x)
        ax.set_ylim(0, num_squares_y)
        ax.set_aspect('equal')
        ax.grid(True)

        # Customize the grid to have 1-inch spacing
        ax.set_xticks(np.arange(0, num_squares_x + 1, 1))
        ax.set_yticks(np.arange(0, num_squares_y + 1, 1))

        # Hide the tick labels for a cleaner grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Remove the outer frame (spines) for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the grid in different formats
        self.save_plot_in_formats(fig, ax, filename, output_dir)

    def save_plot_in_formats(self, fig, ax, plot_type, output_dir):
        """
        Saves the given plot in multiple formats: PNG, SVG, HPGL, and DXF.

        :param fig: Matplotlib figure to save.
        :param ax: Matplotlib axis of the figure.
        :param plot_type: Type of plot (e.g., "combined_plot").
        :param output_dir: Directory to save the plot.
        """
        formats = ['svg', 'png', 'hpgl', 'dxf']
        for fmt in formats:
            save_path = os.path.join(output_dir, f"{plot_type}.{fmt}")
            try:
                if fmt == 'svg':
                    self.data_processing_utils.save_plot_as_svg(fig, ax, self.plot_config.width, self.plot_config.height, save_path)
                elif fmt == 'png':
                    fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                elif fmt == 'hpgl':
                    self.data_processing_utils.svg_to_hpgl(save_path.replace('svg', 'svg'), save_path)
                elif fmt == 'dxf':
                    self.data_processing_utils.svg_to_dxf(save_path.replace('svg', 'svg'), save_path)
                logging.info(f"Saved {plot_type} as {fmt} to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save {plot_type} as {fmt}: {e}")

    def plot_polylines_table(self, output_dir="data/output/plots/"):
        """
        Plots the combined data of original, altered, and reduced vertices, and saves the results in multiple formats.

        :param output_dir: Output directory to save the plots.
        """
        piece_dir = os.path.join(output_dir, self.piece_name)
        self.ensure_directory(piece_dir)

        self.ensure_directory(os.path.join(piece_dir, "svg"))
        self.ensure_directory(os.path.join(piece_dir, "hpgl"))
        self.ensure_directory(os.path.join(piece_dir, "dxf"))
        self.ensure_directory(os.path.join(piece_dir, "png"))
        self.ensure_directory(os.path.join(piece_dir, "pdf"))

        if self.grid:
            sns.set(style="whitegrid")
        
        if self.plot_actual_size:
            self.calculate_dimensions()
            self.dpi = self.override_dpi or self.plot_config.calculate_dpi(
                self.plot_config.display_resolution_width, 
                self.plot_config.display_resolution_height, 
                self.plot_config.display_size
            )
            logging.info(f"Calculated DPI: {self.dpi}")
        else:
            self.dpi = self.override_dpi or self.dpi
            logging.info(f"Default DPI: {self.dpi}")

        # Create and save plots
        fig_combined, ax_combined = plt.subplots(figsize=(self.width, self.height))
        fig_vertices, ax_vertices = plt.subplots(figsize=(self.width, self.height))
        fig_alteration, ax_alteration = plt.subplots(figsize=(self.width, self.height))

        self.plot_combined(ax_combined)
        self.plot_only_vertices(ax_vertices)
        self.plot_alteration_table(output_dir=piece_dir, fig=fig_alteration, ax=ax_alteration)

        self.save_plot(fig_combined, ax_combined, "combined_plot", piece_dir)
        self.save_plot(fig_vertices, ax_vertices, "vertices_plot", piece_dir)
        self.save_plot(fig_alteration, ax_alteration, "alteration_table_plot", piece_dir)

        plt.close(fig_vertices)
        plt.close(fig_alteration)
        plt.close(fig_combined)

    def save_plot(self, fig, ax, plot_type, output_dir):
        """
        Saves the given plot in various formats.

        :param fig: Matplotlib figure to save.
        :param ax: Matplotlib axis of the figure.
        :param plot_type: Type of plot (e.g., "combined_plot").
        :param output_dir: Directory to save the plot.
        """
        svg_path = os.path.join(output_dir, "svg", f"{plot_type}_{self.piece_name}.svg")
        png_path = os.path.join(output_dir, "png", f"{plot_type}_{self.piece_name}.png")
        hpgl_path = os.path.join(output_dir, "hpgl", f"{plot_type}_{self.piece_name}.hpgl")
        dxf_path = os.path.join(output_dir, "dxf", f"{plot_type}_{self.piece_name}.dxf")

        logging.info(f"Saving {plot_type} as PNG to {png_path}")
        fig.savefig(png_path, dpi=self.dpi, bbox_inches='tight')
        
        logging.info(f"Saving {plot_type} as SVG to {svg_path}")
        self.data_processing_utils.save_plot_as_svg(fig, ax, self.width, self.height, svg_path, add_labels=False)

        logging.info(f"Converting SVG to HPGL and saving to {hpgl_path}")
        self.data_processing_utils.svg_to_hpgl(svg_path, hpgl_path)

        logging.info(f"Converting SVG to DXF and saving to {dxf_path}")
        self.data_processing_utils.svg_to_dxf(svg_path, dxf_path)

    def calculate_dimensions(self):
        """
        Calculate dimensions based on the data to be plotted.
        """
        xs_all = [x for xs in self.plot_df['unique_vertices_x'].dropna() for x in xs]
        ys_all = [y for ys in self.plot_df['unique_vertices_y'].dropna() for y in ys]

        self.width = max(xs_all) - min(xs_all)
        self.height = max(ys_all) - min(ys_all)

        logging.info(f"Calculated width: {self.width}, height: {self.height}")

    def plot_only_vertices(self, ax):
        """
        Plot only the original vertices.

        :param ax: Matplotlib axis to plot on.
        """
        font_size = self.plot_config.get_font_size()
        marker_size = self.plot_config.get_marker_size()
        line_width = self.plot_config.get_line_width()

        for _, row in self.plot_df.iterrows():
            if pd.notna(row['unique_vertices_x']) and pd.notna(row['unique_vertices_y']):
                xs, ys = row['unique_vertices_x'], row['unique_vertices_y']
                ax.plot(xs, ys, marker='o', linewidth=line_width, markersize=marker_size, color="#006400", label="Original")
        
        ax.set_title(f'Original Vertices Plot for {self.piece_name}', fontsize=font_size+2)
        ax.set_xlabel('X Coordinate [in]', fontsize=font_size)
        ax.set_ylabel('Y Coordinate [in]', fontsize=font_size)

        self.plot_config.apply_tick_size(ax)

    def plot_combined(self, ax):
        """
        Plot the combined vertices (original, altered, and reduced).

        :param ax: Matplotlib axis to plot on.
        """
        font_size = self.plot_config.get_font_size()
        marker_size = self.plot_config.get_marker_size()
        line_width = self.plot_config.get_line_width()

        for _, row in self.plot_df.iterrows():
            xs, ys = row['unique_vertices_x'], row['unique_vertices_y']
            xs_alt = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_x'])
            ys_alt = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_y'])
            xs_alt_reduced = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_reduced_x'])
            ys_alt_reduced = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_reduced_y'])

            ax.plot(xs, ys, marker='o', linewidth=line_width, markersize=marker_size, color="#006400", label="Original")
            ax.plot(xs_alt_reduced, ys_alt_reduced, marker='x', linestyle='-.', linewidth=line_width+0.5, markersize=marker_size+3, color="#BA55D3", alpha=0.7, label="Altered Reduced")
            ax.plot(xs_alt, ys_alt, marker='o', linestyle='-', linewidth=line_width, markersize=marker_size, color="#00CED1", alpha=0.85, label="Altered")

        self.plot_mtm_points(ax)
        
        ax.set_title(f'Combined Vertices Plot for {self.piece_name}', fontsize=font_size+2)
        ax.set_xlabel('X Coordinate [in]', fontsize=font_size)
        ax.set_ylabel('Y Coordinate [in]', fontsize=font_size)

        self.plot_config.apply_tick_size(ax)

        ax.legend(handles=[
            Line2D([0], [0], color="#006400", marker='o', label="Original", linewidth=line_width, markersize=marker_size),
            Line2D([0], [0], color="#00CED1", linestyle='-', marker='o', label="Altered", linewidth=line_width, markersize=marker_size),
            Line2D([0], [0], color="#BA55D3", linestyle='-.', marker='x', label="Altered Reduced", linewidth=line_width+0.5, markersize=marker_size+3)
        ])

    def plot_alteration_table(self, output_dir, fig=None, ax=None):
        """
        Plot the alteration table, showing MTM points and their dependencies.

        :param output_dir: Output directory to save the plot.
        :param fig: Matplotlib figure.
        :param ax: Matplotlib axis to plot on.
        """
        font_size = self.plot_config.get_font_size()
        marker_size = self.plot_config.get_marker_size()
        line_width = self.plot_config.get_line_width()

        if self.grid:
            sns.set(style="whitegrid")

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))

        ax.set_aspect('equal', 'box')

        plot_df = self.plot_df.copy()

        mtm_alt_labels = []
        coords = {
            "old_coords": [],
            "new_coords": [],
            "mtm_dependent": [],
            "mtm_dependent_coords": [],
            "nearby_labels_left": [],
            "nearby_labels_right": [],
            "nearby_coords_left": [],
            "nearby_coords_right": []
        }

        for _, row in plot_df.iterrows():
            point = row['mtm_points']
            label = point['label']
            belongs_to = point['belongs_to']
            mtm_dependent = point['mtm_dependent']
            mdp_x, mdp_y = point['mtm_dependent_x'], point['mtm_dependent_y']
            nearby_labels_left = point['nearby_point_left']
            nearby_labels_right = point['nearby_point_right']
            nearby_coords_left = point['nearby_point_left_coords']
            nearby_coords_right = point['nearby_point_right_coords']

            if belongs_to == "altered":
                mtm_alt_labels.append(label)

                coords["old_coords"].append(((point['x_old']), (point['y_old'])))
                coords["new_coords"].append(((point['x']), (point['y'])))
                coords["nearby_labels_left"].append(nearby_labels_left)
                coords["nearby_labels_right"].append(nearby_labels_right)
                coords["nearby_coords_left"].append(nearby_coords_left)
                coords["nearby_coords_right"].append(nearby_coords_right)

                if isinstance(mdp_x, str) and isinstance(mdp_y, str):
                    mdp_x = ast.literal_eval(mdp_x) 
                    mdp_y = ast.literal_eval(mdp_y) 

                    if isinstance(mdp_x, list) and isinstance(mdp_y, list):
                        mtm_dependent_coords = list(zip(mdp_x, mdp_y))
                    else: 
                        mtm_dependent_coords = ((mdp_x), (mdp_y))
                else:
                    mtm_dependent_coords = ((mdp_x), (mdp_y))
                
                coords["mtm_dependent_coords"].append(mtm_dependent_coords)
                
                if isinstance(mtm_dependent, str):
                    mtm_dependent = ast.literal_eval(mtm_dependent)
                if mtm_dependent not in coords["mtm_dependent"]:
                    coords["mtm_dependent"].append(mtm_dependent)

        coords["mtm_dependent"] = self.data_processing_utils.flatten_if_needed(coords["mtm_dependent"])
        coords["mtm_dependent_coords"] = self.data_processing_utils.flatten_if_needed(coords["mtm_dependent_coords"])
        coords["mtm_dependent_coords"] = self.data_processing_utils.remove_duplicates(coords["mtm_dependent_coords"])
        coords["mtm_dependent_coords"] = self.data_processing_utils.sort_by_x(coords["mtm_dependent_coords"])

        dependent_first = coords["mtm_dependent_coords"][0]
        dependent_last = coords["mtm_dependent_coords"][-1]

        xs_alt_list, ys_alt_list = [], []
        for _, row in plot_df.iterrows():
            xs_alt = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_x'])
            ys_alt = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_y'])
            xs, ys = row['unique_vertices_x'], row['unique_vertices_y']
            
            if not pd.isna(xs) and not pd.isna(ys):
                coords_list = list(zip(xs, ys))
                try:
                    start_index = coords_list.index(dependent_first)
                    end_index = coords_list.index(dependent_last)
                    points_between = coords_list[start_index + 1:end_index]
                except:
                    points_between = []

                segments = []
                current_segment = []

                for index, pair in enumerate(coords_list):
                    if pair not in points_between:
                        current_segment.append(pair)
                    else:
                        if current_segment:
                            segments.append(current_segment)
                            current_segment = []

                if current_segment:
                    segments.append(current_segment)

                # Replace vertices with new pairs and plot each segment separately
                for segment in segments:
                    for index, pair in enumerate(segment):
                        if pair in coords["old_coords"]:
                            old_index = coords["old_coords"].index(pair)
                            new_pair = coords["new_coords"][old_index]
                            segment[index] = new_pair

                    # Extract x and y coordinates for the segment
                    x_coords = tuple(x for x, y in segment)
                    y_coords = tuple(y for x, y in segment)

                    # Plot the segment
                    if x_coords and y_coords:  # Ensure that the segment is not empty
                        ax.plot(x_coords, y_coords, marker='o', linestyle='-', linewidth=line_width, markersize=marker_size, color="#00CED1", alpha=0.85, label="Altered")

                    if xs_alt and ys_alt:  # Check if xs_alt and ys_alt are not empty
                        xs_alt_list.append(xs_alt)
                        ys_alt_list.append(ys_alt)

        xs_alt_list = self.data_processing_utils.flatten_if_needed(xs_alt_list)
        ys_alt_list = self.data_processing_utils.flatten_if_needed(ys_alt_list)

        # Set aspect ratio to be equal to maintain the scale in actual size
        ax.set_aspect('equal', 'box')
        
        # Plot the altered segment
        ax.plot(xs_alt_list, ys_alt_list, marker='o', linestyle='-', linewidth=line_width, markersize=marker_size, color="#00CED1", alpha=0.85, label="Altered")

        ax.set_title(f'Alteration plot for {self.piece_name}', fontsize=font_size+2)
        ax.set_xlabel('X Coordinate [in]', fontsize=font_size)
        ax.set_ylabel('Y Coordinate [in]', fontsize=font_size)

        self.plot_config.apply_tick_size(ax)

        # Plot MTM points
        for _, row in plot_df.iterrows():
            point = row['mtm_points']
            label = point['label']
            belongs_to = point['belongs_to']

            if belongs_to == "original":
                if label in mtm_alt_labels:
                    continue
                else:
                    ax.plot(point['x'], point['y'], 'o', color=point['color'], markersize=marker_size)
                    ax.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=font_size)

            if belongs_to == "altered":
                ax.plot(point['x'], point['y'], 'o', color=point['color'], markersize=marker_size)
                ax.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=font_size)

    def plot_mtm_points(self, ax):
        marker_size = self.plot_config.get_marker_size()
        font_size = self.plot_config.get_font_size()

        # Plot MTM points
        for _, row in self.plot_df.iterrows():
            point = row['mtm_points']

            ax.plot(point['x'], point['y'], 'o', color=point['color'], markersize=marker_size)
            ax.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=font_size)

def process_files():
    """
    Process alterations and vertices files by matching their `piece_name` and creating visualizations.
    """
    processed_alterations_dir = "data/staging_processed/processed_alterations/"
    processed_vertices_dir = "data/staging_processed/processed_vertices/"
    
    for altered_file in os.listdir(processed_alterations_dir):
        if altered_file.endswith(".csv"):
            input_altered_table_path = os.path.join(processed_alterations_dir, altered_file)
            
            # Load the alterations dataframe
            alteration_df = pd.read_csv(input_altered_table_path)
            
            try:
                altered_piece_name = GeneratePlots.get_piece_name(alteration_df)
            except ValueError as e:
                logging.error(f"Error with alteration file {altered_file}: {e}")
                continue  # Skip to the next file if there's an issue with the piece name
            
            # Now loop through the vertices directory to find the matching file
            matching_vertices_file = None
            for vertices_file in os.listdir(processed_vertices_dir):
                if vertices_file.endswith(".csv"):
                    input_vertices_table_path = os.path.join(processed_vertices_dir, vertices_file)
                    
                    # Load the vertices dataframe
                    vertices_df = pd.read_csv(input_vertices_table_path)
                    
                    try:
                        vertices_piece_name = GeneratePlots.get_piece_name(vertices_df)
                    except ValueError as e:
                        logging.error(f"Error with vertices file {vertices_file}: {e}")
                        continue  # Skip to the next vertices file
                    
                    # Check if the `piece_name` matches
                    if altered_piece_name == vertices_piece_name:
                        matching_vertices_file = input_vertices_table_path
                        break  # We found a match, stop looking further
            
            if matching_vertices_file:
                # Proceed with visualization using the matched files
                # NOTE: DPI can only be overridden if plot_actual_size = False
                try:
                    visualize_alteration = GeneratePlots(
                        input_table_path=input_altered_table_path,
                        input_vertices_path=matching_vertices_file, 
                        grid=False, plot_actual_size=True, 
                        display_size=16, # inches
                        display_resolution_width=3456, 
                        display_resolution_height=2234
                    )
                    visualize_alteration.prepare_plot_data()
                    visualize_alteration.plot_polylines_table()
                except Exception as e:
                    logging.error(f"Error processing {altered_file} and {matching_vertices_file}: {e}")
            else:
                logging.warning(f"No matching vertices file found for {altered_piece_name}")

if __name__ == "__main__":
    # Display resolution set for Macbook Pro m2 - change to whatever your screen res is
    display_resolution_width = 3456
    display_resolution_height = 2234
    display_size = 16  

    # Create and save the grid once
    visualize_alteration = GeneratePlots(
        input_table_path="",  # Temporary placeholder
        input_vertices_path="",  # Temporary placeholder
        grid=False, 
        plot_actual_size=True, 
        display_size=display_size,
        display_resolution_width=display_resolution_width, 
        display_resolution_height=display_resolution_height
    )

    # Create and save grid only once
    grid_filename = '10x10_grid'
    logging.info(f"Creating grid: {grid_filename}")
    visualize_alteration.create_and_save_grid(grid_filename)

    # Process all files
    #process_files()

    # Or process one at a time (DEBUGGING)
    input_table_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_1LTH-FULL.csv"
    input_vertices_path = "data/staging_processed/processed_vertices_by_piece/processed_vertices_LGFG-SH-01-CCB-FO.csv"
    
    # Call Generate Plots
    visualize_alteration = GeneratePlots(
        input_table_path=input_table_path,  # Temporary placeholder
        input_vertices_path=input_vertices_path,  # Temporary placeholder
        grid=False, 
        plot_actual_size=True, 
        display_size=display_size,
        display_resolution_width=display_resolution_width, 
        display_resolution_height=display_resolution_height
    )

    visualize_alteration.prepare_plot_data()
    visualize_alteration.plot_polylines_table()

