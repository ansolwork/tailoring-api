import pandas as pd
import numpy as np
import math
import xml.etree.ElementTree as ET

import subprocess
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments

class DataProcessingUtils:
    """
    A utility class for common data processing tasks, including CSV operations
    and list manipulations.
    """

    def load_csv(self, table_path):
        """
        Load a CSV file into a pandas DataFrame.
        
        Parameters:
        table_path (str): The file path to the CSV file.

        Returns:
        pd.DataFrame: The loaded DataFrame.
        """
        return pd.read_csv(table_path)
    
    def save_csv(self, df, save_path):
        """
        Save a pandas DataFrame to a CSV file.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to save.
        save_path (str): The file path where the CSV should be saved.
        """
        df.to_csv(save_path, index=False)

    def load_excel(self, input_table_path):
        return pd.read_excel(input_table_path)
    
    def load_all_excel_sheets(self, df):
        """
        Load all sheets from an Excel file into a dictionary of DataFrames.
        
        Parameters:
        df (str): The file path to the Excel file.

        Returns:
        dict of pd.DataFrame: A dictionary where each key is a sheet name
                              and each value is a DataFrame containing the data.
        """
        return pd.read_excel(df, sheet_name=None)
    
    def drop_columns(self, df, columns_to_drop):
        """
        Drop specified columns from a pandas DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame from which to drop columns.
        columns_to_drop (list): A list of column names to be dropped.

        Returns:
        pd.DataFrame: The DataFrame with the specified columns dropped.
        """
        return df.drop(columns=columns_to_drop)
    
    def remove_duplicates(self, input_list):
        """
        Remove duplicates from a list of tuples or coordinates.

        Parameters:
        input_list (list): A list of tuples where each tuple represents coordinates.

        Returns:
        list: A list with duplicates removed, maintaining the original order.
        """
        seen = set()
        output_list = []
        for item in input_list:
            if item not in seen:
                seen.add(item)
                output_list.append(item)
        return output_list
    
    def drop_nans(self, lst):
        return [item for item in lst if not (isinstance(item, float) and math.isnan(item))]
    
    def flatten_tuple(self, nested_tuple):
        flat_list = []
        for item in nested_tuple:
            if isinstance(item, (list, tuple)):
                flat_list.extend(self.flatten_tuple(item))
            else:
                flat_list.append(item)
        return flat_list
    
    def filter_valid_coordinates(self, coords):
        if isinstance(coords, (list, tuple)):
            return [coord for coord in coords if not pd.isna(coord)]
        return []
    
    def scale_coordinates(self, xs, ys, scaling_factor):
        xs = tuple(x * scaling_factor for x in xs)
        ys = tuple(y * scaling_factor for y in ys)
        return xs, ys

    def remove_duplicates_preserve_order(self, flattened_list):
        """
        Remove duplicates from a list while preserving the order of elements.
        
        Parameters:
        flattened_list (list): The list from which to remove duplicates.

        Returns:
        list: A list with duplicates removed, preserving the original order.
        """
        unique_list = []
        seen = set()

        for item in flattened_list:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)

        return unique_list
    
    def sort_by_x(self, coords):
        """
        Sort a list of coordinates by the x-coordinate.
        
        Parameters:
        coords (list): A list of tuples where each tuple represents (x, y) coordinates.

        Returns:
        list: The list sorted by the x-coordinate.
        """
        if isinstance(coords, list) and all(isinstance(coord, (list, tuple)) for coord in coords):
            return sorted(coords, key=lambda coord: coord[0])
        return coords  # Return as is if not a list of tuples
    
    def reduce_points(self, vertices, threshold=0.1):
        """
        Reduce the number of points in a list of vertices based on a threshold using the Visvalingam-Whyatt algorithm.
        
        Parameters:
        vertices (list): A list of tuples representing the vertices (x, y).
        threshold (float): The threshold for point reduction. Lower values result in more points being removed.

        Returns:
        list: The reduced list of vertices.
        """
        if isinstance(vertices, list) and len(vertices) > 2:
            points = np.array(vertices)
            reduced_points = self.visvalingam_whyatt(points, threshold)
            return reduced_points
        return vertices

    def flatten_if_needed(self, nested_list):
        flattened = []
        for item in nested_list:
            if isinstance(item, list):  # Check if the item is a list
                flattened.extend(item)  # If so, extend the flattened list
            else:
                flattened.append(item)  # If not, simply append the item
        return flattened
    
    def visvalingam_whyatt(self, points, threshold):
        """
        Simplify a list of points using the Visvalingam-Whyatt algorithm.
        
        Parameters:
        points (list): A list of tuples representing the points (x, y).
        threshold (float): The area threshold used to remove points. Smaller areas will be removed.

        Returns:
        list: A simplified list of points.
        """
        def calculate_area(a, b, c):
            """
            Calculate the area of the triangle formed by three points.
            
            Parameters:
            a, b, c (tuple): Points (x, y) that form the vertices of the triangle.

            Returns:
            float: The area of the triangle.
            """
            return 0.5 * abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))

        def filter_points(points, threshold):
            """
            Filter points by removing those that form the smallest areas below the threshold.
            
            Parameters:
            points (list): The list of points (x, y) to be filtered.
            threshold (float): The area threshold below which points are removed.

            Returns:
            list: The filtered list of points.
            """
            if len(points) < 3:
                return points
            areas = [float('inf')] * len(points)
            for i in range(1, len(points) - 1):
                areas[i] = calculate_area(points[i - 1], points[i], points[i + 1])
            while min(areas) < threshold:
                min_index = areas.index(min(areas))
                points.pop(min_index)
                areas.pop(min_index)
                if min_index > 0 and min_index < len(points) - 1:
                    areas[min_index - 1] = calculate_area(points[min_index - 2], points[min_index - 1], points[min_index]) if min_index - 1 != 0 else float('inf')
                    areas[min_index] = calculate_area(points[min_index - 1], points[min_index], points[min_index + 1]) if min_index + 1 != len(points) - 1 else float('inf')
            return points
        
        points = [tuple(point) for point in points]
        simplified_points = filter_points(points, threshold)

        return simplified_points
    
    def modify_svg_dimensions(self, svg_path, desired_width_in_inches, desired_height_in_inches):
        """
        Modifies the width, height, and viewBox attributes of the saved SVG to reflect real-world inches.
        Ensures correct scaling of the SVG content to match the desired dimensions.
        """
        # Parse the SVG file
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Set the new width and height attributes in inches
        root.set('width', f'{desired_width_in_inches}in')
        root.set('height', f'{desired_height_in_inches}in')

        # Extract original dimensions from the viewBox if available
        viewBox = root.get('viewBox')
        if viewBox:
            original_dimensions = list(map(float, viewBox.split()[2:]))
            original_width, original_height = original_dimensions
        else:
            # If viewBox is not present, fallback to width and height attributes in points or pixels
            original_width = float(root.get('width').replace('pt', '').replace('px', ''))
            original_height = float(root.get('height').replace('pt', '').replace('px', ''))

        # Set the viewBox to match the desired dimensions
        root.set('viewBox', f'0 0 {original_width} {original_height}')

        # Remove any scaling applied to individual elements
        for element in root.findall(".//{http://www.w3.org/2000/svg}g"):
            if "transform" in element.attrib:
                del element.attrib["transform"]

        # Save the modified SVG back to disk
        tree.write(svg_path)

        print(f"SVG dimensions updated to {desired_width_in_inches}x{desired_height_in_inches} inches in {svg_path}")

    def save_plot_as_svg(self, fig, ax, svg_width, svg_height, output_svg_path, add_labels=False):
        """
        Saves the given plot to an SVG file.
        """

        # Ensure DPI is set to match the intended figure size
        fig.set_size_inches(svg_width, svg_height)
        fig.dpi = 72  # This ensures 1:1 inch scaling in the SVG

        # Remove the legend if it exists
        legend = ax.get_legend()
        if legend:
            legend.remove()

        # Remove point labels (text annotations)
        for text in ax.texts:
            text.remove()

        if not add_labels:
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            
            # Hide tick labels for a clean grid
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Set the aspect ratio to be equal and adjust ticks to represent inches
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, svg_width + 1, 1))  # 1-inch spacing on the x-axis
        ax.set_yticks(np.arange(0, svg_height + 1, 1))  # 1-inch spacing on the y-axis

        svg_width = np.round(svg_width, 1)
        svg_height = np.round(svg_height, 1)

        # Adjust axes limits to match the SVG dimensions exactly
        ax.set_xlim(0, svg_width)
        ax.set_ylim(0, svg_height)

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Remove any potential padding, or adjust as needed
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Ensure no extra padding is added by ticks
        ax.tick_params(which='both', width=0, length=0)

        # Save the figure as an SVG file with the specified DPI
        fig.savefig(output_svg_path, format='svg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure after saving

        print(f"SVG plot saved to {output_svg_path} with dimensions {svg_width}x{svg_height} inches")

        # Modify the SVG file to ensure correct width and height in inches
        #self.modify_svg_dimensions(output_svg_path, svg_width, svg_height)

    def svg_to_hpgl(self, svg_path, output_hpgl_path):
        """
        Converts an SVG file to HPGL format using Inkscape.
        """
        try:
            command = [
                "inkscape", 
                svg_path, 
                "--export-type=hpgl", 
                f"--export-filename={output_hpgl_path}",
                "--export-dpi=72"  # Explicitly set DPI to 72
            ]
            subprocess.run(command, check=True)
            print(f"HPGL file saved to {output_hpgl_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting SVG to HPGL: {e}")

    def svg_to_dxf(self, svg_path, output_dxf_path):
        try:
            command = [
                "inkscape", 
                svg_path, 
                "--export-type=dxf", 
                f"--export-filename={output_dxf_path}"
            ]
            subprocess.run(command, check=True)
            print(f"DXF file saved to {output_dxf_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting SVG to DXF: {e}")
