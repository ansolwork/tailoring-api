import pandas as pd
import os
import ast
import numpy as np
import seaborn as sns
import math
import hashlib
from svgpathtools import svg2paths2
import vpype

import xml.etree.ElementTree as ET

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments

from data_processing_utils import DataProcessingUtils

# Next make a grid 1 inch per square - length/width. Square grid?

# TODO: Add some margin to the bounding box? So we can see the points
# TODO: Show labels in SVG Plot?
# TODO: Fix the other cases: Check why unique vertices has multiple columns in other files. Maybe thats where the error comes?
# TODO: Move grid out of the class

class VisualizeAlteration:
    def __init__(self, input_table_path, input_vertices_path, grid=True, plot_actual_size=False, 
                 override_dpi=None, display_size=None, display_resolution_width=None, display_resolution_height=None):
        
        self.data_processing_utils = DataProcessingUtils()
        self.input_table_path = input_table_path
        self.piece_name = self.get_piece_name(self.input_table_path)
        self.input_vertices_path = input_vertices_path
        self.df = self.data_processing_utils.load_csv(self.input_table_path)
        self.vertices_df = self.data_processing_utils.load_csv(self.input_vertices_path)
        self.plot_df = ""
        
        # Scaling
        #self.scaling_factor = 25.4 # to mm
        self.scaling_factor = 1
        self.multiplier = math.sqrt(self.scaling_factor)        
        self.scaled_unique_vertices = []
        self.scaled_altered_vertices = []
        self.scaled_altered_vertices_reduced = []
        
        # Plot parameters
        self.grid = grid
        self.plot_actual_size = plot_actual_size
        self.width = 10
        self.height = 6

        # Display settings
        # DPI: Dots / Pixels per inch
        # Needs to be calculated dynamically to fit image to screen
        self.dpi = 300 # default value
        self.override_dpi = override_dpi
        self.display_size = display_size
        self.display_resolution_width = display_resolution_width
        self.display_resolution_height = display_resolution_height 

    def get_piece_name(self, input_table_path):
        # Get the base name of the file (e.g., "altered_LGFG-1648-FG-07S.xlsx")
        file_name = os.path.basename(input_table_path)

        # Remove the extension (e.g., ".xlsx") to get "altered_LGFG-1648-FG-07S"
        file_name_without_extension = os.path.splitext(file_name)[0]

        # Split the string by "_" and get the last part
        piece_name = file_name_without_extension.split('_', 1)[-1]

        return piece_name
    
    def calculate_dpi(self, pixel_width, pixel_height, diagonal_size):
        """
        Calculate DPI based on pixel resolution and physical screen diagonal size.
        """
        # Calculate the aspect ratio
        aspect_ratio = pixel_width / pixel_height
        
        # Calculate the physical width and height using the diagonal and aspect ratio
        physical_width = diagonal_size / math.sqrt(1 + (1 / aspect_ratio) ** 2)
        physical_height = physical_width / aspect_ratio
        
        # Calculate DPI using the physical width and height
        dpi_width = pixel_width / physical_width
        dpi_height = pixel_height / physical_height
        
        # Use the smaller DPI to ensure both dimensions fit within the desired physical size
        dpi = min(dpi_width, dpi_height)
        
        return dpi
    
    def get_font_size(self):
        """Calculate font size based on the figure size."""
        # Base size for a 10x6 inch plot
        base_width = 10
        base_height = 6
        base_font_size = 14

        # Calculate scaling factors based on the current width and height
        width_scale = self.width / base_width
        height_scale = self.height / base_height

        # Use the smaller scale to ensure labels fit well
        scale = min(width_scale, height_scale)

        # Return the scaled font size
        return base_font_size * scale
    
    def get_tick_size(self):
        """Calculate tick size based on the figure size."""
        # Base size for a 10x6 inch plot
        base_width = 10
        base_height = 6
        base_tick_size = 12

        # Calculate scaling factors based on the current width and height
        width_scale = self.width / base_width
        height_scale = self.height / base_height

        # Use the smaller scale to ensure ticks fit well
        scale = min(width_scale, height_scale)

        # Return the scaled tick size
        return base_tick_size * scale
    
    def apply_tick_size(self, ax):
        """Apply the dynamically calculated tick size to the axis."""
        tick_size = self.get_tick_size()
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)

    def get_line_width(self):
        """Calculate line width based on the figure size."""
        # Base line width for a 10x6 inch plot
        base_width = 10
        base_height = 6
        base_line_width = 1

        # Calculate scaling factors based on the current width and height
        width_scale = self.width / base_width
        height_scale = self.height / base_height

        # Use the smaller scale to ensure line width fits well
        scale = min(width_scale, height_scale)

        # Return the scaled line width
        return base_line_width * scale

    def get_marker_size(self):
        """Calculate marker size based on the figure size."""
        # Base marker size for a 10x6 inch plot
        base_width = 10
        base_height = 6
        base_marker_size = 5

        # Calculate scaling factors based on the current width and height
        width_scale = self.width / base_width
        height_scale = self.height / base_height

        # Use the smaller scale to ensure markers fit well
        scale = min(width_scale, height_scale)

        # Return the scaled marker size
        return base_marker_size * scale

    def get_point_label_size(self):
        """Calculate point label size based on the figure size."""
        # Base label size for a 10x6 inch plot
        base_width = 10
        base_height = 6
        base_label_size = 12

        # Calculate scaling factors based on the current width and height
        width_scale = self.width / base_width
        height_scale = self.height / base_height

        # Use the smaller scale to ensure labels fit well
        scale = min(width_scale, height_scale)

        # Return the scaled label size
        return base_label_size * scale

    def apply_line_marker_and_label_size(self, ax):
        """Apply the dynamically calculated line width, marker size, and label size to the axis."""
        line_width = self.get_line_width()
        marker_size = self.get_marker_size()
        label_size = self.get_point_label_size()

        # Example: Apply these settings to all lines and markers in the axis
        for line in ax.get_lines():
            line.set_linewidth(line_width)
            line.set_markersize(marker_size)

        # Apply label size to all text annotations
        for text in ax.texts:
            text.set_fontsize(label_size)
    
    def initialize_plot_data(self):
        return {
            'unique_vertices': [],
            'unique_vertices_xs': [],
            'unique_vertices_ys': [],
            'altered_vertices': [],
            'altered_vertices_xs': [],
            'altered_vertices_ys': [],
            'altered_vertices_reduced': [],
            'altered_vertices_reduced_xs': [],
            'altered_vertices_reduced_ys': [],
            'mtm_points': []
        }
    
    def process_vertices(self, row, plot_data, scaled_vertices):
        vertices_list = ast.literal_eval(row['original_vertices_reduced'])
        
        if vertices_list:
            xs, ys = zip(*vertices_list)
            xs, ys = self.data_processing_utils.scale_coordinates(xs, ys, self.scaling_factor)

            # Convert the scaled vertices to a tuple and create a unique identifier
            scaled_vertex_tuple = tuple(zip(xs, ys))
            vertex_hash = hashlib.md5(str(scaled_vertex_tuple).encode()).hexdigest()

            # Check if this hash is not already in the plot_data
            if vertex_hash not in plot_data['unique_vertices']:
                plot_data['unique_vertices'].append(vertex_hash)
                plot_data['unique_vertices_xs'].append(xs)
                plot_data['unique_vertices_ys'].append(ys)
                scaled_vertices.append(scaled_vertex_tuple)

    def process_altered_vertices(self, row, plot_data, scaled_vertices):
        raw_altered = row['altered_vertices']
        if pd.notna(raw_altered):
            altered_vertices_list = ast.literal_eval(raw_altered)
            if altered_vertices_list and altered_vertices_list not in plot_data['altered_vertices']:
                plot_data['altered_vertices'].append(altered_vertices_list)
                xs, ys = zip(*altered_vertices_list)
                xs, ys = self.data_processing_utils.scale_coordinates(xs, ys, self.scaling_factor)
                plot_data['altered_vertices_xs'].append(xs)
                plot_data['altered_vertices_ys'].append(ys)
                scaled_vertices.append(tuple(zip(xs, ys)))

    def process_altered_vertices_reduced(self, row, plot_data, scaled_vertices):
        raw_altered_reduced = row['altered_vertices_reduced']
        if not pd.isna(raw_altered_reduced):
            altered_vertices_list_reduced = ast.literal_eval(raw_altered_reduced)
            if altered_vertices_list_reduced and altered_vertices_list_reduced not in plot_data['altered_vertices_reduced']:
                plot_data['altered_vertices_reduced'].append(altered_vertices_list_reduced)
                xs, ys = zip(*altered_vertices_list_reduced)
                xs, ys = self.data_processing_utils.scale_coordinates(xs, ys, self.scaling_factor)
                plot_data['altered_vertices_reduced_xs'].append(xs)
                plot_data['altered_vertices_reduced_ys'].append(ys)
                scaled_vertices.append(tuple(zip(xs, ys)))

    def process_mtm_points(self, row, plot_data):
        def add_point(x, y, x_old, y_old, label, mtm_dependent, mtm_dependent_x, mtm_dependent_y, 
                    color, movement_x=0., movement_y=0., nearby_point_left=None, nearby_point_right=None,
                    nearby_point_left_coords=None, nearby_point_right_coords=None):
            if x is not None and y is not None:
                plot_data['mtm_points'].append({
                    'x': x * self.scaling_factor,
                    'y': y * self.scaling_factor,
                    'x_old': x_old * self.scaling_factor,
                    'y_old': y_old * self.scaling_factor,
                    'movement_x': movement_x,
                    'movement_y': movement_y,
                    'mtm_dependent': mtm_dependent,
                    'mtm_dependent_x': mtm_dependent_x,
                    'mtm_dependent_y': mtm_dependent_y,
                    'label': str(int(label)),
                    'belongs_to': 'original' if color == 'red' else 'altered',
                    'nearby_point_left': nearby_point_left,
                    'nearby_point_right': nearby_point_right,
                    'nearby_point_left_coords': nearby_point_left_coords,
                    'nearby_point_right_coords': nearby_point_right_coords,
                    'color': color
                })

        mtm_points = row.get('mtm points')
        if not pd.isna(mtm_points):
            # Add the original point
            add_point(x=row['pl_point_x'], y=row['pl_point_y'], 
                    x_old=row['pl_point_x'], y_old=row['pl_point_y'], 
                    label=row['mtm points'], mtm_dependent=row['mtm_dependant'], 
                    mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'], 
                    color='red')

            # Add the altered point, if applicable
            mtm_point_alteration = row.get('mtm_points_alteration')
            if pd.notna(mtm_point_alteration):
                mtm_new_coords = ast.literal_eval(row['new_coordinates'])

                # Safely handle inbetween_points to avoid index errors
                inbetween_points = ast.literal_eval(row.get('mtm_points_in_altered_vertices', '[]'))
                if len(inbetween_points) >= 3:
                    add_point(x=mtm_new_coords[0], y=mtm_new_coords[1], 
                            x_old=row['pl_point_x'], y_old=row['pl_point_y'],
                            label=mtm_point_alteration, mtm_dependent=row['mtm_dependant'], 
                            mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'],
                            movement_x=row['movement_x'], movement_y=row['movement_y'],
                            nearby_point_left=inbetween_points[0]['mtm_point'], nearby_point_right=inbetween_points[2]['mtm_point'],
                            nearby_point_left_coords=inbetween_points[0]['original_coordinates'], 
                            nearby_point_right_coords=inbetween_points[2]['original_coordinates'],
                            color='blue')
                else:
                    # Handle the case where inbetween_points is not as expected
                    add_point(x=mtm_new_coords[0], y=mtm_new_coords[1], 
                            x_old=row['pl_point_x'], y_old=row['pl_point_y'],
                            label=mtm_point_alteration, mtm_dependent=row['mtm_dependant'], 
                            mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'],
                            color='blue', movement_x=row['movement_x'], movement_y=row['movement_y'])

        # Process mtm_points_in_altered_vertices
        mtm_points_in_altered_vertices = ast.literal_eval(row.get('mtm_points_in_altered_vertices', '[]'))
        for mtm_info in mtm_points_in_altered_vertices:
            original_coords = mtm_info.get('original_coordinates')
            altered_coords = mtm_info.get('altered_coordinates')
            mtm_label = mtm_info.get('mtm_point')

            # Add the original coordinates if present
            if original_coords and original_coords[0] is not None and original_coords[1] is not None:
                add_point(x=original_coords[0], y=original_coords[1], 
                        x_old=original_coords[0], y_old=original_coords[1],
                        label=mtm_label, mtm_dependent=row['mtm_dependant'], 
                        mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'], 
                        color='red')
                    
            if altered_coords and altered_coords[0] is not None and altered_coords[1] is not None:
                # Safely handle inbetween_points again
                if len(inbetween_points) >= 3:
                    add_point(altered_coords[0], altered_coords[1], 
                            original_coords[0], original_coords[1], 
                            mtm_label, row['mtm_dependant'], mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'], 
                            movement_x=row['movement_x'], movement_y=row['movement_y'],
                            nearby_point_left=inbetween_points[0]['mtm_point'], nearby_point_right=inbetween_points[2]['mtm_point'],
                            nearby_point_left_coords=inbetween_points[0]['original_coordinates'], 
                            nearby_point_right_coords=inbetween_points[2]['original_coordinates'],
                            color='blue')
                else:
                    add_point(altered_coords[0], altered_coords[1], 
                            original_coords[0], original_coords[1], 
                            mtm_label, row['mtm_dependant'], mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'], 
                            color='blue', movement_x=row['movement_x'], movement_y=row['movement_y'])

    def create_and_save_grid(self, filename, num_squares_x=10, num_squares_y=10, output_dir="../data/output_graphs/plots/"):
        """
        Creates a grid with 1x1 inch squares and saves it to a file.
        """
        # Ensure the calibration directory exists
        calibration_dir = os.path.join(output_dir, "calibration")
        os.makedirs(calibration_dir, exist_ok=True)

        # Remove special characters from the filename
        filename = filename.replace('#', '').replace(' ', '_')

        # Construct the full paths for each file format
        full_filename = os.path.join(calibration_dir, filename)
        png_filename = f"{full_filename}.png"
        svg_filename = f"{full_filename}.svg"
        hpgl_filename = f"{full_filename}.hpgl"
        dxf_filename = f"{full_filename}.dxf"

        # Set the figure size to match the number of squares in inches
        fig_width_inch = num_squares_x
        fig_height_inch = num_squares_y
        fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

        # Set the limits of the plot to match the number of squares
        ax.set_xlim(0, num_squares_x)
        ax.set_ylim(0, num_squares_y)

        # Set the aspect of the plot to be equal
        ax.set_aspect('equal')

        # Enable grid
        ax.grid(True)

        # Customize the grid to have 1-inch spacing
        ax.set_xticks(np.arange(0, num_squares_x + 1, 1))  # 1-inch spacing on the x-axis
        ax.set_yticks(np.arange(0, num_squares_y + 1, 1))  # 1-inch spacing on the y-axis

        # Hide tick labels for a clean grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Remove the outer bounding box completely
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Adjust the layout to ensure the grid fits the entire plot area
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the plot as a PNG file
        plt.savefig(png_filename, dpi=self.dpi, format='png', bbox_inches='tight', pad_inches=0)

        # Save the plot as an SVG file using the modified save_plot_as_svg method
        self.data_processing_utils.save_plot_as_svg(fig, ax, svg_width=fig_width_inch, svg_height=fig_height_inch, output_svg_path=svg_filename)

        # Convert the SVG to HPGL and DXF if needed
        self.data_processing_utils.svg_to_hpgl(svg_filename, hpgl_filename)
        self.data_processing_utils.svg_to_dxf(svg_filename, dxf_filename)

        print(f"Grid saved as {png_filename}, {svg_filename}, {hpgl_filename}, and {dxf_filename}")
    
    def prepare_plot_data(self, output_dir="../data/output_tables/"):
        plot_data = self.initialize_plot_data()
        
        df = self.df.copy()
        vertices_df = self.vertices_df.copy()

        # Fix vertices df below
        for _, row in df.iterrows():
            try:
                #self.process_vertices(row, plot_data, self.scaled_unique_vertices)  
                self.process_altered_vertices(row, plot_data, self.scaled_altered_vertices)
                self.process_altered_vertices_reduced(row, plot_data, self.scaled_altered_vertices_reduced)
                self.process_mtm_points(row, plot_data)

            except Exception as e:
                continue
        
        for _, row in vertices_df.iterrows():
            try:
                self.process_vertices(row, plot_data, self.scaled_unique_vertices)  
            except Exception as e:
                continue
        
        # Todo: Fix this
        plot_data['unique_vertices'] = self.scaled_unique_vertices
        #plot_data['altered_vertices'] = scaled_altered_vertices
        #plot_data['altered_vertices_reduced'] = scaled_altered_vertices_reduced

        self.save_plot_data(plot_data, output_dir)

    def save_plot_data(self, plot_data, output_dir):
        df_unique = pd.DataFrame({
            'unique_vertices': plot_data['unique_vertices'],
            'unique_vertices_x': plot_data['unique_vertices_xs'],
            'unique_vertices_y': plot_data['unique_vertices_ys'],
        })

        df_altered = pd.DataFrame({
            'altered_vertices': plot_data['altered_vertices'],
            'altered_vertices_x': plot_data['altered_vertices_xs'],
            'altered_vertices_y': plot_data['altered_vertices_ys'],
            'altered_vertices_reduced': plot_data['altered_vertices_reduced'],
            'altered_vertices_reduced_x': plot_data['altered_vertices_reduced_xs'],
            'altered_vertices_reduced_y': plot_data['altered_vertices_reduced_ys']
        })

        df_mtm_points = pd.DataFrame({
            'mtm_points': plot_data['mtm_points']
        })

        plot_df = pd.concat([df_unique, df_altered, df_mtm_points], axis=1)
        plot_df.columns = ['unique_vertices', 'unique_vertices_x', 'unique_vertices_y', 
                           'altered_vertices', 'altered_vertices_x', 'altered_vertices_y',
                           'altered_vertices_reduced', 'altered_vertices_reduced_x', 'altered_vertices_reduced_y',
                           'mtm_points']
        
        plot_df = plot_df.drop_duplicates(subset=["mtm_points"])

        output_path = os.path.join(output_dir, "unique_vertices.xlsx")
        plot_df.to_excel(output_path, index=False)
        print(f"Unique vertices saved to {output_path}")
        self.plot_df = plot_df

    def plot_polylines_table(self, output_dir="../data/output_graphs/plots/"):
        piece_dir = os.path.join(output_dir, self.piece_name)
        if not os.path.exists(piece_dir):
            os.makedirs(piece_dir, exist_ok=True)
        svg_dir = os.path.join(piece_dir, "svg")
        hpgl_dir = os.path.join(piece_dir, "hpgl")
        dxf_dir = os.path.join(piece_dir, "dxf")
        png_dir = os.path.join(piece_dir, "png")
        os.makedirs(svg_dir, exist_ok=True)
        os.makedirs(hpgl_dir, exist_ok=True)
        os.makedirs(dxf_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        
        if self.grid:
            sns.set(style="whitegrid")
        
        if self.plot_actual_size:
            xs_all = [x for xs in self.plot_df['unique_vertices_x'].dropna() for x in xs]
            ys_all = [y for ys in self.plot_df['unique_vertices_y'].dropna() for y in ys]
            self.width = max(xs_all) - min(xs_all)
            self.height = max(ys_all) - min(ys_all)
            print(f"Calculated width: {self.width}, height: {self.height}")

            xs_alt = [x for xs in self.plot_df['altered_vertices_x'].dropna() for x in xs]
            ys_alt = [y for ys in self.plot_df['altered_vertices_y'].dropna() for y in ys]

            xs_alt_reduced = [x for xs in self.plot_df['altered_vertices_reduced_x'].dropna() for x in xs]
            ys_alt_reduced = [y for ys in self.plot_df['altered_vertices_reduced_y'].dropna() for y in ys]

            # Combine all xs and ys lists
            all_xs = xs_all + xs_alt + xs_alt_reduced
            all_ys = ys_all + ys_alt + ys_alt_reduced

            if len(all_xs) > 1 and len(all_ys) > 1:
                width_alt = max(all_xs) - min(all_xs)
                height_alt = max(all_ys) - min(all_ys)

            else:
                # Fallback to original logic if there's only one point or no points
                width_alt = all_xs[0] if len(all_xs) == 1 else self.width
                height_alt = all_ys[0] if len(all_ys) == 1 else self.height

            # Check if the altered dimensions are larger than the initial ones
            width_alt = max(self.width, width_alt)
            height_alt = max(self.height, height_alt)

            if self.override_dpi:
                self.dpi = self.override_dpi
            else:
                self.dpi = self.calculate_dpi(
                    self.display_resolution_width, 
                    self.display_resolution_height, 
                    self.display_size
                )          
            print(f"Calculated DPI: {self.dpi}")
        else:
            if self.override_dpi:
                self.dpi = self.override_dpi
            print(f"Default width: {self.width}, height: {self.height}")
            print(f"Default DPI: {self.dpi}")

        fig_combined, ax_combined = plt.subplots(figsize=(self.width, self.height))
        fig_vertices, ax_vertices = plt.subplots(figsize=(self.width, self.height))
        fig_alteration, ax_alteration = plt.subplots(figsize=(self.width, self.height))

        self.plot_combined(ax_combined)
        self.plot_only_vertices(ax_vertices)
        self.plot_alteration_table(output_dir=piece_dir, fig=fig_alteration, ax=ax_alteration)

        combined_filename = f"combined_plot_{self.piece_name}"
        output_combined_path_png = os.path.join(png_dir, f"{combined_filename}.png")
        output_combined_path_svg = os.path.join(svg_dir, f"{combined_filename}.svg")
        output_combined_path_hpgl = os.path.join(hpgl_dir, f"{combined_filename}.hpgl")
        output_combined_path_dxf = os.path.join(dxf_dir, f"{combined_filename}.dxf")

        fig_combined.savefig(output_combined_path_png, dpi=self.dpi, bbox_inches='tight')
        self.data_processing_utils.save_plot_as_svg(fig_combined, ax_combined, width_alt, height_alt, output_combined_path_svg, add_labels=False)
        self.data_processing_utils.svg_to_hpgl(output_combined_path_svg, output_combined_path_hpgl)
        self.data_processing_utils.svg_to_dxf(output_combined_path_svg, output_combined_path_dxf)
        print(f"Combined Plot Saved To {output_combined_path_png}, {output_combined_path_svg}, {output_combined_path_hpgl}, {output_combined_path_dxf}")

        vertices_filename = f"vertices_plot_{self.piece_name}"
        output_vertices_path_png = os.path.join(png_dir, f"{vertices_filename}.png")
        output_vertices_path_svg = os.path.join(svg_dir, f"{vertices_filename}.svg")
        output_vertices_path_hpgl = os.path.join(hpgl_dir, f"{vertices_filename}.hpgl")
        output_vertices_path_dxf = os.path.join(dxf_dir, f"{vertices_filename}.dxf")

        fig_vertices.savefig(output_vertices_path_png, dpi=self.dpi, bbox_inches='tight')
        self.data_processing_utils.save_plot_as_svg(fig_vertices, ax_vertices, self.width, self.height, output_vertices_path_svg, add_labels=False)
        self.data_processing_utils.svg_to_hpgl(output_vertices_path_svg, output_vertices_path_hpgl)
        self.data_processing_utils.svg_to_dxf(output_vertices_path_svg, output_vertices_path_dxf)
        print(f"Vertices Plot Saved To {output_vertices_path_png}, {output_vertices_path_svg}, {output_vertices_path_hpgl}, {output_vertices_path_dxf}")

        alteration_filename = f"alteration_table_plot_{self.piece_name}"
        output_alteration_path_png = os.path.join(png_dir, f"{alteration_filename}.png")
        output_alteration_path_svg = os.path.join(svg_dir, f"{alteration_filename}.svg")
        output_alteration_path_hpgl = os.path.join(hpgl_dir, f"{alteration_filename}.hpgl")
        output_alteration_path_dxf = os.path.join(dxf_dir, f"{alteration_filename}.dxf")

        fig_alteration.savefig(output_alteration_path_png, dpi=self.dpi, bbox_inches='tight')
        self.data_processing_utils.save_plot_as_svg(fig_alteration, ax_alteration, width_alt, height_alt, output_alteration_path_svg, add_labels=False)
        self.data_processing_utils.svg_to_hpgl(output_alteration_path_svg, output_alteration_path_hpgl)
        self.data_processing_utils.svg_to_dxf(output_alteration_path_svg, output_alteration_path_dxf)
        print(f"Alteration Table Plot Saved To {output_alteration_path_png}, {output_alteration_path_svg}, {output_alteration_path_hpgl}, {output_alteration_path_dxf}")

        plt.close(fig_vertices)
        plt.close(fig_alteration)
        plt.close(fig_combined)

    def plot_only_vertices(self, ax):
        font_size = self.get_font_size()
        marker_size = self.get_marker_size()
        line_width = self.get_line_width()
        label_size = self.get_point_label_size()
        
        for _, row in self.plot_df.iterrows():
            if not pd.isna(row['unique_vertices_x']) and not pd.isna(row['unique_vertices_y']):
                xs, ys = row['unique_vertices_x'], row['unique_vertices_y']
                ax.plot(xs, ys, marker='o', linewidth=line_width, markersize=marker_size, color="#006400", label="Original")
        
        ax.set_title(f'Original Vertices Plot for {self.piece_name}', fontsize=font_size+2)
        ax.set_xlabel('X Coordinate [in]', fontsize=font_size)
        ax.set_ylabel('Y Coordinate [in]', fontsize=font_size)

        self.apply_tick_size(ax)

    def plot_combined(self, ax):
        font_size = self.get_font_size()
        marker_size = self.get_marker_size()
        line_width = self.get_line_width()

        for _, row in self.plot_df.iterrows():
            xs, ys = row['unique_vertices_x'], row['unique_vertices_y']

            xs_alt = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_x'])
            ys_alt = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_y'])

            xs_alt_reduced = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_reduced_x'])
            ys_alt_reduced = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_reduced_y'])

            ax.plot(xs, ys, marker='o', linewidth=line_width, markersize=marker_size, color="#006400", label="Original")
            ax.plot(xs_alt_reduced, ys_alt_reduced, marker='x', linestyle='-.', linewidth=line_width+0.5, markersize=marker_size+3, color="#BA55D3", alpha=0.7, label="Altered Reduced")
            ax.plot(xs_alt, ys_alt, marker='o', linestyle='-', linewidth=line_width, markersize=marker_size, color="#00CED1", alpha=0.85, label="Altered")
            
            #ax.tick_params(axis='both', which='both', labelsize=12)
            ax.set_title(f'Combined Vertices Plot for {self.piece_name}', fontsize=font_size+2)
            ax.set_xlabel('X Coordinate [in]', fontsize=font_size)
            ax.set_ylabel('Y Coordinate [in]', fontsize=font_size)

            # Apply dynamic tick size
            self.apply_tick_size(ax)

        ax.legend(handles=[
            Line2D([0], [0], color="#006400", marker='o', label="Original", linewidth=line_width, markersize=marker_size),
            Line2D([0], [0], color="#00CED1", linestyle='-', marker='o', label="Altered", linewidth=line_width, markersize=marker_size),
            Line2D([0], [0], color="#BA55D3", linestyle='-.', marker='x', label="Altered Reduced", linewidth=line_width+0.5, markersize=marker_size+3)
        ])

    def plot_alteration_table(self, output_dir, fig=None, ax=None):
        font_size = self.get_font_size()
        marker_size = self.get_marker_size()
        line_width = self.get_line_width()

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

        #ax.tick_params(axis='both', which='both', labelsize=12)
        # Apply dynamic tick size
        self.apply_tick_size(ax)

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

    def plot_all_mtm_points(self, ax, row):
        marker_size = self.get_marker_size()
        label_size = self.get_point_label_size()

        for point in row['mtm_points']:
            ax.plot(point['x'], point['y'], 'o', color=point['color'], markersize=marker_size)
            ax.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=label_size)

            # Plot movement for altered points
            #if point['color'] == 'blue':
            #    plt.text(point['x'], point['y'], (point['movement_x'], point['movement_y']), color='black', ha='right', va='center', fontsize=10)  # Moves text slightly to the right

if __name__ == "__main__":
    input_table_path="../data/output_tables/processed_alterations/1LTH-FULL_SQUARE-12BY12-INCH.csv"
    input_vertices_path = "../data/output_tables/processed_vertices/processed_vertices_SQUARE-12BY12-INCH.csv"

    #input_table_path="../data/output_tables/processed_alterations/4-WAIST_LGFG-SH-01-CCB-FO.csv"
    #input_vertices_path = "../data/output_tables/processed_vertices/processed_vertices_LGFG-SH-01-CCB-FO.csv"

    #input_table_path="../data/output_tables/processed_alterations/1LTH-FULL_CIRCLE-12BY12-INCH.csv"
    #input_vertices_path = "../data/output_tables/processed_vertices/processed_vertices_CIRCLE-12BY12-INCH.csv"

    #input_table_path="../data/output_tables/processed_alterations/3-COLLAR_LGFG-1648-FG-07P.csv"
    #input_vertices_path = "../data/output_tables/processed_vertices/processed_vertices_LGFG-V2-BC2-FG-08.csv"

    # DPI can only be overriden if plot_actual_size = False
    # Display resolution set for Macbook Pro m2 - change to whatever your screen res is
    visualize_alteration = VisualizeAlteration(input_table_path, input_vertices_path, 
                                               grid=False, plot_actual_size=True, override_dpi=None, display_size=16,
                                               display_resolution_width=3456, display_resolution_height=2234)
    
    # Run this only once (TEST)
    visualize_alteration.create_and_save_grid('10x10_grid')

    # Run these multiple times
    visualize_alteration.prepare_plot_data()
    visualize_alteration.plot_polylines_table()
