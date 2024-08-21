from matplotlib import pyplot as plt
import pandas as pd
import os
import ast
import numpy as np
import seaborn as sns
import math
import hashlib
from svgpathtools import svg2paths2
import vpype
import subprocess

from matplotlib.lines import Line2D
from data_processing_utils import DataProcessingUtils

class VisualizeAlteration:
    def __init__(self, input_table_path, input_vertices_path):
        self.data_processing_utils = DataProcessingUtils()
        self.input_table_path = input_table_path
        self.input_vertices_path = input_vertices_path
        self.df = self.data_processing_utils.load_excel(self.input_table_path)
        self.vertices_df = self.data_processing_utils.load_excel(self.input_vertices_path)
        self.plot_df = ""
        #self.scaling_factor = 25.4 # to mm
        self.scaling_factor = 1
        self.scaled_unique_vertices = []
        self.scaled_altered_vertices = []
        self.scaled_altered_vertices_reduced = []

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

    def scale_coordinates(self, xs, ys):
        xs = tuple(x * self.scaling_factor for x in xs)
        ys = tuple(y * self.scaling_factor for y in ys)
        return xs, ys
    
    def process_vertices(self, row, plot_data, scaled_vertices):
        vertices_list = ast.literal_eval(row['original_vertices_reduced'])
        
        if vertices_list:
            xs, ys = zip(*vertices_list)
            xs, ys = self.scale_coordinates(xs, ys)

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
                xs, ys = self.scale_coordinates(xs, ys)
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
                xs, ys = self.scale_coordinates(xs, ys)
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

    def plot_polylines_table(self, output_dir="../data/output_graphs/"):

        sns.set(style="whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6))

        plot_df = self.plot_df.copy()

        for _, row in plot_df.iterrows():
    
            self.plot_polylines(ax, row)
            self.plot_all_mtm_points(ax, plot_df)

        ax.set_title('Polyline Plot for ALT Table', fontsize=16)
        ax.set_xlabel('X Coordinate [mm]', fontsize=14)
        ax.set_ylabel('Y Coordinate [mm]', fontsize=14)

        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.grid(True)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "polygon_debug.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Altered Plot Saved To {output_path}")

    def plot_polylines(self, ax, row):
        # Extract coordinates
        xs, ys = row['unique_vertices_x'], row['unique_vertices_y']

        # Filter and extract valid coordinates for altered and altered reduced vertices
        xs_alt = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_x'])
        ys_alt = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_y'])

        xs_alt_reduced = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_reduced_x'])
        ys_alt_reduced = self.data_processing_utils.filter_valid_coordinates(row['altered_vertices_reduced_y'])

        # Plot original vertices
        ax.plot(xs, ys, marker='o', linewidth=0.5, markersize=5, color="#006400", label="Original")

        # Plot reduced altered vertices (first so it goes under the main altered line)
        ax.plot(xs_alt_reduced, ys_alt_reduced, marker='x', linestyle='-.', linewidth=1.5, markersize=8, color="#BA55D3", alpha=0.7, label="Altered Reduced")

        # Plot altered vertices on top with cyan color
        ax.plot(xs_alt, ys_alt, marker='o', linestyle='-', linewidth=1, markersize=3, color="#00CED1", alpha=0.85, label="Altered")

        # Customize and clean up the legend
        ax.legend(handles=[
            Line2D([0], [0], color="#006400", marker='o', label="Original", linewidth=0.5, markersize=5),
            Line2D([0], [0], color="#00CED1", linestyle='-', marker='o', label="Altered", linewidth=0.5, markersize=3),
            Line2D([0], [0], color="#BA55D3", linestyle='-.', marker='x', label="Altered Reduced", linewidth=1, markersize=8)
        ])

        # Optional: Adjust axes, titles, etc. for better presentation
        ax.set_title("Polyline Visualization")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

    def save_plot_as_svg(self, fig, ax, output_svg_path):
        """
        Saves the given plot to an SVG file.
        """
        # Set the figure size and adjust layout to ensure the plot fits well
        fig.set_size_inches(10, 6)  # Adjust as needed
        plt.tight_layout()

        # Set titles and labels if needed
        ax.set_title('Polyline Plot for ALT Table', fontsize=16)
        ax.set_xlabel('X Coordinate [in]', fontsize=14)
        ax.set_ylabel('Y Coordinate [in]', fontsize=14)

        # Set tick parameters and grid for better visualization
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True)

        # Save the figure as an SVG file
        fig.savefig(output_svg_path, format='svg', bbox_inches='tight')
        plt.close(fig)  # Close the figure after saving

        print(f"SVG plot saved to {output_svg_path}")

    def svg_to_hpgl(self, svg_path, output_hpgl_path):
        """
        Converts an SVG file to HPGL format using Inkscape.
        """
        try:
            command = [
                "inkscape", 
                svg_path, 
                "--export-type=hpgl", 
                f"--export-filename={output_hpgl_path}"
            ]
            subprocess.run(command, check=True)
            print(f"HPGL file saved to {output_hpgl_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting SVG to HPGL: {e}")

    def plot_alteration_table(self, output_dir="../data/output_graphs/"):

        sns.set(style="whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6))

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
                ax.plot(point['x'], point['y'], 'o', color=point['color'], markersize=5)
                ax.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=12)
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
                    mtm_dependent_coords = list(zip(mdp_x, mdp_y))
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
            
        for _, row in plot_df.iterrows():
            point = row['mtm_points']
            label = point['label']
            belongs_to = point['belongs_to']

            if belongs_to == "original":
                if label in mtm_alt_labels:
                    continue
                else:
                    ax.plot(point['x'], point['y'], 'o', color=point['color'], markersize=5)
                    ax.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=12)

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
                    print(f"Points between {dependent_first} and {dependent_last}: {points_between}")
                except:
                    print(f"Point(s) {dependent_first} and {dependent_last} not found")
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
                        ax.plot(x_coords, y_coords, marker='o', linestyle='-', linewidth=1, markersize=3, color="#00CED1", alpha=0.85, label="Altered")

                    if xs_alt and ys_alt:  # Check if xs_alt and ys_alt are not empty
                        xs_alt_list.append(xs_alt)
                        ys_alt_list.append(ys_alt)

        xs_alt_list = self.data_processing_utils.flatten_if_needed(xs_alt_list)
        ys_alt_list = self.data_processing_utils.flatten_if_needed(ys_alt_list)

        # Plot the altered segment
        ax.plot(xs_alt_list, ys_alt_list, marker='o', linestyle='-', linewidth=1, markersize=3, color="#00CED1", alpha=0.85, label="Altered")

        print(coords)

        ax.set_title('Polyline Plot for ALT Table', fontsize=16)
        ax.set_xlabel('X Coordinate [in]', fontsize=14)
        ax.set_ylabel('Y Coordinate [in]', fontsize=14)

        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.grid(True)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "altered_polygon.png")
        output_path_svg = os.path.join(output_dir, "altered_polygon.svg")

        plt.savefig(output_path, dpi=300)
        plt.close()

        self.save_plot_as_svg(fig, ax, output_path_svg)

        print(f"Altered Plot Saved To {output_path}")

        # Convert SVG to HPGL (Inkscape method)
        #output_hpgl_path = os.path.join(output_dir, "altered_polygon.hpgl")
        #self.svg_to_hpgl(output_path_svg, output_hpgl_path)

    def plot_all_mtm_points(self, ax, row):
        for point in row['mtm_points']:
            ax.plot(point['x'], point['y'], 'o', color=point['color'], markersize=5)
            ax.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=12)

            # Plot movement for altered points
            #if point['color'] == 'blue':
            #    plt.text(point['x'], point['y'], (point['movement_x'], point['movement_y']), color='black', ha='right', va='center', fontsize=10)  # Moves text slightly to the right

if __name__ == "__main__":
    input_table_path = "../data/output_tables/processed_alterations_2.xlsx"
    input_vertices_path = "../data/output_tables/vertices_df.xlsx"
    visualize_alteration = VisualizeAlteration(input_table_path, input_vertices_path)
    visualize_alteration.prepare_plot_data()
    visualize_alteration.plot_polylines_table()
    visualize_alteration.plot_alteration_table()