from matplotlib import pyplot as plt
import pandas as pd
import os
import ast
import numpy as np
import seaborn as sns
import math
import hashlib

from matplotlib.lines import Line2D

class VisualizeAlteration:
    def __init__(self, input_table_path):
        self.input_table_path = input_table_path
        self.df = self.load_df()
        self.plot_df = ""
        self.scaling_factor = 25.4 # to mm
        self.scaled_unique_vertices = []
        self.scaled_altered_vertices = []
        self.scaled_altered_vertices_reduced = []

    def load_df(self):
        return pd.read_excel(self.input_table_path)

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
            'mtm_points': [],
        }
    
    @staticmethod
    def filter_valid_coordinates(coords):
        if isinstance(coords, (list, tuple)):
            return [coord for coord in coords if not pd.isna(coord)]
        return []
    
    @staticmethod
    def flatten_tuple(nested_tuple):
        flat_list = []
        for item in nested_tuple:
            if isinstance(item, (list, tuple)):
                flat_list.extend(VisualizeAlteration.flatten_tuple(item))
            else:
                flat_list.append(item)
        return flat_list
    
    @staticmethod
    def drop_nans(lst):
        return [item for item in lst if not (isinstance(item, float) and math.isnan(item))]

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
        def add_point(x, y, label, color, movement_x=0., movement_y=0.):
            if x is not None and y is not None:
                plot_data['mtm_points'].append({
                    'x': x * self.scaling_factor,
                    'y': y * self.scaling_factor,
                    'movement_x': movement_x,
                    'movement_y': movement_y,
                    'label': str(int(label)),
                    'color': color,
                    'belongs_to': 'original' if color == 'red' else 'altered'
                })

        mtm_points = row.get('mtm points')
        if not pd.isna(mtm_points):
            # Add the original point
            add_point(row['pl_point_x'], row['pl_point_y'], mtm_points, 'red')

            # Add the altered point, if applicable
            mtm_point_alteration = row.get('mtm_points_alteration')
            if pd.notna(mtm_point_alteration):
                mtm_new_coords = ast.literal_eval(row['new_coordinates'])
                add_point(mtm_new_coords[0], mtm_new_coords[1], mtm_point_alteration, 'blue', 
                        row['movement_x'], row['movement_y'])

        # Process mtm_points_in_altered_vertices
        mtm_points_in_altered_vertices = ast.literal_eval(row.get('mtm_points_in_altered_vertices', '[]'))
        for mtm_info in mtm_points_in_altered_vertices:
            original_coords = mtm_info.get('original_coordinates')
            altered_coords = mtm_info.get('altered_coordinates', (None, None))
            mtm_label = mtm_info.get('mtm_point')

            # Add the original coordinates if present
            if original_coords and original_coords[0] is not None and original_coords[1] is not None:
                add_point(original_coords[0], original_coords[1], mtm_label, 'red')

            # Add the altered coordinates if present
            if altered_coords and altered_coords[0] is not None and altered_coords[1] is not None:
                add_point(altered_coords[0], altered_coords[1], mtm_label, 'blue', 
                        row['movement_x'], row['movement_y'])
    
    def prepare_plot_data(self, output_dir="../data/output_tables/"):
        plot_data = self.initialize_plot_data()
        
        df = self.df.copy()

        for _, row in df.iterrows():
            try:
                self.process_vertices(row, plot_data, self.scaled_unique_vertices)  
                self.process_altered_vertices(row, plot_data, self.scaled_altered_vertices)
                self.process_altered_vertices_reduced(row, plot_data, self.scaled_altered_vertices_reduced)
                self.process_mtm_points(row, plot_data)

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

        output_path = os.path.join(output_dir, "unique_vertices.xlsx")
        plot_df.to_excel(output_path, index=False)
        print(f"Unique vertices saved to {output_path}")
        self.plot_df = plot_df

    def plot_altered_table(self, output_dir="../data/output_graphs/"):

        sns.set(style="whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6))

        plot_df = self.plot_df.copy()

        for _, row in plot_df.iterrows():
    
            self.plot_polylines(ax, row)
            self.plot_mtm_points(ax, plot_df)

        ax.set_title('Polyline Plot for ALT Table', fontsize=16)
        ax.set_xlabel('X Coordinate [mm]', fontsize=14)
        ax.set_ylabel('Y Coordinate [mm]', fontsize=14)

        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.grid(True)

        plt.tight_layout()

        output_path = os.path.join(output_dir, "polyline_plot_test.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Altered Plot Saved To {output_path}")

    def plot_polylines(self, ax, row):
        xs, ys = row['unique_vertices_x'], row['unique_vertices_y']
        xs_alt_reduced = self.filter_valid_coordinates(row['altered_vertices_reduced_x'])
        ys_alt_reduced = self.filter_valid_coordinates(row['altered_vertices_reduced_y'])

        ax.plot(xs, ys, marker='o', linewidth=0.5, markersize=5, color="#006400")
        ax.plot(xs_alt_reduced, ys_alt_reduced, marker='x', linestyle='--', linewidth=1.5, markersize=10, color="#BA55D3", alpha=0.7)
        ax.legend(handles=[
            Line2D([0], [0], color="#006400", label="Original"),
            Line2D([0], [0], color="#BA55D3", linestyle='--', marker='x', label="Altered Reduced"),
            #Line2D([0], [0], color="black", marker='o', linestyle='None', label="Movement (Δx, Δy) [%]"),
        ])

    def plot_mtm_points(self, ax, row):
        for point in row['mtm_points']:
            ax.plot(point['x'], point['y'], 'o', color=point['color'], markersize=5)
            ax.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=12)

            # Plot movement for altered points
            #if point['color'] == 'blue':
            #    plt.text(point['x'], point['y'], (point['movement_x'], point['movement_y']), color='black', ha='right', va='center', fontsize=10)  # Moves text slightly to the right

if __name__ == "__main__":
    input_table_path = "../data/output_tables/processed_alterations.xlsx"
    visualize_alteration = VisualizeAlteration(input_table_path)
    visualize_alteration.prepare_plot_data()
    visualize_alteration.plot_altered_table()
