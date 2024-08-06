from matplotlib import pyplot as plt
import pandas as pd
import os
import ast
import numpy as np
from alter_file import MakeAlteration
import seaborn as sns
import math

class VisualizeAlteration:
    def __init__(self, input_table_path):
        self.input_table_path = input_table_path
        self.df = self.load_df()
        self.plot_df = ""

    def load_df(self):
        return pd.read_excel(self.input_table_path)

    def get_plotting_info(self, df):
        plotted_vertices = set()
        plot_info = {'original': [], 'altered': [], 'new': [], 'xy_move_points': [], 'mtm_points': []}

        start_points = df[['pl_point_x', 'pl_point_y']].to_numpy()
        #print("Start Points")
        #print(start_points)

        for _, row in df.iterrows():
            try:
                if pd.isna(row['vertices']) or row['vertices'] in ['nan', 'None', '', 'NaN']:
                    continue
                vertices_list = ast.literal_eval(row['vertices'])
                #print("Vertices List")
                #print(vertices_list)
                unique_vertices = [v for v in vertices_list if tuple(v) not in plotted_vertices]

                plotted_vertices.update(tuple(v) for v in unique_vertices)
                if unique_vertices:
                    reduced_vertices = MakeAlteration.reduce_points(vertices=unique_vertices, threshold=0.1)
                    plot_info['original'].append({'vertices': reduced_vertices})

                #print("Altered Vertices")
                altered_vertices = row['altered_vertices']
                #print(altered_vertices)

                if isinstance(altered_vertices, list):
                    unique_altered_vertices = [altered_vertices[0]] + [v for v in altered_vertices[1:-1] if tuple(v) not in plotted_vertices] + [altered_vertices[-1]]
                    plotted_vertices.update(tuple(v) for v in unique_altered_vertices)
                    if unique_altered_vertices:
                        reduced_altered_vertices = MakeAlteration.reduce_points(vertices=unique_altered_vertices, threshold=0.1)
                        plot_info['altered'].append({'vertices': reduced_altered_vertices})
                        plot_info['new'].append({'vertices': reduced_altered_vertices})

                mtm_points = row['mtm points']
                if not pd.isna(mtm_points):
                    pl_point_x = row['pl_point_x']
                    pl_point_y = row['pl_point_y']
                    if pl_point_x is not None and pl_point_y is not None:
                        plot_info['mtm_points'].append({'x': pl_point_x, 'y': pl_point_y, 'label': str(int(mtm_points)), 'color': 'red', 'belongs_to': 'original'})

                if row['alteration'] == 'X Y MOVE':
                    modified_x = row['pl_point_x_modified']
                    modified_y = row['pl_point_y_modified']
                    if modified_x is not None and modified_y is not None:
                        plot_info['xy_move_points'].append({'x': modified_x, 'y': modified_y, 'label': str(int(row['mtm points'])), 'color': 'blue'})
                    original_y = row['pl_point_y']
                    if modified_y < original_y:
                        mtm_points_in_altered_vertices = row.get('mtm_points_in_altered_vertices', [])
                        if isinstance(mtm_points_in_altered_vertices, list) and len(mtm_points_in_altered_vertices) >= 3:
                            prev_point = mtm_points_in_altered_vertices[0]['original_coordinates']
                            current_point = mtm_points_in_altered_vertices[1]['altered_coordinates']
                            next_point = mtm_points_in_altered_vertices[2]['original_coordinates']
                            if (current_point[0] is not None and current_point[1] is not None and
                                next_point[0] is not None and next_point[1] is not None and
                                prev_point[0] is not None and prev_point[1] is not None):

                                new_xy_move_points = {'line': [(prev_point[0], current_point[0], next_point[0]), (prev_point[1], current_point[1], next_point[1])]}
                                plot_info['xy_move_points'].append(new_xy_move_points)

                second_pt = row['mtm points']
                if isinstance(altered_vertices, list) and not pd.isna(second_pt):
                    altered_mtm_label = str(int(second_pt))
                    second_point_altered = altered_vertices[0] if row['alt type'] == 'CCW Ext' else altered_vertices[-1]
                    if second_point_altered[0] is not None and second_point_altered[1] is not None:
                        plot_info['mtm_points'].append({'x': second_point_altered[0], 'y': second_point_altered[1], 'label': altered_mtm_label, 'color': 'blue', 'belongs_to': 'altered'})

                mtm_points_in_altered_vertices = row.get('mtm_points_in_altered_vertices', [])
                if isinstance(mtm_points_in_altered_vertices, list):
                    for mtm_info in mtm_points_in_altered_vertices:
                        original_coordinates = mtm_info['original_coordinates']
                        altered_coordinates = mtm_info.get('altered_coordinates', (None, None))
                        mtm_label = mtm_info['mtm_point']
                        if original_coordinates[0] is not None and original_coordinates[1] is not None:
                            plot_info['mtm_points'].append({'x': original_coordinates[0], 'y': original_coordinates[1], 'label': str(int(mtm_label)), 'color': 'red', 'belongs_to': 'original'})
                        if altered_coordinates[0] is not None and altered_coordinates[1] is not None:
                            plot_info['mtm_points'].append({'x': altered_coordinates[0], 'y': altered_coordinates[1], 'label': str(int(mtm_label)), 'color': 'blue', 'belongs_to': 'altered'})

                for vertex in vertices_list:
                    vertex_array = np.array(vertex)
                    match = np.all(start_points == vertex_array, axis=1)
                    matching_indices = np.where(match)[0]
                    if matching_indices.size > 0:
                        mtm_label = str(int(df.iloc[matching_indices[0]]['mtm points']))
                        plot_info['mtm_points'].append({'x': vertex[0], 'y': vertex[1], 'label': mtm_label, 'color': 'red', 'belongs_to': 'original'})

            except (ValueError, SyntaxError):
                continue

        df['plot_info'] = [plot_info] * len(df)
        df.to_excel("../data/output_tables/plotting_info.xlsx", index=False)
        return df
    
    def prepare_plot_data(self, output_dir="../data/output_tables/"):
        # Initialize plot_data with lists for unique vertices, xs, and ys
        plot_data = {
            'unique_vertices': [],
            'unique_vertices_xs': [],
            'unique_vertices_ys': [],
            
            'altered_vertices': [],
            'altered_vertices_xs': [],
            'altered_vertices_ys': [],

            'altered_vertices_reduced': [],
            'altered_vertices_reduced_xs': [],
            'altered_vertices_reduced_ys': [],
        }
        
        # Create a copy of the DataFrame to avoid modifying the original data
        df = self.df.copy()
        count = 0

        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():

            # --------------------------------- #
            #### Handle Original Vertices ####
            # --------------------------------- #

            try:         
                # Evaluate the string representation of the list of vertices into an actual list
                vertices_list = ast.literal_eval(row['original_vertices_reduced'])

                # Check if this list of vertices is already in unique_vertices
                if vertices_list not in plot_data['unique_vertices'] and len(vertices_list) != 0:
                    # Append the unique vertices to the list
                    plot_data['unique_vertices'].append(vertices_list)

                    # Unzip the list of vertices into x and y coordinates
                    xs, ys = zip(*vertices_list)
                    
                    plot_data['unique_vertices_xs'].append(xs)
                    plot_data['unique_vertices_ys'].append(ys)

                    count += 1                

            except Exception as e:
                # Handle exceptions and continue with the next row
                continue

            # --------------------------------- #
            #### Handle altered vertices ####
            # --------------------------------- #

            # Alteration Unmodified
            raw_altered = row['altered_vertices']
            if not pd.isna(raw_altered):
                altered_vertices_list = ast.literal_eval(raw_altered)
                if altered_vertices_list not in plot_data['altered_vertices'] and len(altered_vertices_list) != 0:
                    plot_data['altered_vertices'].append(altered_vertices_list)
                    xs_altered, ys_altered = zip(*altered_vertices_list)

                    # Handle single point cases 
                    # TODO: not ideal -- make them already part of the altered list, then remove this condition
                    if len(xs_altered) == 1:
                        plot_data['altered_vertices_xs'].append(xs_altered[0])
                        plot_data['altered_vertices_ys'].append(ys_altered[0])
                    else:
                        plot_data['altered_vertices_xs'].append(xs_altered)
                        plot_data['altered_vertices_ys'].append(ys_altered)

            # Smoothened & Reduced
            raw_altered_reduced = row['altered_vertices_reduced']
            if not pd.isna(raw_altered_reduced):
                altered_vertices_list_reduced = ast.literal_eval(raw_altered_reduced)
                if altered_vertices_list_reduced not in plot_data['altered_vertices_reduced'] and len(altered_vertices_list_reduced) != 0:
                    plot_data['altered_vertices_reduced'].append(altered_vertices_list_reduced)
                    xs_altered_reduced, ys_altered_reduced = zip(*altered_vertices_list_reduced)
                    plot_data['altered_vertices_reduced_xs'].append(xs_altered_reduced)
                    plot_data['altered_vertices_reduced_ys'].append(ys_altered_reduced)
        
            # --------------------------------- #

        print(f"Number of Unique Original Vertices = {count}")

        # Save plot_data to a DataFrame and then to an Excel file
        df_unique = pd.DataFrame({
            'unique_vertices': plot_data['unique_vertices'],
            'unique_vertices_x': plot_data['unique_vertices_xs'],
            'unique_vertices_y': plot_data['unique_vertices_ys'],
        })

        # Includes all smoothing and reducing points
        df_altered = pd.DataFrame({

            'altered_vertices': plot_data['altered_vertices'],
            'altered_vertices_x': plot_data['altered_vertices_xs'],
            'altered_vertices_y': plot_data['altered_vertices_ys'],

            'altered_vertices_reduced': plot_data['altered_vertices_reduced'],
            'altered_vertices_reduced_x': plot_data['altered_vertices_reduced_xs'],
            'altered_vertices_reduced_y': plot_data['altered_vertices_reduced_ys']
        })

        # Concatenate these DataFrames along the columns
        plot_df = pd.concat([df_unique, df_altered], axis=1)
        plot_df.columns = ['unique_vertices', 'unique_vertices_x', 'unique_vertices_y', 
                           'altered_vertices', 'altered_vertices_x', 'altered_vertices_y',
                           'altered_vertices_reduced', 'altered_vertices_reduced_x', 'altered_vertices_reduced_y']

        # Define the output path for the Excel file
        output_path = os.path.join(output_dir, "unique_vertices.xlsx")

        # Save the DataFrame to an Excel file
        plot_df.to_excel(output_path, index=False)
        print(f"Unique vertices saved to {output_path}")

        self.plot_df = plot_df

    def filter_valid_coordinates(coords):
        """Filter out NaN values and ensure coords is a list or tuple."""
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

    def plot_altered_table(self, output_dir="../data/output_graphs/"):

        # Set a Seaborn style for the plot
        sns.set(style="whitegrid")

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Copy the plot data
        plot_df = self.plot_df.copy()

        xs_alt_list, ys_alt_list = [], []
        # Loop through the rows and plot each set of vertices
        for _, row in plot_df.iterrows():
    
            xs = row['unique_vertices_x']
            ys = row['unique_vertices_y']

            xs_alt = row['altered_vertices_x']
            ys_alt = row['altered_vertices_y']

            xs_alt_list.append(xs_alt)
            ys_alt_list.append(ys_alt)

            xs_alt_list = VisualizeAlteration.flatten_tuple(xs_alt_list)
            ys_alt_list = VisualizeAlteration.flatten_tuple(ys_alt_list)

            # Smoothened Reduced
            xs_alt_reduced = row['altered_vertices_reduced_x']
            ys_alt_reduced = row['altered_vertices_reduced_y']      

            # Plot figure
            original_line, = ax.plot(xs, ys, marker='o', linewidth=0.5, markersize=5, color="#006400")
            altered_line, = ax.plot(xs_alt_list, ys_alt_list, marker='o', linewidth=0.5, markersize=5, color="#DAA520")
            altered_line_reduced, = ax.plot(xs_alt_reduced, ys_alt_reduced, marker='o', linewidth=0.5, markersize=5, color="#40E0D0")
        
        print("Altered (Y)")
        print(ys_alt_list)

        # What I got
        x_test = (16.918, 19.794, 20.608, 23.119, 27.527, 27.712, 28.335, 11.385, 14.524, 16.307, 16.912)
        y_test = (17.537499999999998, 17.063633333333332, 16.847199999999997, 16.54733333333333, 16.277749999999997, 15.891316666666667, 15.824883333333332, 15.814166666666665, 14.659866666666666, 14.181716666666667, 13.873)

        # From old code

        # Altered
        altered_vertices_ref = [
                (9.74, 14.2044),
                (9.74, 14.2044),
                (11.385, 14.657),
                (14.524, 15.239227272727272),
                (16.918, 15.78375),
                (16.918, 15.78375),
                (19.794, 15.447434210526316),
                (20.608, 15.276776315789474),
                (23.119, 15.06846052631579),
                (27.527, 14.913315789473684),
                (28.335, 14.552)
            ]
        
        print("Altered Vertices Ref (Y)")

        #xs_old = (11.385, 14.524, 16.918)
        #ys_old = (14.657, 15.239227272727272, 15.78375)

        x_old_list, y_old_list = zip(*altered_vertices_ref)
        print(y_old_list)

        altered_ref, = ax.plot(x_old_list, y_old_list, marker='o', linewidth=0.5, markersize=5, color="#333FFF")

        original_line.set_label("Original")
        altered_ref.set_label("Altered Ref")
        altered_line.set_label("Altered")
        altered_line_reduced.set_label("Altered Reduced")

        ax.legend()

        # Set the title and labels with Seaborn's font scale
        ax.set_title('Polyline Plot for ALT Table', fontsize=16)
        ax.set_xlabel('X Coordinate', fontsize=14)
        ax.set_ylabel('Y Coordinate', fontsize=14)

        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Enable the grid
        ax.grid(True)

        # Adjust the layout for tight fit
        plt.tight_layout()

        # Save the plot to a file
        output_path = os.path.join(output_dir, "polyline_plot_test.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Altered Plot Saved To {output_path}")

if __name__ == "__main__":
    input_table_path = "../data/output_tables/processed_alterations.xlsx"
    visualize_alteration = VisualizeAlteration(input_table_path)
    visualize_alteration.prepare_plot_data()
    visualize_alteration.plot_altered_table()