import pandas as pd
from matplotlib import pyplot as plt
import ast
import os
import numpy as np

class MakeAlteration:
    def __init__(self, input_table_path):
        self.input_table_path = input_table_path
        self.start_df = self.load_dataframe()
        self.df_alt = self.start_df.copy()

    def load_dataframe(self):
        # Load the Excel file
        start_df = pd.read_excel(self.input_table_path)
        return start_df
    
    def prepare_dataframe(self, df):
        df['pl_point_x'] = pd.to_numeric(df['pl_point_x'], errors='coerce').fillna(0)
        df['pl_point_y'] = pd.to_numeric(df['pl_point_y'], errors='coerce').fillna(0)
        df['movement x'] = df['movement x'].str.replace('%', '').astype(float).fillna(0)
        df['movement y'] = df['movement y'].str.replace('%', '').astype(float).fillna(0)
        df['pl_point_x_modified'] = ""
        df['pl_point_y_modified'] = ""
        df['altered_vertices'] = ""
        return df
    
    def apply_alteration_rules(self):
        df = self.prepare_dataframe(self.df_alt)
        df = df.apply(self.process_alteration_rules, axis=1)
        self.df_alt = df
        self.df_alt.to_excel("../data/output_tables/test_df.xlsx", index=False)

    def process_alteration_rules(self, row):
        if pd.isna(row['alt type']):
            return row
        if row['alt type'] == 'X Y MOVE':
            row = self.apply_xy_move(row)
        elif row['alt type'] in ['CW Ext', 'CCW Ext']:
            if row['mtm points'] != row['first pt']:
                row = self.apply_xy_move(row)
                row = self.apply_extension(row)
        return row

    def apply_xy_move(self, row):
        row['pl_point_x_modified'] = row['pl_point_x'] * (1 + row['movement x'] / 100.0)
        row['pl_point_y_modified'] = row['pl_point_y'] * (1 + row['movement y'] / 100.0)
        return row

    def get_pl_points(self, matching_pt):
        # Find the row where mtm points matches the provided pt
        matching_row = self.start_df.loc[self.start_df['mtm points'] == matching_pt]
        if not matching_row.empty:
            pl_point_x = matching_row['pl_point_x'].values[0]
            pl_point_y = matching_row['pl_point_y'].values[0]
            return pl_point_x, pl_point_y
        return None, None

    def find_vertex_indices(self, vertices_list, first_point, second_point):
        first_index = None
        second_index = None
        for i, vertex in enumerate(vertices_list):
            if vertex == first_point:
                first_index = i
            if vertex == second_point:
                second_index = i
        return first_index, second_index

    def apply_smoothing(self, vertices, start_index, end_index, shift, ascending, change_x, change_y, reverse=False):
        num_points = abs(end_index - start_index) + 1

        x_shift = np.zeros(num_points)
        y_shift = np.zeros(num_points)
        
        if change_x > 0 and change_y == 0:
            x_shift = np.linspace(0, shift[0], num_points) if ascending else np.linspace(shift[0], 0, num_points)
        elif change_y > 0 and change_x == 0:
            y_shift = np.linspace(0, shift[1], num_points) if ascending else np.linspace(shift[1], 0, num_points)
        elif change_x > 0 and change_y > 0:
            x_shift = np.linspace(0, shift[0], num_points) if ascending else np.linspace(shift[0], 0, num_points)
            y_shift = np.linspace(0, shift[1], num_points) if ascending else np.linspace(shift[1], 0, num_points)

        if reverse:
            x_shift = x_shift[::-1]
            y_shift = y_shift[::-1]

        for i in range(num_points):
            index = start_index + i if start_index < end_index else start_index - i
            vertices[index] = (vertices[index][0] + x_shift[i], vertices[index][1] + y_shift[i])

        return vertices

    def apply_extension(self, row):
        mtm_points = row['mtm points']
        first_pt = row['first pt']
        second_pt = row['second pt']

        print("------")
        print(f"Alt Type: {row['alt type']}")
        print(f"MTM Points: {mtm_points}")
        print(f"Rule Name: {row['rule name']}")
        print(f"Point Label: {row['point label']}")
        print(f"Vertex Index: {row['vertex_index']}")

        # Get pl_point_x and pl_point_y for the first pt
        first_point = self.get_pl_points(first_pt)
        second_point = self.get_pl_points(second_pt)

        print(f"First PT: {first_point}")
        print(f"Second PT: {second_point}")

        second_point_altered = (row['pl_point_x_modified'], row['pl_point_y_modified'])
        print(f"Second PT after alteration: {second_point_altered}")

        # Determine the direction of the alteration
        change_x = abs(second_point_altered[0] - second_point[0])
        change_y = abs(second_point_altered[1] - second_point[1])

        print(f"Direction Change: X: {change_x}, Y: {change_y}")

        # Determine if smoothing should be ascending or descending
        ascending = False
        if (second_point_altered[0] > first_point[0] and second_point_altered[1] > first_point[1]) or \
        (second_point_altered[0] < first_point[0] and second_point_altered[1] > first_point[1]):
            ascending = True

        # Find the vertices between first and second PT
        vertices_list = ast.literal_eval(row['vertices'])
        first_index, second_index = self.find_vertex_indices(vertices_list, first_point, second_point)

        if first_index is not None and second_index is not None:
            if row['alt type'] == 'CW Ext' and first_index < second_index:
                row['altered_vertices'] = vertices_list[first_index:second_index + 1]
            elif row['alt type'] == 'CCW Ext' and first_index > second_index:
                row['altered_vertices'] = vertices_list[second_index:first_index + 1]
            else:
                row['altered_vertices'] = "Invalid vertices range"
        else:
            row['altered_vertices'] = "Vertices not found"

        print(f"Altered Vertices Before Smoothing: {row['altered_vertices']}")

        # Collect MTM points that are in the altered vertices and their original indices
        mtm_points_in_altered_vertices = []
        if isinstance(row['altered_vertices'], list):
            # Filter out NaN values from the altered_vertices
            altered_vertices = [v for v in row['altered_vertices'] if not pd.isna(v)]

            print(f"Filtered Altered Vertices: {altered_vertices}")

            # Create a dictionary to map coordinates to their original indices
            vertex_index_map = {v: i for i, v in enumerate(row['altered_vertices'])}

            # Collect MTM points from the start_df
            for _, mtm_row in self.start_df.iterrows():
                mtm_point = (mtm_row['pl_point_x'], mtm_row['pl_point_y'])
                mtm_point_label = mtm_row['mtm points']
                print(f"Checking MTM Point: {mtm_point_label}, Coordinates: {mtm_point}")
                
                if not pd.isna(mtm_point_label) and mtm_point in altered_vertices:
                    # Exclude the MTM points that correspond to the first and second points
                    if mtm_point != first_point and mtm_point != second_point:
                        # Find the original index of the MTM point in the altered vertices list
                        original_index = vertex_index_map.get(mtm_point, 'Not found')

                        mtm_points_in_altered_vertices.append({
                            'mtm_point': mtm_point_label,
                            'original_coordinates': mtm_point,
                            'original_index': original_index
                        })

        # Apply smoothing to the altered vertices
        altered_vertices = row['altered_vertices']
        if isinstance(altered_vertices, list):
            if row['alt type'] == 'CW Ext':
                altered_vertices[-1] = second_point_altered
                shift = (second_point_altered[0] - altered_vertices[-2][0],
                        second_point_altered[1] - altered_vertices[-2][1])
                row['altered_vertices'] = self.apply_smoothing(altered_vertices, 0, len(altered_vertices) - 2, shift, ascending, change_x, change_y)
            elif row['alt type'] == 'CCW Ext':
                altered_vertices[0] = second_point_altered
                shift = (second_point_altered[0] - altered_vertices[1][0],
                        second_point_altered[1] - altered_vertices[1][1])
                row['altered_vertices'] = self.apply_smoothing(altered_vertices, 1, len(altered_vertices) - 1, shift, ascending, change_x, change_y, reverse=True)

        print(f"Altered Vertices After Smoothing: {row['altered_vertices']}")

        # Update the altered_coordinates after smoothing
        if isinstance(row['altered_vertices'], list):
            # Create a dictionary to map original indices to coordinates after smoothing
            altered_index_map = {i: coord for i, coord in enumerate(row['altered_vertices'])}
            
            for point_info in mtm_points_in_altered_vertices:
                original_index = point_info['original_index']
                if original_index is not None:
                    point_info['altered_coordinates'] = altered_index_map.get(original_index, (None, None))

        print("MTM Points within Altered Vertices:")
        for point_info in mtm_points_in_altered_vertices:
            print(f"MTM Point: {point_info['mtm_point']}, Original Coordinates: {point_info['original_coordinates']}, Original Index: {point_info['original_index']}, Altered Coordinates: {point_info['altered_coordinates']}")

        row['mtm_points_in_altered_vertices'] = mtm_points_in_altered_vertices

        print(f"Final Altered Vertices: {row['altered_vertices']}")
        print("")
        return row

    def plot_altered_table(self, output_dir="../data/output_graphs/"):
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(14, 10))  # Set the figure size here

        plotted_vertices = set()
        
        for _, row in self.df_alt.iterrows():
            try:
                if pd.isna(row['vertices']) or row['vertices'] in ['nan', 'None', '', 'NaN']:
                    continue
                
                # Plot original vertices
                vertices_list = ast.literal_eval(row['vertices'])
                unique_vertices = [v for v in vertices_list if v not in plotted_vertices]
                plotted_vertices.update(unique_vertices)
                if unique_vertices:
                    xs, ys = zip(*unique_vertices)
                    plt.plot(xs, ys, marker='o', linewidth=0.5, markersize=5)
                
                # Plot altered vertices
                altered_vertices = row['altered_vertices']
                if isinstance(altered_vertices, list):
                    unique_altered_vertices = [v for v in altered_vertices if v not in plotted_vertices]
                    plotted_vertices.update(unique_altered_vertices)
                    if unique_altered_vertices:
                        altered_xs, altered_ys = zip(*unique_altered_vertices)
                        plt.plot(altered_xs, altered_ys, marker='x', linewidth=0.5, markersize=5, linestyle='--')

                # Add MTM point labels for original vertices
                mtm_points = row['mtm points']
                if not pd.isna(mtm_points):
                    pl_point_x = row['pl_point_x']
                    pl_point_y = row['pl_point_y']
                    mtm_label = str(int(mtm_points))  # Remove trailing zeros
                    plt.text(pl_point_x, pl_point_y, f'{mtm_label}', color='red', fontsize=12)
                    plt.plot(pl_point_x, pl_point_y, 'ro')  # Mark the point in red

                # Plot points altered by 'X Y MOVE'
                if row['alt type'] == 'X Y MOVE':
                    modified_x = row['pl_point_x_modified']
                    modified_y = row['pl_point_y_modified']
                    plt.plot(modified_x, modified_y, 'go', markersize=8)  # Plot the modified points in green
                    plt.text(modified_x, modified_y, str(int(row['mtm points'])), color='green', fontsize=12)  # Label them

                # Add MTM point labels for the altered second point
                second_pt = row['second pt']
                if isinstance(altered_vertices, list) and not pd.isna(second_pt):
                    altered_mtm_label = str(int(second_pt))  # Remove trailing zeros from second MTM point
                    second_point_altered = altered_vertices[0] if row['alt type'] == 'CCW Ext' else altered_vertices[-1]
                    plt.text(second_point_altered[0], second_point_altered[1], altered_mtm_label, color='blue', fontsize=12)
                    plt.plot(second_point_altered[0], second_point_altered[1], 'bo')  # Mark the point in blue

                # Add MTM labels for the points in the altered list
                mtm_points_in_altered_vertices = row.get('mtm_points_in_altered_vertices', [])
                if isinstance(mtm_points_in_altered_vertices, list):
                    for mtm_info in mtm_points_in_altered_vertices:
                        altered_coordinates = mtm_info['altered_coordinates']
                        mtm_label = mtm_info['mtm_point']
                        plt.text(altered_coordinates[0], altered_coordinates[1], str(int(mtm_label)), color='purple', fontsize=12)
                        plt.plot(altered_coordinates[0], altered_coordinates[1], 'mo')  # Mark the point in magenta

            except (ValueError, SyntaxError):
                continue

        plt.title('Polyline Plot for ALT Table', fontsize=16)
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "polyline_plot_test.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    input_table_path = "../data/output_tables/merged_with_rule_subset.xlsx"
    make_alteration = MakeAlteration(input_table_path)
    make_alteration.apply_alteration_rules()
    make_alteration.plot_altered_table()
