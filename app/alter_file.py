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
        self.df_alt.to_excel("../data/output_tables/processed_alterations.xlsx", index=False)

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
    
    def find_closest_points(self, mtm_point, current_x, current_y):
        df_existing_mtm = self.start_df.dropna(subset=['mtm points'])
        df_existing_mtm['distance'] = np.sqrt((df_existing_mtm['pl_point_x'] - current_x)**2 + (df_existing_mtm['pl_point_y'] - current_y)**2)
        df_sorted = df_existing_mtm[df_existing_mtm['mtm points'] != mtm_point].sort_values(by='distance')
        prev_point = df_sorted.iloc[0] if not df_sorted.empty else None
        next_point = df_sorted.iloc[1] if len(df_sorted) > 1 else None
        return prev_point, next_point
    
    def apply_xy_move(self, row):
        row['pl_point_x_modified'] = row['pl_point_x'] * (1 + row['movement x'] / 100.0)
        row['pl_point_y_modified'] = row['pl_point_y'] * (1 + row['movement y'] / 100.0)
        mtm_point = row['mtm points']
        prev_point, next_point = self.find_closest_points(mtm_point, row['pl_point_x'], row['pl_point_y'])
        prev_coordinates = (prev_point['pl_point_x'], prev_point['pl_point_y']) if prev_point is not None else (None, None)
        next_coordinates = (next_point['pl_point_x'], next_point['pl_point_y']) if next_point is not None else (None, None)
        altered_coordinates = (row['pl_point_x_modified'], row['pl_point_y_modified'])
        mtm_points_in_altered_vertices = [
            {'mtm_point': prev_point['mtm points'] if prev_point is not None else None, 'original_coordinates': prev_coordinates},
            {'mtm_point': mtm_point, 'original_coordinates': (row['pl_point_x'], row['pl_point_y']), 'altered_coordinates': altered_coordinates},
            {'mtm_point': next_point['mtm points'] if next_point is not None else None, 'original_coordinates': next_coordinates}
        ]
        row['mtm_points_in_altered_vertices'] = mtm_points_in_altered_vertices

        # Include the XY move point in the altered vertices
        if isinstance(row['altered_vertices'], str) and row['altered_vertices'] in ['', 'nan', 'None', 'NaN']:
            row['altered_vertices'] = [altered_coordinates]
        else:
            altered_vertices = ast.literal_eval(row['altered_vertices']) if isinstance(row['altered_vertices'], str) else row['altered_vertices']
            altered_vertices.append(altered_coordinates)
            row['altered_vertices'] = altered_vertices

        return row

    def get_pl_points(self, matching_pt):
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
        first_point = self.get_pl_points(first_pt)
        second_point = self.get_pl_points(second_pt)
        second_point_altered = (row['pl_point_x_modified'], row['pl_point_y_modified'])
        change_x = abs(second_point_altered[0] - second_point[0])
        change_y = abs(second_point_altered[1] - second_point[1])
        ascending = False
        if (second_point_altered[0] > first_point[0] and second_point_altered[1] > first_point[1]) or \
        (second_point_altered[0] < first_point[0] and second_point_altered[1] > first_point[1]):
            ascending = True
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
        mtm_points_in_altered_vertices = []
        if isinstance(row['altered_vertices'], list):
            altered_vertices = [v for v in row['altered_vertices'] if not pd.isna(v)]
            vertex_index_map = {v: i for i, v in enumerate(row['altered_vertices'])}
            for _, mtm_row in self.start_df.iterrows():
                mtm_point = (mtm_row['pl_point_x'], mtm_row['pl_point_y'])
                mtm_point_label = mtm_row['mtm points']
                if not pd.isna(mtm_point_label) and mtm_point in altered_vertices:
                    if mtm_point != first_point and mtm_point != second_point:
                        original_index = vertex_index_map.get(mtm_point, 'Not found')
                        mtm_points_in_altered_vertices.append({
                            'mtm_point': mtm_point_label,
                            'original_coordinates': mtm_point,
                            'original_index': original_index
                        })
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
        if isinstance(row['altered_vertices'], list):
            altered_index_map = {i: coord for i, coord in enumerate(row['altered_vertices'])}
            for point_info in mtm_points_in_altered_vertices:
                original_index = point_info['original_index']
                if original_index is not None:
                    point_info['altered_coordinates'] = altered_index_map.get(original_index, (None, None))
        row['mtm_points_in_altered_vertices'] = mtm_points_in_altered_vertices
        return row
    
    def reduce_points(self, vertices, threshold=0.1):
        if isinstance(vertices, list) and len(vertices) > 2:
            points = np.array(vertices)
            reduced_points = self.visvalingam_whyatt(points, threshold)
            return reduced_points
        return vertices
    
    def visvalingam_whyatt(self, points, threshold):
        def calculate_area(a, b, c):
            return 0.5 * abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        
        def filter_points(points, threshold):
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
        points = [list(point) for point in points]
        simplified_points = filter_points(points, threshold)
        return simplified_points

    def get_unique_vertices(self):
        unique_vertices_set = set()
        unique_vertices = []
        for _, row in self.df_alt.iterrows():
            if pd.isna(row['vertices']) or row['vertices'] in ['nan', 'None', '', 'NaN']:
                continue
            vertices_list = ast.literal_eval(row['vertices'])
            for vertex in vertices_list:
                if vertex not in unique_vertices_set:
                    unique_vertices_set.add(vertex)
                    unique_vertices.append(vertex)
        return unique_vertices

    def get_plotting_info(self):
        plotted_vertices = set()
        plot_info = {'original': [], 'altered': [], 'xy_move_points': [], 'mtm_points': []}

        for _, row in self.df_alt.iterrows():
            try:
                if pd.isna(row['vertices']) or row['vertices'] in ['nan', 'None', '', 'NaN']:
                    continue
                vertices_list = ast.literal_eval(row['vertices'])
                unique_vertices = [v for v in vertices_list if v not in plotted_vertices]
                plotted_vertices.update(unique_vertices)
                if unique_vertices:
                    reduced_vertices = self.reduce_points(unique_vertices, threshold=0.1)
                    plot_info['original'].append(reduced_vertices)

                altered_vertices = row['altered_vertices']
                if isinstance(altered_vertices, list):
                    unique_altered_vertices = [altered_vertices[0]] + [v for v in altered_vertices[1:-1] if v not in plotted_vertices] + [altered_vertices[-1]]
                    plotted_vertices.update(unique_altered_vertices)
                    if unique_altered_vertices:
                        reduced_altered_vertices = self.reduce_points(unique_altered_vertices, threshold=0.1)
                        plot_info['altered'].append(reduced_altered_vertices)

                # Unmodified MTM points
                mtm_points = row['mtm points']
                if not pd.isna(mtm_points):
                    pl_point_x = row['pl_point_x']
                    pl_point_y = row['pl_point_y']
                    if pl_point_x is not None and pl_point_y is not None:
                        plot_info['mtm_points'].append({'x': pl_point_x, 'y': pl_point_y, 'label': str(int(mtm_points)), 'color': 'red'})
                
                # Modified mtm points
                if row['alt type'] == 'X Y MOVE':
                    modified_x = row['pl_point_x_modified']
                    modified_y = row['pl_point_y_modified']
                    if modified_x is not None and modified_y is not None:
                        plot_info['xy_move_points'].append({'x': modified_x, 'y': modified_y, 'label': str(int(row['mtm points'])), 'color': 'blue'})
                    original_y = row['pl_point_y']
                    if modified_y < original_y:
                        mtm_points_in_altered_vertices = row.get('mtm_points_in_altered_vertices', [])
                        if isinstance(mtm_points_in_altered_vertices, list) and len(mtm_points_in_altered_vertices) >= 2:
                            current_point = mtm_points_in_altered_vertices[1]['altered_coordinates']
                            next_point = mtm_points_in_altered_vertices[2]['original_coordinates']
                            if current_point[0] is not None and current_point[1] is not None and next_point[0] is not None and next_point[1] is not None:
                                plot_info['xy_move_points'].append({'line': [(current_point[0], next_point[0]), (current_point[1], next_point[1])]})

                second_pt = row['second pt']
                if isinstance(altered_vertices, list) and not pd.isna(second_pt):
                    altered_mtm_label = str(int(second_pt))
                    second_point_altered = altered_vertices[0] if row['alt type'] == 'CCW Ext' else altered_vertices[-1]
                    if second_point_altered[0] is not None and second_point_altered[1] is not None:
                        plot_info['mtm_points'].append({'x': second_point_altered[0], 'y': second_point_altered[1], 'label': altered_mtm_label, 'color': 'blue'})

                mtm_points_in_altered_vertices = row.get('mtm_points_in_altered_vertices', [])
                if isinstance(mtm_points_in_altered_vertices, list):
                    for mtm_info in mtm_points_in_altered_vertices:
                        original_coordinates = mtm_info['original_coordinates']
                        altered_coordinates = mtm_info.get('altered_coordinates', (None, None))
                        mtm_label = mtm_info['mtm_point']
                        if original_coordinates[0] is not None and original_coordinates[1] is not None:
                            plot_info['mtm_points'].append({'x': original_coordinates[0], 'y': original_coordinates[1], 'label': str(int(mtm_label)), 'color': 'red'})
                        if altered_coordinates[0] is not None and altered_coordinates[1] is not None:
                            plot_info['mtm_points'].append({'x': altered_coordinates[0], 'y': altered_coordinates[1], 'label': str(int(mtm_label)), 'color': 'blue'})
            except (ValueError, SyntaxError):
                continue
        
        self.df_alt['plot_info'] = [plot_info] * len(self.df_alt)
        self.df_alt.to_excel("../data/output_tables/plotting_info.xlsx", index=False)

    def plot_altered_table(self, output_dir="../data/output_graphs/"):
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(14, 10))

        plot_info = self.df_alt['plot_info'].iloc[0]

        for original_vertices in plot_info['original']:
            xs, ys = zip(*original_vertices)
            plt.plot(xs, ys, marker='o', linewidth=0.5, markersize=5)

        for altered_vertices in plot_info['altered']:
            altered_xs, altered_ys = zip(*altered_vertices)
            plt.plot(altered_xs, altered_ys, marker='x', linewidth=0.5, markersize=5, linestyle='--', color='cyan')

        for point in plot_info['mtm_points']:
            plt.plot(point['x'], point['y'], 'o', color=point['color'], markersize=8)
            plt.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=12)

        for line in plot_info['xy_move_points']:
            if 'line' in line:
                plt.plot(line['line'][0], line['line'][1], marker='x', linewidth=0.5, markersize=5, linestyle='--', color='cyan')
            else:
                plt.plot(line['x'], line['y'], 'bo', markersize=8)
                plt.text(line['x'], line['y'], line['label'], color='blue', fontsize=12)

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

    def plot_final_altered_table(self, output_dir="../data/output_graphs/"):
        plot_info = self.df_alt['plot_info'].iloc[0]

        for altered_vertices in plot_info['altered']:
            altered_xs, altered_ys = zip(*altered_vertices)
            plt.plot(altered_xs, altered_ys, marker='o', linewidth=0.5, markersize=5, linestyle='--', color='blue')

        for line in plot_info['xy_move_points']:
            if 'line' in line:
                plt.plot(line['line'][0], line['line'][1], marker='x', linewidth=0.5, markersize=5, linestyle='--', color='cyan')
            else:
                plt.plot(line['x'], line['y'], 'bo', markersize=8)
                plt.text(line['x'], line['y'], line['label'], color='blue', fontsize=12)

        # TODO: Remove original points for the altered versions
        for point in plot_info['mtm_points']:
            plt.plot(point['x'], point['y'], 'o', color=point['color'], markersize=8)
            plt.text(point['x'], point['y'], point['label'], color=point['color'], fontsize=12)

        plt.title('Polyline Plot Alteration', fontsize=16)
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join(output_dir, "polyline_altered.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    input_table_path = "../data/output_tables/merged_with_rule_subset.xlsx"
    make_alteration = MakeAlteration(input_table_path)
    make_alteration.apply_alteration_rules()
    make_alteration.get_plotting_info()
    make_alteration.plot_altered_table()
    make_alteration.plot_final_altered_table()
