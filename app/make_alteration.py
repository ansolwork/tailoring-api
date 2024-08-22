import pandas as pd
import ast
import numpy as np
from smoothing import SmoothingFunctions  # Import the SmoothingFunctions class
from data_processing_utils import DataProcessingUtils
import os

# Further notes:
# Vertices have to be sorted (by x-coordinate) before smoothing is applied
# Vertices need to not have duplicate coordinate entries 

# Done: Fixed coordinates
# Next -> ?

# TODO: Optional, order columns

class MakeAlteration:
    def __init__(self, input_table_path, input_vertices_path, 
                 piece_name, save_folder, file_format):
        
        self.processing_utils = DataProcessingUtils()

        self.input_table_path = input_table_path
        self.input_vertices_path = input_vertices_path

        self.piece_name = piece_name
        self.start_df = self.processing_utils.load_csv(input_table_path)
        self.start_df = self.filter_by_piece_name()

        self.vertices_df = self.processing_utils.load_csv(input_vertices_path)

        self.df_alt = ""
        self.total_alt = []
        self.vertices_list = []
        self.scaling_factor = 25.4 # to mm

        self.save_folder = save_folder
        self.file_format = file_format

    def filter_by_piece_name(self):
        return self.start_df[self.start_df['piece_name'] == self.piece_name]

    def prepare_dataframe(self, df):
        df['pl_point_x'] = pd.to_numeric(df['pl_point_x'], errors='coerce').fillna(0)
        df['pl_point_y'] = pd.to_numeric(df['pl_point_y'], errors='coerce').fillna(0)
        df['maximum_movement_inches_positive'] = pd.to_numeric(df['maximum_movement_inches_positive'], errors='coerce').fillna(0)
        df['maximum_movement_inches_negative'] = pd.to_numeric(df['maximum_movement_inches_negative'], errors='coerce').fillna(0)
        df['minimum_movement_inches_positive'] = pd.to_numeric(df['minimum_movement_inches_positive'], errors='coerce').fillna(0)
        df['minimum_movement_inches_negative'] = pd.to_numeric(df['minimum_movement_inches_negative'], errors='coerce').fillna(0)
        df['movement_x'] = df['movement_x'].astype(str).str.replace('%', '').astype(float).fillna(0)
        df['movement y'] = df['movement_y'].astype(str).str.replace('%', '', regex=False).astype(float).fillna(0)
        df['pl_point_x_modified'] = ""
        df['pl_point_y_modified'] = ""
        df['altered_vertices'] = ""
        df['distance_euq'] = ""
        return df
    
    def apply_alteration_rules(self, custom_alteration=False):
        alteration_df = self.prepare_dataframe(self.start_df.copy())
        self.vertices_df = self.vertices_df.copy().apply(self.reduce_original_vertices, axis=1)
        self.vertices_df.to_excel("../data/output_tables/vertices_df.xlsx", index=False)
        
        # Extract vertices column from DataFrame as a list of strings
        vertices_string_list = self.vertices_df['vertices'].tolist()

        # Convert the string representation of lists to actual lists
        vertices_nested_list = [ast.literal_eval(vertices) for vertices in vertices_string_list]

        # Flatten the list of lists into a single list of coordinates
        flattened_vertices_list = [vertex for sublist in vertices_nested_list for vertex in sublist]

        # Remove duplicates while preserving order
        self.vertices_list = self.processing_utils.remove_duplicates_preserve_order(flattened_vertices_list)

        print("Start Alteration Data")
        print(self.start_df)

        # Apply alteration rules
        alteration_df = alteration_df.apply(self.process_alteration_rules, axis=1)
        self.vertex_smoothing()
        self.reduce_and_smooth_vertices()

        # Merge and save
        total_alt_df = pd.DataFrame(self.total_alt)
        merged_df = self.merge_with_original_df(alteration_df, total_alt_df)
        merged_df = self.get_mtm_dependent_coords(merged_df)

        # Save final table
        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)
        save_filepath = f"{self.save_folder}/altered_{self.piece_name}{self.file_format}"
        if self.file_format == '.xlsx':
            merged_df.to_excel(save_filepath, index=False)
        elif self.file_format == '.csv':
            merged_df.to_csv(save_filepath, index=False)
    
    def process_alteration_rules(self, row):
        if pd.isna(row['alteration_type']):
            return row
        
        def update_or_add_alt_set(alt_set):
            for existing in self.total_alt:
                if existing['mtm_point'] == alt_set['mtm_point']:
                    if not isinstance(existing['mtm_dependant'], list):
                        existing['mtm_dependant'] = [alt_set['mtm_dependant'], existing['mtm_dependant']]
                    else:
                        existing['mtm_dependant'] = existing['mtm_dependant'] + [alt_set['mtm_dependant']]

                    existing['alteration'] = [alt_set['alteration'], existing['alteration']]
                    # Update existing entry
                    existing['movement_x'] += alt_set['movement_x']
                    existing['movement_y'] += alt_set['movement_y']
                    
                    # Recalculate new coordinates based on the summed movements
                    # Assuming coordinates are in decimal form: e.g. 12.5 % <-> 0.125
                    # Multiplication: (old_coordinates) * (1 + 0.125), to apply movement
                    # Base unit already in inches. Forget above
                    existing['new_coordinates'] = (
                        existing['old_coordinates'][0] * (1 + existing['movement_x']),
                        existing['old_coordinates'][1] + (1 + existing['movement_y'])
                    )
                    
                    existing['mtm_points_in_altered_vertices'] = alt_set['mtm_points_in_altered_vertices'] + existing['mtm_points_in_altered_vertices']

                    # Don't forget to include all the altered vertices
                    print("Altered Vertices Failure")
                    print(altered_vertices)
                    altered_vertices_copy = altered_vertices.copy()
                    update_altered_vertices = altered_vertices_copy + existing['altered_vertices']

                    if not existing.get("split_vertices"):
                        existing["split_vertices"] = {}

                    if alt_set['alteration'] not in existing["split_vertices"]:
                        existing["split_vertices"][alt_set['alteration']] = []

                    existing["split_vertices"][alt_set['alteration']].extend(altered_vertices_copy)

                    existing['altered_vertices'] = update_altered_vertices

                    return existing  # Return the updated existing entry
                
            # If no existing entry found, add new entry
            alt_set["split_vertices"] = {alt_set['alteration']: alt_set['altered_vertices']}
            self.total_alt.append(alt_set)
            return alt_set
        
        # The issue is that it is not enough to filter by alteration_type, since it can have other mtm points
        if row['alteration_type'] == 'X Y MOVE' and pd.notna(row['mtm points']):
            print("---------")
            print("Alteration: ")
            print(row['alteration_type'])
            print("---------")
            row = self.apply_xy_move(row)

            alt_set = {"mtm_point": int(row['mtm points']),
                       "mtm_dependant" : int(row['mtm_dependent']),
                    "alteration": row['alteration_type'],
                    "movement_x": row['movement_x'],
                    "movement_y": row['movement_y'],
                    "old_coordinates": (row['pl_point_x'], row['pl_point_y']),
                    "new_coordinates": (
                        row['pl_point_x_modified'],
                        row['pl_point_y_modified']
                    ), 
                    "altered_vertices" : [(row['pl_point_x'], row['pl_point_y'])],
                        # TODO: If this is more than one point, may need to be handled
                        "mtm_points_in_altered_vertices" : row['mtm_points_in_altered_vertices']
                        }
            
            update_or_add_alt_set(alt_set)
            alt_set["altered_vertices_smoothened"] = row["altered_vertices"]
            alt_set["altered_vertices_smoothened_reduced"] = row["altered_vertices"]
            
        elif row['alteration_type'] in ['CW Ext', 'CCW Ext'] and pd.notna(row['mtm points']):
            if row['mtm points'] != row['mtm_dependent']:
                print("")
                print("---------")
                print("Alteration Type (process rules): ")
                print(row['alteration_type'])
                print("---------")
                print("")
                print(f"Extension MTM {row['mtm points']}")
                row = self.apply_xy_move(row)

                # Apply smoothing after getting the new coordinates (i.e. if they aggregate)
                altered_vertices, mtm_points_in_altered_vertices = self.apply_extension(row)

                print(row['mtm points'])
                
                alt_set = {"mtm_point": int(row['mtm points']),
                           "mtm_dependant" : int(row['mtm_dependent']),
                        "alteration": row['alteration_type'],
                        "movement_x": row['movement_x'],
                        "movement_y": row['movement_y'],
                        "old_coordinates": (row['pl_point_x'], row['pl_point_y']),
                        "new_coordinates": (
                            row['pl_point_x_modified'] ,
                            row['pl_point_y_modified']
                        ), 
                        "altered_vertices" : self.processing_utils.sort_by_x(self.processing_utils.remove_duplicates(altered_vertices)),
                        "split_vertices" : [],
                        "mtm_points_in_altered_vertices" : mtm_points_in_altered_vertices,
                        }
                alt_set = update_or_add_alt_set(alt_set)
                print(f"Alteration Set Before Smoothing: {alt_set["altered_vertices"]}")

        return row
    
    def apply_extension(self, row):

        first_pt = row['mtm_dependent']
        second_pt = row['mtm points'] 

        first_point = self.get_pl_points(first_pt)
        second_point = self.get_pl_points(second_pt)
        
        #vertices_list = ast.literal_eval(row['vertices'])
        vertices_list = self.vertices_list.copy()
        
        first_index, second_index = self.find_vertex_indices(vertices_list, first_point, second_point)
        if first_index is not None and second_index is not None:
            if row['alteration_type'] == 'CW Ext' and first_index < second_index:
                row['altered_vertices'] = vertices_list[first_index:second_index + 1]
            elif row['alteration_type'] == 'CCW Ext' and first_index > second_index:
                row['altered_vertices'] = vertices_list[second_index:first_index + 1]
            else:
                row['altered_vertices'] = "Invalid vertices range"
        else:
            row['altered_vertices'] = "Vertices not found"

        # This is where the error is. Find point 8050
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
                            'original_index': original_index, 
                        })
        altered_vertices = row['altered_vertices']
        return altered_vertices, mtm_points_in_altered_vertices

    def calculate_distance(self):
        for index, row in self.df_alt.iterrows():
            if isinstance(row['mtm_points_in_altered_vertices'], list):
                distances = []
                for point_info in row['mtm_points_in_altered_vertices']:
                    if 'altered_coordinates' in point_info and 'original_coordinates' in point_info:
                        original = point_info['original_coordinates']
                        altered = point_info['altered_coordinates']
                        if None not in original and None not in altered:
                            distance = np.sqrt((altered[0] - original[0])**2 + (altered[1] - original[1])**2)
                            distances.append(distance)
                            print(f"MTM Point {point_info['mtm_point']}, Distance: {distance:.2f} cm")
                if distances:
                    max_distance = max(distances) * 100  # Convert to cm
                    self.df_alt.at[index, 'distance_euq'] = max_distance

                # Check for additional MTM points and calculate distance
                if not pd.isna(row['mtm points']):
                    mtm_x = row['pl_point_x_modified']
                    mtm_y = row['pl_point_y_modified']
                    if not pd.isna(mtm_x) and not pd.isna(mtm_y):
                        distance = np.sqrt((mtm_x - row['pl_point_x'])**2 + (mtm_y - row['pl_point_y'])**2) 
                        distances.append(distance)
                        print(f"MTM Point {row['mtm points']}, Distance: {distance:.2f} cm")
                    if distances:
                        max_distance = max(distances)  # Convert to cm
                        self.df_alt.at[index, 'distance_euq'] = max_distance

    def reduce_and_smooth_vertices(self, use_smoothened = True):
        for entry in self.total_alt:
            try:
                if use_smoothened:
                    reduced_vertices = self.processing_utils.reduce_points(vertices=entry['altered_vertices_smoothened'], threshold=0.1)
                    entry["altered_vertices_smoothened_reduced"] = reduced_vertices
                else:
                    reduced_vertices = self.processing_utils.reduce_points(vertices=entry['altered_vertices'], threshold=0.1)
                    entry["altered_vertices_reduced"] = reduced_vertices
            
            except KeyError or ValueError:
                print("")
                print(f"No reducing points needed for {entry['alteration']}")

    def reduce_original_vertices(self, row):
        if pd.isna(row['vertices']) or row['vertices'] in ['nan', 'None', '', 'NaN']:
            row['original_vertices_reduced'] = []
        else:
            try:
                vertices_list = ast.literal_eval(row['vertices'])
                print(f"Parsed vertices_list: {vertices_list}")
                if isinstance(vertices_list, list) and all(isinstance(vertex, (list, tuple)) and len(vertex) == 2 for vertex in vertices_list):
                    #print(f"Original vertices length: {len(vertices_list)}")
                    reduced_vertices = self.processing_utils.reduce_points(vertices=vertices_list, threshold=0.1)
                    # Ensure the reduced vertices are in the same structure (list of tuples)
                    reduced_vertices = [tuple(vertex) for vertex in reduced_vertices]
                    #print(f"Reduced vertices length: {len(reduced_vertices)}")
                    row['original_vertices_reduced'] = reduced_vertices
                else:
                    print(f"Invalid format in 'vertices' column for row: {row}")
                    row['original_vertices_reduced'] = []
            except (ValueError, SyntaxError) as e:
                print(f"Error processing vertices for row: {row}, Error: {e}")
                row['original_vertices_reduced'] = []
        return row

    def find_closest_points(self, mtm_point, current_x, current_y):
        df_existing_mtm = self.start_df.dropna(subset=['mtm points'])
        df_existing_mtm['distance'] = np.sqrt((df_existing_mtm['pl_point_x'] - current_x)**2 + (df_existing_mtm['pl_point_y'] - current_y)**2)
        df_sorted = df_existing_mtm[df_existing_mtm['mtm points'] != mtm_point].sort_values(by='distance')
        prev_point = df_sorted.iloc[0] if not df_sorted.empty else None
        next_point = df_sorted.iloc[1] if len(df_sorted) > 1 else None
        return prev_point, next_point

    def apply_xy_move(self, row):
        # Old version - multiply as a base percentage
        #row['pl_point_x_modified'] = row['pl_point_x'] * (1 + row['movement_x'])
        #row['pl_point_y_modified'] = row['pl_point_y'] * (1 + row['movement_y'])

        row['pl_point_x_modified'] = row['pl_point_x'] * (1 + row['movement_x'])
        row['pl_point_y_modified'] = row['pl_point_y'] * (1 + row['movement_y'])
        
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

        if isinstance(row['altered_vertices'], str) and row['altered_vertices'] in ['', 'nan', 'None', 'NaN']:
            row['altered_vertices'] = [altered_coordinates]
        else:
            altered_vertices = ast.literal_eval(row['altered_vertices']) if isinstance(row['altered_vertices'], str) else row['altered_vertices']
            altered_vertices.append(altered_coordinates)
            row['altered_vertices'] = altered_vertices

        return row

    # Get matching points for a particular MTM
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
    
    def vertex_smoothing(self, smoothing_method='linear', rbf_function='multiquadric', epsilon=None):
        coord_mapping = {}

        for alt_set in self.total_alt:
            alt_index = 0
            store_index = None

            old_coordinates = alt_set["old_coordinates"]
            new_coordinates_x = alt_set["new_coordinates"][0]
            new_coordinates_y = alt_set["new_coordinates"][1]
            altered_vertices = alt_set["altered_vertices"]
            split_vertices = alt_set["split_vertices"]

            # This one has to be after the alteration set is done
            # Now add the alteration on the second point
            # This new coordinates refers to the SECOND PT coordinates in the table
            # This is now Altered and need to be accounted for.
            new_coordinates = (new_coordinates_x, new_coordinates_y)
            coord_mapping[old_coordinates] = new_coordinates

            mtm_point = alt_set['mtm_point']
            if isinstance(alt_set['alteration'], list):
                for alteration_type in alt_set['alteration']:

                    # Get the dependant point first (by index)
                    dependant_mtm = alt_set['mtm_dependant'][alt_index]
                    print(f"Dependant MTM {dependant_mtm}")

                    first_point = self.get_pl_points(dependant_mtm)
                    second_point = self.get_pl_points(mtm_point)

                    # Get coordinates of First PT (e.g. 8015)
                    print(f"First point {first_point}")

                    # Get coordinates of Second PT (e.g. 8016)
                    print(f"Second point {second_point}")

                    change_x = abs(new_coordinates[0] - second_point[0])
                    change_y = abs(new_coordinates[1] - second_point[1])
                    ascending = (new_coordinates[0] > first_point[0] and new_coordinates[1] > first_point[1]) or \
                                (new_coordinates[0] < first_point[0] and new_coordinates[1] > first_point[1])

                    print("Is it Ascending? " + str(ascending))

                    if alteration_type == 'CW Ext':

                        # Get correct split vertices
                        vertices_to_mod = split_vertices[alteration_type]

                        print("")
                        print(f"Applying smoothing on {alteration_type}")
                        print(f"Vertices to smooth before updating coordinates")
                        print(vertices_to_mod)

                        # Update all occurrences of old_coordinates
                        for i, vertex in enumerate(altered_vertices):
                            if vertex in coord_mapping:
                                altered_vertices[i] = coord_mapping[vertex]

                        for i, vertex in enumerate(vertices_to_mod):
                            if vertex in coord_mapping:
                                vertices_to_mod[i] = coord_mapping[vertex]

                        print(f"The coordinates {old_coordinates} have been replaced by {new_coordinates} in the lists.")
                        print(f"Altered vertices after updating coordinates: {vertices_to_mod}")

                        # This shift should account for the double change?
                        shift = (new_coordinates_x - altered_vertices[-2][0],
                                new_coordinates_y - altered_vertices[-2][1])

                        print(f"Coordinate Shift {shift}")

                        # Run smoothing on a copy of the entry
                        vertices_to_mod_copy = vertices_to_mod.copy()

                        smoothing = SmoothingFunctions(vertices_to_mod, 
                                                       start_index = 0, 
                                                       end_index = len(vertices_to_mod_copy) - 2,
                                                       reverse=False)
                        
                        new_altered_vertices_tst = smoothing.apply_smoothing(
                            method='linear',
                            vertices=vertices_to_mod_copy,
                            shift=shift,
                            ascending=ascending,
                            change_x=change_x,
                            change_y=change_y,
                        )
                        new_altered_vertices_tst = sorted(new_altered_vertices_tst)
                        
                        print("Smoothened Vertices to mod")
                        print(new_altered_vertices_tst)
                        
                        print(f"CW EXT Ascending: {ascending}")

                        # Concatenate new vertices if exists, else create
                        if 'altered_vertices_smoothened' in alt_set:
                            alt_set['altered_vertices_smoothened'] += new_altered_vertices_tst
                        else:
                            alt_set['altered_vertices_smoothened'] = new_altered_vertices_tst

                        # Adjust for possible MTM Points inbetween (smoothened)
                        altered_index_map = {i: coord for i, coord in enumerate(new_altered_vertices_tst)}
                        print("Altered Index map")
                        print(altered_index_map)
                        for point_info in alt_set['mtm_points_in_altered_vertices']:
                            original_index = point_info.get('original_index')
                            original_coordinates = point_info.get('original_coordinates')
                            if original_coordinates in vertices_to_mod:
                                if original_index is not None:
                                    point_info['altered_coordinates'] = altered_index_map.get(original_index, (None, None))
                                    print("New Point Info")
                                    print(point_info)

                        alt_index += 1

                    elif alteration_type == 'CCW Ext':
                        vertices_to_mod = split_vertices[alteration_type]

                        print("")
                        print(f"Applying smoothing on {alteration_type}")
                        print(f"Vertices to smooth")
                        print(vertices_to_mod)

                        # Update all occurrences of old_coordinates
                        for i, vertex in enumerate(altered_vertices):
                            if vertex in coord_mapping:
                                altered_vertices[i] = coord_mapping[vertex]

                        for i, vertex in enumerate(vertices_to_mod):
                            if vertex in coord_mapping:
                                vertices_to_mod[i] = coord_mapping[vertex]

                        print(f"The coordinates {old_coordinates} have been replaced by {new_coordinates} in the lists.")
                        print(f"Altered vertices after updating coordinates: {vertices_to_mod}")

                        shift = (new_coordinates_x - altered_vertices[1][0],
                                new_coordinates_y - altered_vertices[1][1])

                        print(f"Coordinate Shift for Alteration Type {shift}")
                        print(f"CCW EXT Ascending: {ascending}")

                        # Run smoothing on a copy of the entry
                        vertices_to_mod_copy = vertices_to_mod.copy()
                        smoothing = SmoothingFunctions(vertices_to_mod_copy, 
                                                    start_index=1, 
                                                    end_index=len(vertices_to_mod_copy) - 1,
                                                    reverse=True)
                        new_altered_vertices_tst = smoothing.apply_smoothing(
                            method='linear',
                            vertices=vertices_to_mod_copy,
                            shift=shift,
                            ascending=ascending,
                            change_x=change_x,
                            change_y=change_y,
                        )
                        new_altered_vertices_tst = sorted(new_altered_vertices_tst)

                        print("Smoothened Vertices to mod")
                        print(new_altered_vertices_tst)

                        # Concatenate new vertices if exists, else create
                        if 'altered_vertices_smoothened' in alt_set:
                            alt_set['altered_vertices_smoothened'] += new_altered_vertices_tst
                        else:
                            alt_set['altered_vertices_smoothened'] = new_altered_vertices_tst

                        # Update the altered vertices back into the alt_set
                        #alt_set['altered_vertices_smoothened'] = new_altered_vertices

                        # Adjust for possible MTM Points inbetween (smoothened)
                        altered_index_map = {i: coord for i, coord in enumerate(new_altered_vertices_tst)}
                        print("Altered Index map")
                        print(altered_index_map)
                        for point_info in alt_set['mtm_points_in_altered_vertices']:
                            original_index = point_info.get('original_index')
                            original_coordinates = point_info.get('original_coordinates')
                            if original_coordinates in vertices_to_mod:
                                if original_index is not None:
                                    point_info['altered_coordinates'] = altered_index_map.get(original_index, (None, None))
                                    print("New Point Info")
                                    print(point_info)

                    alt_index += 1
            else:
                print(f"Altering {alt_set['alteration']}")
                first_pt = alt_set["mtm_point"]
                second_pt = alt_set["mtm_dependant"]

                print("Alt Set")
                print(alt_set)

                first_point = self.get_pl_points(first_pt)
                second_point = self.get_pl_points(second_pt)

                # Get coordinates of First PT (e.g. 8015)
                print("MTM Point Coordinates")
                print(first_pt)
                print(first_point)

                # Get coordinates of Second PT (e.g. 8016)
                print("MTM Dependent Coordinates")
                print(second_pt)
                print(second_point)

                print("Second Point Altered (New coordinates)")
                print(new_coordinates)

                change_x = abs(new_coordinates[0] - second_point[0])
                change_y = abs(new_coordinates[1] - second_point[1])
                ascending = (new_coordinates[0] > first_point[0] and new_coordinates[1] > first_point[1]) or \
                            (new_coordinates[0] < first_point[0] and new_coordinates[1] > first_point[1])

        return

    def get_unique_vertices(self, df):
        unique_vertices_set = set()
        unique_vertices = []
        for _, row in df.iterrows():
            if pd.isna(row['vertices']) or row['vertices'] in ['nan', 'None', '', 'NaN']:
                continue
            vertices_list = ast.literal_eval(row['vertices'])
            for vertex in vertices_list:
                if vertex not in unique_vertices_set:
                    unique_vertices_set.add(vertex)
                    unique_vertices.append(vertex)
        return unique_vertices
    
    def merge_with_original_df(self, original_df, total_alt_df):
        original_df.drop(columns = ['alteration_type', 'altered_vertices', 'mtm_dependent',
                                    'movement_x', 'movement_y'], inplace=True)
        
        
        original_df.rename(columns={'mtm_points_in_altered_vertices': 'mtm_points_in_altered_vertices_ref'}, inplace=True)

        # Assuming 'mtm_point' is the key to merge on
        merged_df = original_df.merge(total_alt_df, left_on='mtm points', right_on='mtm_point', how='left')

        # Drop the original 'mtm points' column
        #merged_df.drop(columns=['mtm points'], inplace=True)
        
        # Optionally, rename 'mtm_point' to 'mtm points' if you need to keep the same column name
        merged_df.rename(columns={'mtm_point': 'mtm_points_alteration'}, inplace=True)
        merged_df.rename(columns={'vertices': 'original_vertices'}, inplace=True)

        # Sort columns by X-Coordinate (again)
        merged_df['alteration_set'] = merged_df['altered_vertices'].apply(self.processing_utils.sort_by_x)
        merged_df['altered_vertices'] = merged_df['altered_vertices_smoothened'].apply(self.processing_utils.sort_by_x)
        merged_df['altered_vertices_reduced'] = merged_df['altered_vertices_smoothened_reduced'].apply(self.processing_utils.sort_by_x)
        merged_df.drop(columns = ['altered_vertices_smoothened', 'altered_vertices_smoothened_reduced'], inplace=True)

        return merged_df
        
    def get_mtm_dependent_coords(self, df):
        # Initialize new columns with object dtype to handle lists or strings
        df['mtm_dependant_x'] = pd.Series(dtype='object')
        df['mtm_dependant_y'] = pd.Series(dtype='object')

        def parse_labels(labels):
            if isinstance(labels, str):
                return ast.literal_eval(labels)
            return labels if isinstance(labels, list) else [labels]

        def check_all_labels_in_list(labels, matching_list):
            labels = parse_labels(labels)
            return all(item in matching_list for item in labels)

        # Flatten and print mtm dependant values
        mtm_dependant_vals = df['mtm_dependant'].dropna().tolist()
        mtm_dependant_vals_flattened = self.processing_utils.flatten_if_needed(mtm_dependant_vals)
        print("MTM Dependent:", mtm_dependant_vals)

        # Get matching rows based on mtm points
        matching_rows = df[df['mtm points'].isin(mtm_dependant_vals_flattened)]
        matching_mtm_labels = matching_rows['mtm points'].unique()
        print("Matching mtm labels:", matching_mtm_labels)

        # Define a dictionary to keep track of unique (x, y) pairs
        unique_coords = {}

        # Add unique points to the dictionary
        for label, x, y in zip(matching_rows['mtm points'], matching_rows['pl_point_x'], matching_rows['pl_point_y']):
            if (x, y) not in unique_coords:
                unique_coords[(x, y)] = label

        # Convert the dictionary back to lists
        coords = {
            "coords_x": [key[0] for key in unique_coords.keys()],
            "coords_y": [key[1] for key in unique_coords.keys()],
            "label": list(unique_coords.values())
        }
        print("COORDS:", coords)

        # Now that coords is defined, we can safely use it in the lambda function
        mtm_dependant_labels = df[df['mtm_dependant'].apply(lambda x: check_all_labels_in_list(x, coords["label"]))]
        print("MTM Dependant labels:", mtm_dependant_labels)

        # Iterate over the rows in mtm_dependant_labels
        for _, row in mtm_dependant_labels.iterrows():
            labels = parse_labels(row['mtm_dependant'])
            
            x_coords = []
            y_coords = []
            
            if all(label in coords["label"] for label in labels):
                x_coords = [coords["coords_x"][coords["label"].index(label)] for label in labels]
                y_coords = [coords["coords_y"][coords["label"].index(label)] for label in labels]
            
            # Assign the coordinates to the DataFrame
            df.at[row.name, 'mtm_dependant_x'] = x_coords if len(x_coords) > 1 else x_coords[0]
            df.at[row.name, 'mtm_dependant_y'] = y_coords if len(y_coords) > 1 else y_coords[0]

        return df

if __name__ == "__main__":

    ###### Piece names ####
    # Chest 
    #piece_name = "LGFG-SH-01-CCB-FO"
    #alteration_input = "combined_table_4-WAIST.csv"

    # Cuffs
    #piece_name = "LGFG-FG-CUFF-S2"
    #piece_name = "LGFG-SH-CUFF-D1"
    #alteration_input = "combined_table_2SL-FCUFF.csv"

    # Collar
    piece_name = "LGFG-1648-FG-07S"
    alteration_input = "combined_table_3-COLLAR.csv"
    ########
    
    # Vertices Input
    vertices_input = piece_name + "_vertices.csv"

    # Set Table paths
    input_table_path = "../data/output_tables/combined_alteration_tables/" + alteration_input
    input_vertices_path = "../data/output_tables/vertices/" + vertices_input

    # Specify save folder Dir
    save_folder = "../data/output_tables/processed_alterations/"
    file_format = '.xlsx' 
    #file_format = '.csv'
    
    make_alteration = MakeAlteration(input_table_path, input_vertices_path, 
                                     piece_name=piece_name, save_folder = save_folder, 
                                     file_format=file_format) 
    
    processed_df = make_alteration.apply_alteration_rules(custom_alteration=False)

    # Apply alteration for all, and save to separate directories
