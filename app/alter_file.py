import pandas as pd
from matplotlib import pyplot as plt
import ast
import os
import numpy as np
from scipy.interpolate import splprep, splev, interp1d, Rbf
from smoothing import SmoothingFunctions  # Import the SmoothingFunctions class


# TODO: CHECK IF MTM POINTS APPEAR CORRECTLY IN THE ALTRED SMOOTH LINE

# Further notes:
# Vertices have to be sorted (by x-coordinate) before smoothing is applied
# Vertices need to not have duplicate coordinate entries 

# Todo:
# Add the new altered coordinate to the alteration list
# Consider if all other points need to be doubled? Think!
# Does the alteration list include the main mtm point?
# Which coordinates should the list have?
# Look into what smoothing does

# PRIO: TODO INVESTIGATE THE FIRST LINE WHY THE ALTERED VERSION LOOKS OFF
# SECOND LOOKS OK?

## TODO: Fix CCW EXT FIRST POINT. ADJACANT IS 8017?

# TODO later: Do cleanup

# TODO: append XY POINT TO ALTERED VERTICES

class MakeAlteration:
    def __init__(self, input_table_path):
        self.input_table_path = input_table_path
        self.sheets_df = self.load_sheets()
        self.df_alt = ""
        self.start_df = ""
        self.total_alt = []

    @staticmethod
    def remove_duplicates(input_list):
        '''
            Remove Duplicates from list of Tuples / coordinates.
        '''
        seen = set()
        output_list = []
        for item in input_list:
            if item not in seen:
                seen.add(item)
                output_list.append(item)
        return output_list

    @staticmethod
    def reduce_points(vertices, threshold=0.1):
        if isinstance(vertices, list) and len(vertices) > 2:
            points = np.array(vertices)
            reduced_points = MakeAlteration.visvalingam_whyatt(points, threshold)
            return reduced_points
        return vertices

    def load_sheets(self):
        return pd.read_excel(self.input_table_path, sheet_name=None)
    
    # Function to sort coordinates by x-coordinate
    @staticmethod
    def sort_by_x(coords):
        if isinstance(coords, list) and all(isinstance(coord, (list, tuple)) for coord in coords):
            return sorted(coords, key=lambda coord: coord[0])
        return coords  # Return as is if not a list of tuples

    def prepare_dataframe(self, df):
        df['pl_point_x'] = pd.to_numeric(df['pl_point_x'], errors='coerce').fillna(0)
        df['pl_point_y'] = pd.to_numeric(df['pl_point_y'], errors='coerce').fillna(0)
        df['movement x'] = df['movement x'].astype(str).str.replace('%', '').astype(float).fillna(0)
        df['movement y'] = df['movement y'].astype(str).str.replace('%', '', regex=False).astype(float).fillna(0)
        df['pl_point_x_modified'] = ""
        df['pl_point_y_modified'] = ""
        df['altered_vertices'] = ""
        df['distance_euq'] = ""
        return df
    
    def create_total_alt_df(self):
        return pd.DataFrame(self.total_alt)
    
    def merge_with_original_df(self, original_df, total_alt_df):
        original_df.drop(columns = ['alt type', 'altered_vertices', 'first pt', 'second pt',
                                    'line_end_x', 'line_end_y', 'line_start_x', 'line_start_y',
                                    'movement x', 'movement y'], inplace=True)
        
        
        original_df.rename(columns={'mtm_points_in_altered_vertices': 'mtm_points_in_altered_vertices_ref'}, inplace=True)

        # Assuming 'mtm_point' is the key to merge on
        merged_df = original_df.merge(total_alt_df, left_on='mtm points', right_on='mtm_point', how='left')

        # Drop the original 'mtm points' column
        #merged_df.drop(columns=['mtm points'], inplace=True)
        
        # Optionally, rename 'mtm_point' to 'mtm points' if you need to keep the same column name
        merged_df.rename(columns={'mtm_point': 'mtm_points_alteration'}, inplace=True)
        merged_df.rename(columns={'vertices': 'original_vertices'}, inplace=True)

        # Sort columns by X-Coordinate (again)
        merged_df['alteration_set'] = merged_df['altered_vertices'].apply(MakeAlteration.sort_by_x)
        merged_df['altered_vertices'] = merged_df['altered_vertices_smoothened'].apply(MakeAlteration.sort_by_x)
        merged_df['altered_vertices_reduced'] = merged_df['altered_vertices_smoothened_reduced'].apply(MakeAlteration.sort_by_x)
        merged_df.drop(columns = ['altered_vertices_smoothened', 'altered_vertices_smoothened_reduced'], inplace=True)

        return merged_df

    def apply_alteration_rules(self, df):
        df = self.prepare_dataframe(df)
        df = df.apply(self.reduce_original_vertices, axis=1)
        df = df.apply(self.process_alteration_rules, axis=1)

        print("")
        print("---------")
        print("Initial Alteration Set")
        print("")
        print(self.total_alt)

        # Apply smoothing (DONE)
        self.vertex_smoothing()

        print("")
        print("---------")
        print("Alteration Set After Smoothing")
        print("")
        print(self.total_alt)

        self.reduce_and_smooth_vertices()

        print("")
        print("---------")
        print("Alteration Set After Reducing Points")
        print("")
        print(self.total_alt)

        # Create DataFrame from total_alt
        total_alt_df = self.create_total_alt_df()

        # Combine coordinates - Adjust for inbetween mtm points and add to the smoothened lines
        #total_alt_df = total_alt_df.apply(self.add_inbetween_mtm_points, axis=1)
        #print("")
        #print("---------")
        #print("Alteration Set After Combining Coordinates")
        #print("")
        #print(self.total_alt)

        # Merge with the original DataFrame
        merged_df = self.merge_with_original_df(df, total_alt_df)

        # Adjust for missing mtm points
        #merged_df["original_vertices_reduced"] = merged_df.apply(self.add_mtm_points_to_original, axis=1)

        # Save the resulting DataFrame to an Excel file
        merged_df.to_excel("../data/output_tables/processed_alterations.xlsx", index=False)
        
        self.df_alt = merged_df
        return merged_df

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
                    reduced_vertices = MakeAlteration.reduce_points(vertices=entry['altered_vertices_smoothened'], threshold=0.1)
                    entry["altered_vertices_smoothened_reduced"] = reduced_vertices
                else:
                    reduced_vertices = MakeAlteration.reduce_points(vertices=entry['altered_vertices'], threshold=0.1)
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
                #print(f"Parsed vertices_list: {vertices_list}")
                if isinstance(vertices_list, list) and all(isinstance(vertex, (list, tuple)) and len(vertex) == 2 for vertex in vertices_list):
                    #print(f"Original vertices length: {len(vertices_list)}")
                    reduced_vertices = self.reduce_points(vertices=vertices_list, threshold=0.1)
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

    def process_alteration_rules(self, row):
        if pd.isna(row['alt type']):
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
                    existing['new_coordinates'] = (
                        existing['old_coordinates'][0] * (1 + existing['movement_x'] / 100.0) ,
                        existing['old_coordinates'][1] * (1 + existing['movement_y'] / 100.0)
                    )
                    
                    existing['mtm_points_in_altered_vertices'] = alt_set['mtm_points_in_altered_vertices'] + existing['mtm_points_in_altered_vertices']

                    # Don't forget to include all the altered vertices
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

        if row['alt type'] == 'X Y MOVE':
            print("---------")
            print("Alteration: ")
            print(row['alt type'])
            print("---------")
            row = self.apply_xy_move(row)

            alt_set = {"mtm_point": int(row['mtm points']),
                       "mtm_dependant" : int(row['first pt']),
                    "alteration": row['alt type'],
                    "movement_x": row['movement x'],
                    "movement_y": row['movement y'],
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
            
        elif row['alt type'] in ['CW Ext', 'CCW Ext']:
            if row['mtm points'] != row['first pt']:
                print("")
                print("---------")
                print("Alteration Type (process rules): ")
                print(row['alt type'])
                print("---------")
                print("")
                print(f"Extension MTM {row['mtm points']}")
                row = self.apply_xy_move(row)

                # Apply smoothing after getting the new coordinates (i.e. if they aggregate)
                altered_vertices, mtm_points_in_altered_vertices = self.apply_extension(row)
                
                alt_set = {"mtm_point": int(row['mtm points']),
                           "mtm_dependant" : int(row['first pt']),
                        "alteration": row['alt type'],
                        "movement_x": row['movement x'],
                        "movement_y": row['movement y'],
                        "old_coordinates": (row['pl_point_x'], row['pl_point_y']),
                        "new_coordinates": (
                            row['pl_point_x_modified'] ,
                            row['pl_point_y_modified']
                        ), 
                        "altered_vertices" : MakeAlteration.sort_by_x(MakeAlteration.remove_duplicates(altered_vertices)),
                        "split_vertices" : [],
                        "mtm_points_in_altered_vertices" : mtm_points_in_altered_vertices,
                        }
                alt_set = update_or_add_alt_set(alt_set)
                print(f"Alteration Set Before Smoothing: {alt_set["altered_vertices"]}")

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
    
    def apply_extension(self, row):
        first_pt = row['first pt']
        second_pt = row['second pt'] 

        first_point = self.get_pl_points(first_pt)
        second_point = self.get_pl_points(second_pt)
        
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

            print("Old coordinates")
            print(old_coordinates)

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

                first_point = self.get_pl_points(first_pt)
                second_point = self.get_pl_points(second_pt)

                # Get coordinates of First PT (e.g. 8015)
                print("First point")
                print(first_pt)
                print(first_point)

                # Get coordinates of Second PT (e.g. 8016)
                print("Second Point")
                print(second_pt)
                print(second_point)

                print("Second Point Altered (New coordinates)")
                print(new_coordinates)

                change_x = abs(new_coordinates[0] - second_point[0])
                change_y = abs(new_coordinates[1] - second_point[1])
                ascending = (new_coordinates[0] > first_point[0] and new_coordinates[1] > first_point[1]) or \
                            (new_coordinates[0] < first_point[0] and new_coordinates[1] > first_point[1])

                print("Change X")
                print(change_x)

                print("Change Y")
                print(change_y)

                # Second PT ordering
                # CW EXT: END
                # CCW EXT: BEGINNING
                # Add more cases later...
                print("Old Coordinates")
                print(old_coordinates)
        return



    @staticmethod
    def visvalingam_whyatt(points, threshold):
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
        points = [tuple(point) for point in points]
        simplified_points = filter_points(points, threshold)

        #print(f"Number of Points after Simplification: {len(simplified_points)}")
        return simplified_points

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

if __name__ == "__main__":
    input_table_path = "../data/output_tables/merged_with_rule_subset.xlsx"
    make_alteration = MakeAlteration(input_table_path)

    for sheet_name, df in make_alteration.sheets_df.items():
        print(f"Processing sheet: {sheet_name}")
        make_alteration.start_df = df
        make_alteration.df_alt = df.copy()
        processed_df = make_alteration.apply_alteration_rules(df)
        
        #plotting_df = make_alteration.get_plotting_info(processed_df)

        #output_dir = f"../data/output_graphs/{sheet_name}"
        #make_alteration.plot_altered_table(plotting_df, output_dir=output_dir)
        #make_alteration.plot_final_altered_table(plotting_df, output_dir=output_dir)

        break
