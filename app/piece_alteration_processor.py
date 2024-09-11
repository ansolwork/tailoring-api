import pandas as pd
import ast
import numpy as np
from app.smoothing import SmoothingFunctions  # Import the SmoothingFunctions class
from utils.data_processing_utils import DataProcessingUtils
import os
from functools import partial
from itertools import combinations

# Setup Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
) # Levels Info/Warning/Error/Critical

# TODO: Merge all notch points: DONE
# TODO: When Reducing the Points, THE MTM POINTS DO NOT LOOK EXACTLY ALIGNED ON THE PLOT. DONE

# TODO: NEARBY MTM POINT FOR 8016 is WRONG

class PieceAlterationProcessor:
    """
    Handles the processing of alteration rules for a specific piece.

    This class processes alteration data and corresponding vertices for a piece. 
    It applies alteration rules, reduces the vertices, and saves the processed data.

    :param piece_table_path: Path to the CSV file containing staged alteration data (i.e. Combined Table file for a particular piece).
    :param vertices_table_path: Path to the CSV file containing the staged vertex data.
    :param save_folder_processed_pieces: Folder where processed alterations are saved.
    :param save_folder_processed_vertices: Folder where processed vertices are saved.
    :param save_file_format: Format to save the processed files.
    """
    def __init__(self, 
                piece_table_path=None, 
                vertices_table_path=None, 
                save_folder_processed_pieces="data/staging_processed/processed_alterations_by_piece/", 
                save_folder_processed_vertices="data/staging_processed/processed_vertices_by_piece/", 
                save_file_format=".csv",
                debug_alteration_rule = None):
        
        self.processing_utils = DataProcessingUtils()

        # Load the piece and vertices data
        self.piece_df = self.processing_utils.load_csv(piece_table_path)
        self.vertices_df = self.processing_utils.load_csv(vertices_table_path)
        self.processed_vertices_list = []

        # Get the unique alteration rules
        self.alteration_rules = self.get_alteration_rules()

        # Dictionary to store DataFrames by alteration rule
        self.alteration_dfs = self.split_by_alteration_rule()

        # Store All Processed Alteration Sets
        self.alteration_log = []

        # Piece name (assuming only one unique piece name)
        self.piece_name = self.get_piece_name()

        # Save Options
        self.save_folder_processed_pieces = save_folder_processed_pieces
        self.save_folder_processed_vertices = save_folder_processed_vertices
        self.save_file_format = save_file_format

        # Debugging
        self.debug_alteration_rule = debug_alteration_rule

    def get_alteration_rules(self):
        """
        Returns a list of unique alteration rules.
        """
        return self.piece_df['alteration_rule'].dropna().unique().tolist()

    def get_piece_name(self):
        """
        Returns the unique piece name from the piece_name column.
        Assumes there is only one unique piece name.
        """
        piece_name = self.piece_df['piece_name'].dropna().unique()
        if len(piece_name) == 1:
            return piece_name[0]
        else:
            raise ValueError("There should only be one unique piece name, but multiple were found.")

    def split_by_alteration_rule(self):
        """
        Splits the piece_df into multiple DataFrames, organized by alteration rule.
        Rows without an alteration rule (NaN values) are included in each DataFrame.
        Returns a dictionary where the keys are alteration rules and the values are DataFrames.
        """
        # Separate rows where alteration_rule is NaN
        no_rule_df = self.piece_df[self.piece_df['alteration_rule'].isna()]

        # Create a dictionary to hold DataFrames split by alteration_rule
        alteration_dfs = {}

        # Group the DataFrame by alteration_rule
        for rule, group in self.piece_df.groupby('alteration_rule'):
            # Combine rows with NaN alteration rules with the group
            combined_df = pd.concat([group, no_rule_df])
            alteration_dfs[rule] = combined_df
            #self.processing_utils.save_csv(combined_df, "data/debug/" + rule + ".csv")

        return alteration_dfs
    
    def prepare_dataframe(self, df):
        """
        Prepares the DataFrame by converting columns to numeric values and adding necessary columns.

        :param df: DataFrame to be prepared.
        :return: The modified DataFrame.
        """        
        cols_to_convert = ['pl_point_x', 'pl_point_y', 'maximum_movement_inches_positive', 
                           'maximum_movement_inches_negative', 'minimum_movement_inches_positive', 
                           'minimum_movement_inches_negative']
        
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)        
        
        df['movement_x'] = df['movement_x'].astype(str).str.replace('%', '').astype(float).fillna(0)
        df['movement y'] = df['movement_y'].astype(str).str.replace('%', '', regex=False).astype(float).fillna(0)
        df['pl_point_x_modified'] = ""
        df['pl_point_y_modified'] = ""
        df['altered_vertices'] = ""
        return df
    
    def get_unselected_rows(self, selected_df):
        """
        Finds rows in the original piece DataFrame that are not part of the selected DataFrame based on 'mtm points'.
        For unselected rows, sets 'alteration_rule' and 'alteration_type' to NaN, and resets movement-related columns to 0.
        
        :param selected_df: DataFrame with rows matching the current alteration rule
        :return: DataFrame with unselected rows
        """
        unselected_df = self.piece_df[~self.piece_df['mtm points'].isin(selected_df['mtm points'])].copy()
        unselected_df['alteration_rule'] = np.nan
        unselected_df['alteration_type'] = np.nan
        unselected_df['maximum_movement_inches_negative'] = 0.0
        unselected_df['maximum_movement_inches_positive'] = 0.0
        unselected_df['minimum_movement_inches_negative'] = 0.0
        unselected_df['minimum_movement_inches_positive'] = 0.0
        unselected_df['movement_y'] = 0.0
        
        return unselected_df
    
    def combine_and_clean_data(self, selected_df, unselected_df):
        """
        Combines selected and unselected rows into a single DataFrame. Cleans up the data by ensuring
        numeric consistency and removing duplicate rows based on key columns.
        
        :param selected_df: DataFrame containing the rows for the selected alteration rule
        :param unselected_df: DataFrame with unselected rows
        :return: Cleaned and combined DataFrame
        """
        combined_df = pd.concat([selected_df, unselected_df], ignore_index=True)
        combined_df = self.drop_duplicate_rows(combined_df)
        
        return combined_df

    def drop_duplicate_rows(self, df):
        """
        Removes duplicate rows based on 'mtm points', 'pl_point_x', and 'pl_point_y'.
        Ensures consistency in numeric types and precision before dropping duplicates.
        
        :param df: DataFrame to process
        :return: DataFrame with duplicates removed
        """
        df['mtm points'] = pd.to_numeric(df['mtm points'], errors='coerce')  # Ensure numeric type for 'mtm points'
        df['pl_point_x'] = df['pl_point_x'].round(3)  # Round to avoid floating-point precision issues
        df['pl_point_y'] = df['pl_point_y'].round(3)
        
        return df.drop_duplicates(subset=['mtm points', 'pl_point_x', 'pl_point_y'])
    
    def process_alterations(self):
        """
        Applies alteration rules to each DataFrame and processes vertices. If a debug_alteration_rule is set,
        only that rule is applied. Otherwise, all alteration rules are processed.
        """
        # Prepare vertices
        self.process_and_save_vertices()

        def apply_single_rule(alteration_rule):
            """
            Applies a single alteration rule by combining the selected and unselected rows.
            Prepares the combined DataFrame and processes it row-by-row.
            """
            selected_df = self.alteration_dfs[alteration_rule]
            unselected_df = self.get_unselected_rows(selected_df)
            
            combined_df = self.combine_and_clean_data(selected_df, unselected_df)
            processed_df = self.prepare_dataframe(combined_df)
            
            # Apply alterations row-by-row
            return processed_df.apply(partial(self.apply_alteration_to_row, selected_df=processed_df), axis=1)

        # Case when a specific debug alteration type is provided
        if not pd.isna(self.debug_alteration_rule):

            # Setup Save Info
            debug_savefolder = "data/staging_processed/debug/"
            save_filepath = f"{debug_savefolder}{self.piece_name}_{self.debug_alteration_rule}{self.save_file_format}"
            os.makedirs(debug_savefolder, exist_ok=True)

            if self.debug_alteration_rule in self.alteration_rules:
                # Apply the specific debug alteration rule
                logging.info(f"Running Alteration in Debug Mode: {self.debug_alteration_rule}")
                alteration_df = apply_single_rule(self.debug_alteration_rule)

                # Even in debug mode, we should process and save the result.
                cumulative_alteration_df = pd.DataFrame(self.alteration_log)
                merged_df = self.merge_with_alteration_df(alteration_df, cumulative_alteration_df)

                # Save the result
                logging.info(f"Saved Processed Debug Alterations To: {save_filepath}")
                merged_df = self.get_mtm_dependent_coords(merged_df)
                merged_df.to_csv(save_filepath, index=False)
            else:
                logging.warning(f"Debug alteration type '{self.debug_alteration_rule}' not found in requested alterations.")
            return  # Early return after processing the debug alteration

        # Apply all alteration rules when no specific debug type is provided
        logging.info("Running Full Alteration Process")
        all_merged_df = {}
        for alteration_rule in self.alteration_rules:
            alteration_df = apply_single_rule(alteration_rule)
        
            # After applying each alteration rule, accumulate and merge the data
            cumulative_alteration_df = pd.DataFrame(self.alteration_log)
            merged_df = self.merge_with_alteration_df(alteration_df, cumulative_alteration_df)
            merged_df = self.get_mtm_dependent_coords(merged_df)

            # Concatenate all 

        # Save the result for the full alteration process (concatenate all)
        #logging.info(f"Saved Processed Alterations To: {save_filepath}")
        #merged_df.to_csv(save_filepath, index=False)

    def merge_with_alteration_df(self, alteration_df, cumulative_alteration_df):
        """
        Merges the original alteration DataFrame with the cumulative alterations DataFrame.
        
        This function merges the original alteration data (`alteration_df`) with the cumulative alterations
        (`cumulative_alteration_df`) that have been processed and stored in `self.alteration_log`. The final
        DataFrame will include both the original piece information and the alterations applied to it, along with
        other relevant details such as altered vertices.

        The process involves:
        1. Dropping unnecessary columns from the original `alteration_df` to prevent duplication.
        2. Renaming columns for better clarity and consistency.
        3. Merging `alteration_df` and `cumulative_alteration_df` on the `mtm points` to align the original data 
        with the cumulative alterations.
        4. Sorting the altered vertices by their X-coordinates to maintain spatial integrity.
        5. Cleaning up the final DataFrame by removing intermediate columns no longer required.

        Parameters:
        ----------
        alteration_df : pd.DataFrame
            The original DataFrame containing the staged alterations for the piece.
            
        cumulative_alteration_df : pd.DataFrame
            DataFrame containing cumulative alterations applied to the piece, compiled from `self.alteration_log`.

        Returns:
        -------
        pd.DataFrame
            A merged DataFrame containing the original piece data alongside the cumulative alterations, ready 
            for further analysis or saving.

        Columns in the final DataFrame:
        - `mtm_points_ref`: Reference points from the original `alteration_df`.
        - `mtm_points_alteration`: Alteration points from `cumulative_alteration_df`.
        - `original_vertices`: Vertices from the original `alteration_df` before any alterations.
        - `alteration_set`: A sorted list of altered vertices based on X-coordinates.

        Notes:
        ------
        - The function depends on `self.processing_utils.sort_by_x()` to ensure that altered vertices are 
        sorted by their X-coordinates for spatial consistency.
        """
        
        # Copy original alteration_df to avoid modifying the original DataFrame directly
        original_df = alteration_df.copy()

        # Drop columns that are not required post-merging to avoid duplication
        original_df.drop(columns=['alteration_type', 'altered_vertices', 'mtm_dependent', 'movement_x', 'movement_y', 'vertices'], inplace=True)

        # Rename columns for clarity
        original_df.rename(columns={'mtm_points_in_altered_vertices': 'mtm_points_ref'}, inplace=True)

        # Merge original alteration_df with cumulative_alteration_df using 'mtm points' as the key
        merged_df = original_df.merge(cumulative_alteration_df, left_on='mtm points', right_on='mtm_point', how='left')

        # Rename columns to make clear distinction between original and altered data
        merged_df.rename(columns={'mtm_point': 'mtm_points_alteration'}, inplace=True)

        # Sort the altered vertices by X-coordinates using the processing utility
        merged_df['alteration_set'] = merged_df['altered_vertices'].apply(self.processing_utils.sort_by_x)
        merged_df['altered_vertices'] = merged_df['altered_vertices_smoothened'].apply(self.processing_utils.sort_by_x)
        merged_df['altered_vertices_reduced'] = merged_df['altered_vertices_smoothened_reduced'].apply(self.processing_utils.sort_by_x)
        merged_df.drop(columns = ['altered_vertices_smoothened', 'altered_vertices_smoothened_reduced'], inplace=True)

        return merged_df
    
    def get_mtm_dependent_coords(self, df):
        """
        Populates dependent MTM coordinates (X and Y) for each row in the DataFrame.

        This function identifies the rows in the DataFrame that have dependent MTM points, retrieves the
        coordinates (X and Y) for those dependent points, and populates two new columns: `mtm_dependant_x`
        and `mtm_dependant_y` with these coordinates.

        The process involves:
        1. Initializing two new columns in the DataFrame to store X and Y coordinates for the dependent MTM points.
        2. Parsing the `mtm_dependant` column to handle strings, lists, or individual labels.
        3. Flattening and retrieving all MTM dependent values.
        4. Finding matching rows based on `mtm points` and retrieving their X and Y coordinates.
        5. Assigning the dependent X and Y coordinates to the new columns.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the original data, including the `mtm_dependant` and `mtm points` columns.

        Returns:
        -------
        pd.DataFrame
            The updated DataFrame with two new columns:
            - `mtm_dependant_x`: The X coordinates of the dependent MTM points.
            - `mtm_dependant_y`: The Y coordinates of the dependent MTM points.
        """

        # Initialize new columns with object dtype to handle lists or individual values
        df['mtm_dependant_x'] = pd.Series(dtype='object')
        df['mtm_dependant_y'] = pd.Series(dtype='object')

        # Helper function to parse the 'mtm_dependant' column
        def parse_labels(labels):
            if isinstance(labels, str):
                return ast.literal_eval(labels)
            return labels if isinstance(labels, list) else [labels]

        # Helper function to check if all labels are in the provided list
        def check_all_labels_in_list(labels, matching_list):
            labels = parse_labels(labels)
            return all(item in matching_list for item in labels)

        # Flatten the MTM dependant values
        mtm_dependant_vals = df['mtm_dependant'].dropna().tolist()
        mtm_dependant_vals_flattened = self.processing_utils.flatten_if_needed(mtm_dependant_vals)

        # Get rows where mtm points match the dependant values
        matching_rows = df[df['mtm points'].isin(mtm_dependant_vals_flattened)]
        matching_mtm_labels = matching_rows['mtm points'].unique()

        # Create a dictionary to store unique (x, y) pairs by their labels
        unique_coords = {}
        for label, x, y in zip(matching_rows['mtm points'], matching_rows['pl_point_x'], matching_rows['pl_point_y']):
            if (x, y) not in unique_coords:
                unique_coords[(x, y)] = label

        # Convert the dictionary back to lists
        coords = {
            "coords_x": [key[0] for key in unique_coords.keys()],
            "coords_y": [key[1] for key in unique_coords.keys()],
            "label": list(unique_coords.values())
        }

        # Filter rows where all labels in 'mtm_dependant' exist in the coords['label']
        mtm_dependant_labels = df[df['mtm_dependant'].apply(lambda x: check_all_labels_in_list(x, coords["label"]))]

        # Iterate over the rows to assign the dependent coordinates
        for _, row in mtm_dependant_labels.iterrows():
            labels = parse_labels(row['mtm_dependant'])

            x_coords = []
            y_coords = []

            # Check if all labels exist in the coordinate dictionary
            if all(label in coords["label"] for label in labels):
                x_coords = [coords["coords_x"][coords["label"].index(label)] for label in labels]
                y_coords = [coords["coords_y"][coords["label"].index(label)] for label in labels]

            # Assign coordinates to the DataFrame (handling multiple or single coordinates)
            df.at[row.name, 'mtm_dependant_x'] = x_coords if len(x_coords) > 1 else x_coords[0]
            df.at[row.name, 'mtm_dependant_y'] = y_coords if len(y_coords) > 1 else y_coords[0]

        return df

    def apply_alteration_to_row(self, row, selected_df):
        alteration_type = row['alteration_type']
        mtm_points = row['mtm points']
        
        if pd.isna(alteration_type):
            return row
        else:
            # Create a dictionary to store the alteration data for the current row.
            alteration_set = {
                "mtm_point": mtm_points,
                "mtm_dependant" : int(row['mtm_dependent']),
                "alteration": alteration_type,
                "old_coordinates": (row['pl_point_x'], row['pl_point_y']),
                "movement_x": row['movement_x'],
                "movement_y": row['movement_y'],
            }

        if alteration_type == 'X Y MOVE' and pd.notna(mtm_points):
            logging.info(f"Alteration Type: {alteration_type}")
            _, new_coordinates = self.apply_xy_coordinate_adjustment(row, selected_df)
            
            # Update Alteration Set
            alteration_set["new_coordinates"] = (new_coordinates)     
            alteration_set["altered_vertices"] = [(row['pl_point_x'], row['pl_point_y'])]      
            alteration_set["mtm_points_in_altered_vertices"] = row['mtm_points_in_altered_vertices'] 
            alteration_set["altered_vertices_smoothened"] = [(new_coordinates)]
            alteration_set["altered_vertices_smoothened_reduced"] = [(new_coordinates)]
            alteration_set["split_vertices"] = {}
            
        elif alteration_type in ['CW Ext', 'CCW Ext'] and pd.notna(mtm_points):
            # With Extension: Curve length will change, which means MTM points move in stright X,Y Directions
            logging.info(f"Alteration Type: {alteration_type}")

        elif alteration_type == "CCW No Ext" and pd.notna(mtm_points):
            # No Extension: Curve Length Stays the same 
            # This means the MTM points won't move in straight X,Y Directions.
            logging.info(f"Alteration Type: {alteration_type}")
        else:
            logging.warning(f"No Viable Alteration Types Found")

        self.accumulate_alterations(alteration_set)

        return row

    def apply_xy_coordinate_adjustment(self, row, selected_df):
        """
        Applies an XY movement alteration to the current row by adjusting X and Y coordinates.
        This method modifies the original X and Y coordinates based on movement percentages 
        and updates the altered vertices list with nearest points.
        
        :param row: The row containing point information and movement data.
        :param selected_df: The DataFrame containing all points for the current alteration rule.
        :return: The updated row with modified coordinates and nearest points.
        """

        # Move X/Y Coordinates as a function of 1 inch 
        row['pl_point_x_modified'] = row['pl_point_x'] + (1 * row['movement_x'])
        row['pl_point_y_modified'] = row['pl_point_y'] + (1 * row['movement_y'])

        # Extract the current point's identifier ('mtm points') from the row.
        mtm_point = row['mtm points']

        # Call the 'find_closest_points' method to identify the two closest points (prev and next).
        # This method excludes the current 'mtm_point' from the search to avoid comparing it to itself.
        prev_point, next_point = self.find_closest_points(mtm_point, row['pl_point_x'], row['pl_point_y'], selected_df)

        # Extract the coordinates of the previous closest point (if it exists).
        prev_coordinates = (prev_point['pl_point_x'], prev_point['pl_point_y']) if prev_point is not None else (None, None)
        
        # Extract the coordinates of the next closest point (if it exists).
        next_coordinates = (next_point['pl_point_x'], next_point['pl_point_y']) if next_point is not None else (None, None)

        # The altered coordinates of the current point after applying the movement.
        altered_coordinates = (row['pl_point_x_modified'], row['pl_point_y_modified'])

        # Construct a list that stores information about:
        # - The previous closest point and its original coordinates.
        # - The current point's original and altered coordinates.
        # - The next closest point and its original coordinates.
        mtm_points_in_altered_vertices = [
            {'mtm_point': prev_point['mtm points'] if prev_point is not None else None, 'original_coordinates': prev_coordinates},
            {'mtm_point': mtm_point, 'original_coordinates': (row['pl_point_x'], row['pl_point_y']), 'altered_coordinates': altered_coordinates},
            {'mtm_point': next_point['mtm points'] if next_point is not None else None, 'original_coordinates': next_coordinates}
        ]

        # Update the 'mtm_points_in_altered_vertices' field in the row with the new data.
        row['mtm_points_in_altered_vertices'] = mtm_points_in_altered_vertices

        # Check if 'altered_vertices' is an empty or invalid string (e.g., 'None', 'nan', etc.).
        # If it is empty, initialize it as a list containing the current altered coordinates.
        if isinstance(row['altered_vertices'], str) and row['altered_vertices'] in ['', 'nan', 'None', 'NaN']:
            row['altered_vertices'] = [altered_coordinates]
        
        # If 'altered_vertices' already exists, append the current altered coordinates to the list.
        else:
            # If 'altered_vertices' is a string representing a list, convert it into an actual list using ast.literal_eval.
            altered_vertices = ast.literal_eval(row['altered_vertices']) if isinstance(row['altered_vertices'], str) else row['altered_vertices']
            
            # Append the current altered coordinates to the list of altered vertices.
            altered_vertices.append(altered_coordinates)
            
            # Update the 'altered_vertices' field with the modified list.
            row['altered_vertices'] = altered_vertices

        # Return the updated row with the modified coordinates and altered vertices.
        return row, altered_coordinates
    
    def accumulate_alterations(self, alt_set):
        """
        Update or add new alterations to the cumulative tracking list.
        """
        for existing in self.alteration_log:
            if existing['mtm_point'] == alt_set['mtm_point']:

                # If an alteration already exists for this MTM point, update its values
                existing['movement_x'] += alt_set['movement_x']
                existing['movement_y'] += alt_set['movement_y']
                
                # Recalculate the final coordinates based on the accumulated movements
                existing['new_coordinates'] = (
                    existing['old_coordinates'][0] * (1 + existing['movement_x']),
                    existing['old_coordinates'][1] * (1 + existing['movement_y'])
                )
                
                # Combine altered vertices and mtm points
                existing['altered_vertices'] += alt_set['altered_vertices']
                existing['mtm_points_in_altered_vertices'] += alt_set['mtm_points_in_altered_vertices']
                
                return  # Stop after updating the existing entry
            
        # If no existing entry found, add new entry
        self.alteration_log.append(alt_set)

    def find_closest_points(self, mtm_point, current_x, current_y, selected_df):
        # Create a copy of the selected_df to ensure the original DataFrame isn't modified.
        selected_df_copy = selected_df.copy()

        # Drop rows with missing values for 'mtm points', 'pl_point_x', or 'pl_point_y'.
        df_existing_mtm = selected_df_copy.dropna(subset=['mtm points', 'pl_point_x', 'pl_point_y'])    

        # Calculate the Euclidean distance between the current point (current_x, current_y) and each point in the DataFrame.
        df_existing_mtm['distance'] = np.sqrt(
            (df_existing_mtm['pl_point_x'] - current_x) ** 2 + (df_existing_mtm['pl_point_y'] - current_y) ** 2
        )

        # Exclude the current 'mtm_point' from the comparison (we don't want to compare the point to itself).
        df_sorted = df_existing_mtm[df_existing_mtm['mtm points'] != mtm_point].sort_values(by='distance')

        # Get the two closest points by selecting the first two rows in the sorted DataFrame.
        prev_point = df_sorted.iloc[0] if not df_sorted.empty else None
        next_point = df_sorted.iloc[1] if len(df_sorted) > 1 else None

        # Return the two closest points.
        return prev_point, next_point

    def filter_notch_points(self, perimeter_threshold=0.1575, tolerance=0.50, angle_threshold=60.0, small_displacement_threshold=0.1):
        """
        Identifies and removes notch points from the list of vertices. A notch is defined as three points
        forming a perimeter approximately equal to the given threshold and with sharp angles.

        :param perimeter_threshold: The perimeter threshold in inches (default is 0.1575 inches for 4mm).
        :param tolerance: The allowed tolerance when comparing perimeters.
        :param angle_threshold: The angle threshold to detect sharp turns representing notches.
        :param small_displacement_threshold: The minimum threshold for the displacement to avoid retaining points with
                                            very small changes in coordinates (removes unwanted sharp points).
        :return: Updated DataFrame with notch points removed from the vertices.
        """
        logging.info(f"Starting notch point removal process.")
        total_notches_removed = 0

        for index, row in self.vertices_df.iterrows():
            vertex_list = ast.literal_eval(row['vertices'])
            if len(vertex_list) < 3:
                continue

            # Copy the vertex_list to work on
            cleaned_vertex_list = vertex_list.copy()

            # Iterate through vertex list and analyze points for proximity, angles, and dips
            for i in range(1, len(vertex_list) - 1):
                p1, p2, p3 = vertex_list[i - 1], vertex_list[i], vertex_list[i + 1]

                # Calculate distances
                d12 = np.linalg.norm(np.array(p1) - np.array(p2))
                d23 = np.linalg.norm(np.array(p2) - np.array(p3))
                d31 = np.linalg.norm(np.array(p3) - np.array(p1))

                # Calculate the angle between the points
                angle = self.calculate_angle(p1, p2, p3)

                # Calculate the perimeter (sum of the distances)
                perimeter = d12 + d23 + d31

                # Check for sharp angle (notch) and perimeter size
                if abs(angle) < angle_threshold and (perimeter) < (perimeter_threshold + tolerance):
                    logging.info(f"Notch detected at points: {p1}, {p2}, {p3} with angle: {angle:.2f} degrees")
                    total_notches_removed += 1

                    # Remove notch points
                    if p1 in cleaned_vertex_list:
                        cleaned_vertex_list.remove(p1)
                    if p2 in cleaned_vertex_list:
                        cleaned_vertex_list.remove(p2)
                    if p3 in cleaned_vertex_list:
                        cleaned_vertex_list.remove(p3)

                # Additional check for small displacement points
                if self.is_small_displacement(p1, p2, p3, small_displacement_threshold):
                    logging.info(f"Small displacement detected at point: {p2}. Removing.")
                    if p2 in cleaned_vertex_list:
                        cleaned_vertex_list.remove(p2)

            # Update the DataFrame row with the cleaned vertex list
            self.vertices_df.at[index, 'vertices'] = str(cleaned_vertex_list)

        logging.info(f"Total notches removed: {total_notches_removed}")
        return self.vertices_df

    def is_small_displacement(self, p1, p2, p3, threshold):
        """
        Check if the points form a very small displacement, indicating they are part of an unwanted artifact (sharp perpendicular line).
        
        :param p1: First point (x, y).
        :param p2: Second point (vertex).
        :param p3: Third point (x, y).
        :param threshold: Threshold for detecting small changes.
        :return: True if the displacement is small, False otherwise.
        """
        # Displacement between consecutive points
        disp_x = abs(p2[0] - p1[0]) + abs(p3[0] - p2[0])
        disp_y = abs(p2[1] - p1[1]) + abs(p3[1] - p2[1])
        
        # If both X and Y displacements are smaller than the threshold, consider it a small unwanted artifact
        if disp_x < threshold and disp_y < threshold:
            return True
        
        return False

    def calculate_angle(self, p1, p2, p3):
        """
        Calculate the angle (in degrees) formed by three points (p1, p2, p3).

        :param p1: First point (x, y).
        :param p2: Second point (vertex).
        :param p3: Third point (x, y).
        :return: Angle in degrees.
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        # Vector from p2 to p1 and from p2 to p3
        ba = a - b
        bc = c - b

        # Calculate the cosine of the angle between ba and bc
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        # Handle potential floating-point issues
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        # Return the angle in degrees
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def process_and_save_vertices(self):
        """
        Prepares the vertices, reduces them, flattens, and saves the processed vertices.
        """
        logging.info(f"\nProcessing Vertices")

        # Prepare vertices by removing notch points
        self.filter_notch_points()

        # Prepare vertices by reducing them (Old Method: Does not maintain the right dimensions)
        # When i remove this, it causes problemos
        #self.vertices_df = self.vertices_df.copy().apply(self.reduce_vertices, axis=1)

        # Save processed vertices
        self.save_processed_vertices()

        # Process vertices into a flattened list and remove duplicates
        vertices_nested_list = self.extract_and_flatten_vertices(self.vertices_df['vertices'].tolist())
        self.processed_vertices_list = self.processing_utils.remove_duplicates_preserve_order(vertices_nested_list)

    def save_processed_vertices(self):
        """
        Saves the processed vertices to the specified folder.
        """
        os.makedirs(self.save_folder_processed_vertices, exist_ok=True)
        save_filepath = f"{self.save_folder_processed_vertices}/processed_vertices_{self.piece_name}{self.save_file_format}"
        self.vertices_df.to_csv(save_filepath, index=False)

    def extract_and_flatten_vertices(self, vertices_string_list):
        """
        Converts the string representation of vertices to lists and flattens them into a single list.
        """
        # Convert strings to lists
        vertices_nested_list = [ast.literal_eval(vertices) for vertices in vertices_string_list]

        # Flatten the list of lists into a single list of coordinates
        return [vertex for sublist in vertices_nested_list for vertex in sublist]

    def reduce_vertices(self, row):
        """
        Reduces the number of vertices for each row by simplifying the shape.

        This function parses the 'vertices' column, applies a reduction algorithm using a threshold 
        to simplify the list of vertices while preserving the overall shape.

        :param row: A row from the vertices DataFrame.
        :return: The modified row with reduced vertices in 'original_vertices_reduced'.
        """
        if pd.isna(row['vertices']) or row['vertices'] in ['nan', 'None', '', 'NaN']:
            row['original_vertices_reduced'] = []
        else:
            try:
                vertices_list = ast.literal_eval(row['vertices'])  # Convert string representation of list to actual list

                # Check if the vertices_list is a valid list of 2D points
                if isinstance(vertices_list, list) and all(isinstance(vertex, (list, tuple)) and len(vertex) == 2 for vertex in vertices_list):
                    reduced_vertices = self.processing_utils.reduce_points(vertices=vertices_list, threshold=0.1)
                    row['original_vertices_reduced'] = [tuple(vertex) for vertex in reduced_vertices]  # Ensure list of tuples
                else:
                    logging.warning(f"Invalid format in 'vertices' column for row: {row}")
                    row['original_vertices_reduced'] = []
            except (ValueError, SyntaxError) as e:
                logging.error(f"Error processing vertices for row: {row}, Error: {e}")
                row['original_vertices_reduced'] = []
        return row
    
    def log_info(self):
        """
        Log Alteration Info (for Debugging)
        """
        logging.info(f"\nMake alteration on Piece Name: {self.piece_name}")

        # Define the alteration type you're looking for
        alteration_rule = "7F-SHPOINT"

        # Check if the alteration type exists in the alteration DataFrame
        if alteration_rule in self.alteration_dfs:
            logging.info(f"\nAlteration DFs on Alteration Rule {alteration_rule}:\n{self.alteration_dfs[alteration_rule]}")
        else:
            logging.warning(f"\nAlteration Rule '{alteration_rule}' not found in the DataFrame.")

        #print(f"\nProcessed Vertices List:")
        #print(self.processed_vertices_list)

if __name__ == "__main__":

    piece_table_path = "data/staging/alteration_by_piece/combined_table_LGFG-SH-01-CCB-FO.csv"
    vertices_table_path = "data/staging/vertices/vertices_LGFG-SH-01-CCB-FO.csv"

    #piece_table_path = "data/staging/alteration_by_piece/combined_table_CIRCLE-12BY12-INCH.csv"
    #vertices_table_path = "data/staging/vertices/vertices_CIRCLE-12BY12-INCH.csv"

    #piece_table_path = "data/staging/alteration_by_piece/combined_table_LGFG-V2-SH-01-STBS-F.csv"
    #vertices_table_path = "data/staging/vertices/vertices_LGFG-V2-SH-01-STBS-F.csv"

    #piece_table_path = "data/staging/alteration_by_piece/combined_table_LGFG-SH-04FS-FOA.csv"
    #vertices_table_path = "data/staging/vertices/vertices_LGFG-SH-04FS-FOA.csv"
    
    # Debug: Check by Alteration Rule
    #debug_alteration_rule = "7F-SHPOINT"
    #debug_alteration_rule = "4-WAIST"
    debug_alteration_rule = "1LTH-FULL"
    #debug_alteration_rule = "1LTH-FSLV"

    make_alteration = PieceAlterationProcessor(piece_table_path=piece_table_path,
                                               vertices_table_path=vertices_table_path,
                                               debug_alteration_rule=debug_alteration_rule)
    make_alteration.process_alterations()
    #make_alteration.log_info()

