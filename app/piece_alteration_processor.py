import pandas as pd
import ast
import numpy as np
from app.smoothing import SmoothingFunctions  # Import the SmoothingFunctions class
from utils.data_processing_utils import DataProcessingUtils
import os
from functools import partial

# Setup Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
) # Levels Info/Warning/Error/Critical

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
                save_folder_processed_vertices="data/staging_processed/processed_vertices/", 
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
    
    def process_alterations(self):
        """
        Applies alteration rules to each DataFrame and processes vertices. If debug_alteration_rule is set,
        it only applies the rule for that type. If not, it runs through all rules.
        """
        # Prepare vertices
        self.process_and_save_vertices()

        def apply_single_rule(alteration_rule):
            """Helper function to apply a single alteration rule."""
            selected_df = self.alteration_dfs[alteration_rule]
            selected_df = self.prepare_dataframe(selected_df)
            return selected_df.apply(partial(self.process_single_row, selected_df=selected_df), axis=1)

        # Case when a specific debug alteration type is provided
        if not pd.isna(self.debug_alteration_rule):
            if self.debug_alteration_rule in self.alteration_rules:
                # Apply the specific debug alteration rule
                alteration_df = apply_single_rule(self.debug_alteration_rule)
            else:
                logging.warning(f"Debug alteration type '{self.debug_alteration_rule}' not found in requested alterations.")
            return  # Early return as only the specific debug alteration rule is applied

        # Apply all alteration rules when no specific debug type is provided
        for alteration_rule in self.alteration_rules:
            alteration_df = apply_single_rule(alteration_rule)
        
        cumulative_alteration_df = pd.DataFrame(self.alteration_log)

    def process_single_row(self, row, selected_df):
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
            new_coordinates = self.apply_xy_coordinate_adjustment(row, selected_df)
            
            # Update Alteration Set
            alteration_set["new_coordinates"] = (new_coordinates)            
            print(alteration_set)
            
        elif alteration_type in ['CW Ext', 'CCW Ext'] and pd.notna(mtm_points):
            # With Extension: Curve length will change, which means MTM points move in stright X,Y Directions
            logging.info(f"Alteration Type: {alteration_type}")
        elif alteration_type == "CCW No Ext" and pd.notna(mtm_points):
            # No Extension: Curve Length Stays the same 
            # This means the MTM points won't move in straight X,Y Directions.
            logging.info(f"Alteration Type: {alteration_type}")
        else:
            logging.warning(f"No Viable Alteration Types Found")

    def apply_xy_coordinate_adjustment(self, row, selected_df):
        """
        Applies an XY movement alteration to the current row by adjusting X and Y coordinates.
        This method modifies the original X and Y coordinates based on movement percentages 
        and updates the altered vertices list with nearest points.
        
        :param row: The row containing point information and movement data.
        :param selected_df: The DataFrame containing all points for the current alteration rule.
        :return: The updated row with modified coordinates and nearest points.
        """
        # Modify the X coordinate using the movement percentage in 'movement_x'.
        # New X = Original X * (1 + movement_x)
        row['pl_point_x_modified'] = row['pl_point_x'] * (1 + row['movement_x'])
        
        # Modify the Y coordinate using the movement percentage in 'movement_y'.
        # New Y = Original Y * (1 + movement_y)
        row['pl_point_y_modified'] = row['pl_point_y'] * (1 + row['movement_y'])

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

    def process_and_save_vertices(self):
        """
        Prepares the vertices, reduces them, flattens, and saves the processed vertices.
        """
        # Prepare vertices by reducing them
        self.vertices_df = self.vertices_df.copy().apply(self.reduce_vertices, axis=1)

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
    
    # Debug: Check by Alteration Rule
    #debug_alteration_rule = "7F-SHPOINT"
    #debug_alteration_rule = "4-WAIST"
    debug_alteration_rule = "1LTH-FULL"

    make_alteration = PieceAlterationProcessor(piece_table_path=piece_table_path,
                                               vertices_table_path=vertices_table_path,
                                               debug_alteration_rule=debug_alteration_rule)
    make_alteration.process_alterations()
    #make_alteration.log_info()

