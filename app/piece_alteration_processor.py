import pandas as pd
import ast
import numpy as np
from app.smoothing import SmoothingFunctions  # Import the SmoothingFunctions class
from utils.data_processing_utils import DataProcessingUtils
import os
from functools import partial
from itertools import combinations

# TODO:
# MTM Points in altered vertices are all the MTM points that are moved
# Is this needed when we get new pl points?

# TODO:
# IF two times mention the same point, move that point only (XY Move)
# IF Two different points, then move points from dependent to the mtm 8017 to 8020 (including both points)

# Setup Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
) # Levels Info/Warning/Error/Critical

# Debug
pd.set_option('display.max_colwidth', None)  # No limit on column width
#pd.set_option('display.max_columns', None)   # Display all columns
#pd.set_option('display.width', None)         # No truncation on width


# TODO: Merge all notch points: DONE
# TODO: When Reducing the Points, THE MTM POINTS DO NOT LOOK EXACTLY ALIGNED ON THE PLOT. DONE

# TODO: FIND OUT WHY NOT ALL XY MOVE FOR 1LTH-FULL APPEARS FOR FRONT PIECE
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
        self.altered_df = ""

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
        df['pl_point_altered_x'] = ""
        df['pl_point_altered_y'] = ""
        
        # Drop unnecessary columns
        df = df.drop('vertices', axis=1)
        
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
                self.altered_df = apply_single_rule(self.debug_alteration_rule)
                self.processing_utils.save_csv(self.altered_df, save_filepath)
            else:
                logging.warning(f"Debug alteration type '{self.debug_alteration_rule}' not found in requested alterations.")
            return  # Early return after processing the debug alteration

        # Apply all alteration rules when no specific debug type is provided
        logging.info("Running Full Alteration Process")
        all_altered_dfs = []
        for alteration_rule in self.alteration_rules:
            altered_df = self.apply_single_rule(alteration_rule)  # This should return the altered DataFrame
            all_altered_dfs.append(altered_df)

        # Optionally combine all the altered DataFrames
        self.altered_df = pd.concat(all_altered_dfs, ignore_index=True)

    def apply_alteration_to_row(self, row, selected_df):
        alteration_type_map = {
            'X Y MOVE': self.apply_xy_coordinate_adjustment,
            'CW Ext': self.apply_cw_ext,
            'CCW Ext': self.apply_ccw_ext,
            'CCW No Ext': self.apply_ccw_no_ext,
        }

        alteration_type = row['alteration_type']
        
        if pd.isna(alteration_type):
            return row
        
        # Call the appropriate method for the alteration type
        func = alteration_type_map.get(alteration_type)
        if func:
            return func(row, selected_df)
        else:
            logging.warning(f"No viable alteration types found for {alteration_type}")
            return row
        
    def apply_cw_ext(self):
        print(f"CW Ext: Nothing here yet")

    def apply_ccw_ext(self):
        print(f"CCW EXT Nothing here yet")

    def apply_ccw_no_ext(self):
        print(f"CCW No Ext Nothing here yet")

    def find_points_in_range(self, df, x1, y1, x2, y2):
        """
        Finds all points in the DataFrame that lie between two given coordinates (x1, y1) and (x2, y2).

        :param df: The DataFrame containing the coordinates (columns 'pl_point_x' and 'pl_point_y').
        :param x1, y1: First coordinate.
        :param x2, y2: Second coordinate.
        :return: A DataFrame with the points that lie between (x1, y1) and (x2, y2).
        """

        # Determine the bounding box (min and max for x and y)
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        # Filter the DataFrame to get all points within the bounding box
        points_in_range = df[
            (df['pl_point_x'] >= min_x) & (df['pl_point_x'] <= max_x) &
            (df['pl_point_y'] >= min_y) & (df['pl_point_y'] <= max_y)
        ]

        return points_in_range

    def apply_xy_coordinate_adjustment(self, row, selected_df):
        """
        Applies an XY movement alteration to the current row by adjusting X and Y coordinates.
        This method modifies the original X and Y coordinates based on movement percentages 
        and updates the altered vertices list with nearest points.
        
        :param row: The row containing point information and movement data.
        :param selected_df: The DataFrame containing all points for the current alteration rule.
        :return: The updated row with modified coordinates and nearest points.
        """

        try:
            # Extract the current point's identifier ('mtm points') from the row.
            mtm_point = row['mtm points']
            mtm_dependent = row['mtm_dependent']

            coords = (row['pl_point_x'], row['pl_point_y'])

            # IF MTM Dependent Point is equal to MTM Alteration Point, then only move that point
            if mtm_dependent == mtm_point:
                # Move X/Y Coordinates as a function of 1 inch 
                row['pl_point_altered_x'] = row['pl_point_x'] + (1 * row['movement_x'])
                row['pl_point_altered_y'] = row['pl_point_y'] + (1 * row['movement_y'])

                coords_new = (row['pl_point_altered_x'], row['pl_point_altered_y'])

                logging.info(f"Altered point {mtm_point} from {coords} to {coords_new}")

                return row  # Return the altered row
            
            elif mtm_dependent != mtm_point:
                # Find the dependent point's pl_point_x and pl_point_y in the selected_df
                dependent_row = selected_df[selected_df['mtm points'] == mtm_dependent]

                if dependent_row.empty:
                    logging.error(f"Dependent point {mtm_dependent} not found for MTM Point {mtm_point}.")
                    return row

                # Alter all points between

            return row
        
        except Exception as e:
            logging.error(f"Failed to apply XY adjustment: {e}")

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

        ##### Processing Functions #####
        # Prepare vertices by removing notch points
        #self.filter_notch_points()

        # Prepare vertices by reducing them (Old Method: Does not maintain the right dimensions)
        # When i remove this, it causes problemos
        #self.vertices_df = self.vertices_df.copy().apply(self.reduce_vertices, axis=1)
        #########

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
        :return: The altered row with reduced vertices in 'original_vertices_reduced'.
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
    
    def log_info(self, debug_alteration_rule):
        """
        Log Alteration Info (for Debugging)
        """
        logging.info(f"\nMake alteration on Piece Name: {self.piece_name}")

        # Define the alteration type you're looking for
        alteration_rule = debug_alteration_rule

        # Check if the alteration type exists in the alteration DataFrame
        if alteration_rule in self.alteration_dfs:
            # Get the DataFrame for the specific alteration rule
            df = self.alteration_dfs[alteration_rule]

            # Filter the DataFrame for rows where 'alteration_rule' is not NaN
            non_nan_rows = df[df['alteration_rule'].notna()]

            # Log the non-NaN rows
            logging.info(f"\nAlteration DFs on Alteration Rule {alteration_rule} (Non-NaN rows):\n{non_nan_rows}")
            logging.info(f"\nNumber of Alterations: {len(non_nan_rows)}")
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
    #make_alteration.log_info(debug_alteration_rule)

