import pandas as pd
import ast
import numpy as np
from app.smoothing import SmoothingFunctions  # Import the SmoothingFunctions class
from utils.data_processing_utils import DataProcessingUtils
import os
from functools import partial
from itertools import combinations

# TODO: Fix Check further alteration point logic.. It breaks when I change pieces

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
pd.set_option('display.max_columns', None)   # Display all columns
pd.set_option('display.width', None)         # No truncation on width

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
                debug_alteration_rule = None,
                alteration_movement = 1):
        
        self.processing_utils = DataProcessingUtils()

        # Load the piece and vertices data
        self.piece_df = self.processing_utils.load_csv(piece_table_path)
        self.vertices_df = self.processing_utils.load_csv(vertices_table_path)
        self.processed_vertices_list = []
        self.line_pl_points = [] # Used to store line points

        # Get the unique alteration rules
        self.alteration_rules = self.get_alteration_rules()

        # Dictionary to store DataFrames by alteration rule
        self.alteration_dfs = self.split_by_alteration_rule()
        self.altered_df = ""

        # Piece name (assuming only one unique piece name)
        self.piece_name = self.get_piece_name()

        # Alteration settings
        self.alteration_movement = alteration_movement

        # Save Options
        self.save_folder_processed_pieces = save_folder_processed_pieces
        self.save_folder_processed_vertices = save_folder_processed_vertices
        self.save_file_format = save_file_format

        # Debugging
        self.debug_alteration_rule = debug_alteration_rule

        # Counters: Keeps track of how many times an alteration type is applied
        # Will be set when a rule is found
        self.xy_move_count = 0 
        self.cw_ext_count = 0
        self.ccw_ext_count = 0
        self.ccw_no_ext_count = 0
        # Will be set iteratively per alteration type
        self.xy_move_step_counter = 0
        self.cw_ext_step_counter = 0
        self.ccw_ext_step_counter = 0
        self.ccw_no_ext_step_counter = 0

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
    
    def remove_line_pl_points(self, df):
        """
        Identifies and removes rows in `df` where the `pl_point_x` and `pl_point_y` correspond
        to line segments (vertices with exactly two points) from the `vertices_df`.
        Stores the removed line points in `self.line_pl_points` for later use.

        :param df: The DataFrame from which line points should be removed.
        :return: The DataFrame with line points removed.
        """
        line_pl_points = []

        # Iterate through each row in the vertices DataFrame
        for index, row in self.vertices_df.iterrows():
            try:
                # Parse the vertices (convert string representation to actual list)
                vertices = ast.literal_eval(row['vertices'])
                
                # Check if the length of the vertices is exactly 2 (line segment)
                if len(vertices) == 2:
                    # Extract the pl_points (coordinates) of the vertex line
                    pl_point_1 = vertices[0]
                    pl_point_2 = vertices[1]
                    
                    # Store both points for later use
                    line_pl_points.append(pl_point_1)
                    line_pl_points.append(pl_point_2)

            except Exception as e:
                logging.error(f"Error processing vertices for row {index}: {e}")
        
        # Convert the list of line pl_points to a DataFrame and store in an instance variable
        self.line_pl_points = pd.DataFrame(line_pl_points, columns=['pl_point_x', 'pl_point_y'])

        # Remove rows from df where `pl_point_x` and `pl_point_y` match any of the points in self.line_pl_points
        df_cleaned = df[~df.set_index(['pl_point_x', 'pl_point_y']).index.isin(self.line_pl_points.set_index(['pl_point_x', 'pl_point_y']).index)]

        logging.info(f"Removed {len(df) - len(df_cleaned)} rows corresponding to line PL points.")
        
        return df_cleaned

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

        # Remove PL Points of lines
        df = self.remove_line_pl_points(df)
        
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

        # Remove line PL Points
        
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
    
    def mark_notch_points(self, notch_points):
        """
        Marks the identified notch points in the altered DataFrame by creating a new column 
        that labels each notch point with a unique identifier.

        :param notch_points: List of identified notch points (each entry is a tuple of the three points forming a notch).
        :return: DataFrame with a new column 'notch_labels' indicating the identified notch points.
        """
        df = self.altered_df

        # Add a new column for marking notch points if it doesn't exist
        if 'notch_labels' not in df.columns:
            df['notch_labels'] = ''

        logging.info(f"Starting to mark notch points. Total notch points identified: {len(notch_points)}")

        # Iterate through the identified notch points
        for notch in notch_points:
            p1, p2, p3 = notch

            logging.info(f"Marking notch: ({p1}, {p2}, {p3})")

            # Find and mark each of the three points in the DataFrame
            for point in [p1, p2, p3]:
                # Locate the rows that match the notch point coordinates
                match_condition = (df['pl_point_x'] == point[0]) & (df['pl_point_y'] == point[1])
                if not df[match_condition].empty:
                    # Check if 'notch' is already in the 'notch_labels' column for the point
                    if 'notch' not in df.loc[match_condition, 'notch_labels'].values[0]:
                        df.loc[match_condition, 'notch_labels'] += 'notch'
                        logging.info(f"Marked notch point: {point} in DataFrame.")
                    else:
                        logging.info(f"Duplicate notch point: {point} already marked, skipping.")
                else:
                    logging.warning(f"Notch point {point} not found in DataFrame.")

        logging.info("Completed marking notch points.")
        
        # Return the DataFrame with labeled notch points
        return df

    def remove_empty_rows(self, df):
        """
        Removes rows from the DataFrame where all values (except 'piece_name') are either NaN or zero.
        
        :param df: The DataFrame to process.
        :return: The cleaned DataFrame with empty rows removed.
        """
        # Step 1: Replace empty strings with NaN for uniform handling
        df.replace("", np.nan, inplace=True)

        # Step 2: Check if each element in a row (excluding 'piece_name') is either NaN or zero
        is_nan_or_zero = df.drop(columns=['piece_name']).applymap(lambda x: pd.isna(x) or x == 0)

        # Step 3: Keep rows where not all values in the row (excluding 'piece_name') are NaN or zero
        cleaned_df = df.loc[~is_nan_or_zero.all(axis=1)]

        return cleaned_df

    def process_alterations(self):
        """
        Applies alteration rules to each DataFrame and processes vertices. If a debug_alteration_rule is set,
        only that rule is applied. Otherwise, all alteration rules are processed.
        """
        # Prepare vertices and get notch points
        notch_points = self.process_and_save_vertices()

        def apply_single_rule(alteration_rule):
            """
            Applies a single alteration rule by combining the selected and unselected rows.
            Prepares the combined DataFrame and processes it row-by-row.
            """
            selected_df = self.alteration_dfs[alteration_rule]
            unselected_df = self.get_unselected_rows(selected_df)

            # Combine selected and unselected rows into a single DataFrame
            combined_df = self.combine_and_clean_data(selected_df, unselected_df)
            
            # Prepare DataFrame by converting columns and dropping unnecessary ones
            processed_df = self.prepare_dataframe(combined_df)

            # Assuming you have your DataFrame loaded as `df`
            alteration_type_counts = processed_df['alteration_type'].value_counts()
            self.xy_move_count = alteration_type_counts.get('X Y MOVE', 0)  # Defaults to 0 if 'X Y MOVE' doesn't exist
            self.cw_ext_count = alteration_type_counts.get('CW Ext', 0)
            self.ccw_ext_count = alteration_type_counts.get('CCW Ext', 0)
            self.ccw_no_ext_count = alteration_type_counts.get('CCW No Ext', 0)

            # Make sure new dataset is sorted
            #processed_df = processed_df.sort_values(by='mtm points')

            # Apply alteration rule row-by-row
            for index, row in processed_df.iterrows():

                # Only apply alterations where types exist
                if not pd.isna(row['alteration_type']):
                    logging.info(f"Processing Alteration Type: {row['alteration_type']}")

                    # Apply the alteration and update both the row and the selected_df
                    row, updated_df = self.apply_alteration_to_row(row, processed_df)
                
                    # Update the row back into processed_df (in case it's an individual row alteration)
                    processed_df.loc[index] = row
                
                    # Ensure any DataFrame-wide changes are applied to processed_df
                    processed_df = updated_df

            # Check for further alteration points
            #if self.xy_move_step_counter == self.xy_move_step_counter:
            #    processed_df = self.apply_mtm_correction(processed_df)
            
            # Remove empty rows
            processed_df = self.remove_empty_rows(processed_df)

            return processed_df

        # Debug Mode: Apply a specific alteration rule and save the result
        if not pd.isna(self.debug_alteration_rule):

            # Setup Save Info
            debug_savefolder = "data/staging_processed/debug/"
            save_filepath = f"{debug_savefolder}{self.piece_name}_{self.debug_alteration_rule}{self.save_file_format}"
            os.makedirs(debug_savefolder, exist_ok=True)

            if self.debug_alteration_rule in self.alteration_rules:
                # Apply the specific debug alteration rule
                logging.info(f"Running Alteration in Debug Mode: {self.debug_alteration_rule}")
                self.altered_df = apply_single_rule(self.debug_alteration_rule)
                self.mark_notch_points(notch_points)
                self.processing_utils.save_csv(self.altered_df, save_filepath)
            else:
                logging.warning(f"Debug alteration type '{self.debug_alteration_rule}' not found in requested alterations.")
            return  # Early return after processing the debug alteration
        
        #logging.info("Running Full Alteration Process")
        #all_altered_dfs = []

        # Loop through all alteration rules (not types!)
        #for alteration_rule in self.alteration_rules:
            #selected_df = self.alteration_dfs[alteration_rule]

            # Apply the alteration to every row in the selected_df
            #for index, row in selected_df.iterrows():
                # Apply the alteration and return both row and selected_df
            #    row, selected_df = self.apply_alteration_to_row(row, selected_df)
                # Update the row back into selected_df
            #    selected_df.loc[index] = row

            # Find closest altered Row 
            #for index, row in selected_df.iterrows():
            #    closest_point, closest_distance = self.find_closest_altered_point(row, selected_df)
            #    print(closest_point)
            #    if closest_point is not None:
            #        logging.info(f"Closest altered point to {row['mtm points']} is {closest_point['mtm points']} at distance {closest_distance:.2f}.")

            # Append the fully altered DataFrame for this rule
            #all_altered_dfs.append(selected_df)

        # Concatenate all the altered DataFrames
        #self.altered_df = pd.concat(all_altered_dfs, ignore_index=True)

        # Save the final altered DataFrame to CSV
        #save_filepath = f"{self.save_folder_processed_pieces}/proces sed_alterations_{self.piece_name}{self.save_file_format}"
        #self.processing_utils.save_csv(self.altered_df, save_filepath)
        #logging.info(f"Altered DataFrame saved to {save_filepath}")

    def apply_alteration_to_row(self, row, selected_df):
        """
        Applies the relevant alteration type to a row or the selected DataFrame.
        Handles both individual row updates and DataFrame-level updates.
        
        :param row: The current row to alter.
        :param selected_df: The entire DataFrame being altered.
        :return: A tuple of (row, selected_df) after the alterations.
        """
        alteration_type_map = {
            'X Y MOVE': self.apply_xy_coordinate_adjustment,
            'CW Ext': self.apply_cw_ext,
            'CCW Ext': self.apply_ccw_ext,
            'CCW No Ext': self.apply_ccw_no_ext,
        }

        alteration_type = row['alteration_type']
        
        if pd.isna(alteration_type):
            return row, selected_df  # No alteration, return original row and DataFrame

        # Call the appropriate method for the alteration type
        func = alteration_type_map.get(alteration_type)
        if func:
            return func(row, selected_df)  # Expect both row and DataFrame to be updated
        else:
            logging.warning(f"No viable alteration types found for {alteration_type}")
            return row, selected_df
        
    # Get matching points for a particular MTM, now using selected_df instead of start_df
    def get_pl_points(self, matching_pt, df):
        """
        Finds the pl_point_x and pl_point_y for the given MTM point from the provided DataFrame.
        
        :param matching_pt: The MTM point for which to find coordinates.
        :param df: The DataFrame to search (typically selected_df).
        :return: Tuple of pl_point_x, pl_point_y or (None, None) if not found.
        """
        matching_row = df.loc[df['mtm points'] == matching_pt]
        if not matching_row.empty:
            pl_point_x = matching_row['pl_point_x'].values[0]
            pl_point_y = matching_row['pl_point_y'].values[0]
            return pl_point_x, pl_point_y
        return None, None

    def apply_cw_ext(self, row, selected_df):
        """
        Applies Clockwise Extension (CW Ext) between the mtm_point and mtm_dependent.

        :param row: The current row being altered.
        :param selected_df: The DataFrame containing the points for the alteration rule.
        :return: A tuple of the updated row and DataFrame.
        """
        mtm_point = row['mtm points']
        mtm_dependent = row['mtm_dependent']

        p1 = np.array([row['pl_point_x'], row['pl_point_y']])
        dependent_row = selected_df[selected_df['mtm points'] == mtm_dependent].iloc[0]
        p2 = np.array([dependent_row['pl_point_x'], dependent_row['pl_point_y']])

        # Extract all points between mtm_point and mtm_dependent in a sequential order
        points_in_range = selected_df[(selected_df['pl_point_x'] > min(p1[0], p2[0])) & 
                                    (selected_df['pl_point_x'] <= max(p1[0], p2[0]))]
        
        # Debug limits above if something goes wrong
        #print(points_in_range["mtm points"])

        # Apply XY movement to points between mtm_point and mtm_dependent
        for idx, point in points_in_range.iterrows():

            # Use altered coordinates if available, otherwise use original ones
            current_x = point['pl_point_altered_x'] if pd.notna(point['pl_point_altered_x']) and point['pl_point_altered_x'] != "" else point['pl_point_x']
            current_y = point['pl_point_altered_y'] if pd.notna(point['pl_point_altered_y']) and point['pl_point_altered_y'] != "" else point['pl_point_y']

            # Apply the movement
            altered_x = current_x + (self.alteration_movement * row['movement_x'])
            altered_y = current_y + (self.alteration_movement * row['movement_y'])

            # Update these values directly in selected_df
            selected_df.loc[idx, 'pl_point_altered_x'] = altered_x
            selected_df.loc[idx, 'pl_point_altered_y'] = altered_y
        
        # Update counter
        self.xy_move_step_counter +=1

        # Apply alteration to the mtm_point itself
        row['pl_point_altered_x'] = p1[0] + (self.alteration_movement * row['movement_x'])
        row['pl_point_altered_y'] = p1[1] + (self.alteration_movement * row['movement_y'])
        
        # Update the mtm_point in selected_df
        selected_df.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_x'] = row['pl_point_altered_x']
        selected_df.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_y'] = row['pl_point_altered_y']

        logging.info(f"Altered MTM point {mtm_point} to new coordinates: "
                    f"({row['pl_point_altered_x']}, {row['pl_point_altered_y']})")

        # Apply alteration to the mtm_dependent point itself
        dependent_row['pl_point_altered_x'] = p2[0] + (self.alteration_movement * row['movement_x'])
        dependent_row['pl_point_altered_y'] = p2[1] + (self.alteration_movement * row['movement_y'])

        # Update the mtm_dependent in selected_df
        selected_df.loc[selected_df['mtm points'] == mtm_dependent, 'pl_point_altered_x'] = dependent_row['pl_point_altered_x']
        selected_df.loc[selected_df['mtm points'] == mtm_dependent, 'pl_point_altered_y'] = dependent_row['pl_point_altered_y']

        logging.info(f"Altered dependent MTM point {mtm_dependent} to new coordinates: "
                    f"({dependent_row['pl_point_altered_x']}, {dependent_row['pl_point_altered_y']})")
        
        return row, selected_df

    def apply_ccw_ext(self, row, selected_df):

        return row, selected_df

        mtm_point = row['mtm points']
        mtm_dependent = row['mtm_dependent']

        p1 = np.array([row['pl_point_x'], row['pl_point_y']])
        dependent_row = selected_df[selected_df['mtm points'] == mtm_dependent].iloc[0]
        p2 = np.array([dependent_row['pl_point_x'], dependent_row['pl_point_y']])

        # Extract all points between mtm_point and mtm_dependent in a sequential order
        points_in_range = selected_df[(selected_df['pl_point_x'] > min(p1[0], p2[0])) & 
                                    (selected_df['pl_point_x'] <= max(p1[0], p2[0]))]
        
        # Debug limits above if something goes wrong
        #print(points_in_range["mtm points"])

        # Apply XY movement to points between mtm_point and mtm_dependent
        for idx, point in points_in_range.iterrows():

            # Use altered coordinates if available, otherwise use original ones
            current_x = point['pl_point_altered_x'] if pd.notna(point['pl_point_altered_x']) and point['pl_point_altered_x'] != "" else point['pl_point_x']
            current_y = point['pl_point_altered_y'] if pd.notna(point['pl_point_altered_y']) and point['pl_point_altered_y'] != "" else point['pl_point_y']

            # Apply the movement
            altered_x = current_x + (self.alteration_movement * row['movement_x'])
            altered_y = current_y + (self.alteration_movement * row['movement_y'])

            # Update these values directly in selected_df
            selected_df.loc[idx, 'pl_point_altered_x'] = altered_x
            selected_df.loc[idx, 'pl_point_altered_y'] = altered_y
        
        # Update counter
        self.xy_move_step_counter +=1

        # Apply alteration to the mtm_point itself
        row['pl_point_altered_x'] = p1[0] + (self.alteration_movement * row['movement_x'])
        row['pl_point_altered_y'] = p1[1] + (self.alteration_movement * row['movement_y'])
        
        # Update the mtm_point in selected_df
        selected_df.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_x'] = row['pl_point_altered_x']
        selected_df.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_y'] = row['pl_point_altered_y']

        logging.info(f"Altered MTM point {mtm_point} to new coordinates: "
                    f"({row['pl_point_altered_x']}, {row['pl_point_altered_y']})")

        # Apply alteration to the mtm_dependent point itself
        dependent_row['pl_point_altered_x'] = p2[0] + (self.alteration_movement * row['movement_x'])
        dependent_row['pl_point_altered_y'] = p2[1] + (self.alteration_movement * row['movement_y'])

        # Update the mtm_dependent in selected_df
        selected_df.loc[selected_df['mtm points'] == mtm_dependent, 'pl_point_altered_x'] = dependent_row['pl_point_altered_x']
        selected_df.loc[selected_df['mtm points'] == mtm_dependent, 'pl_point_altered_y'] = dependent_row['pl_point_altered_y']

        logging.info(f"Altered dependent MTM point {mtm_dependent} to new coordinates: "
                    f"({dependent_row['pl_point_altered_x']}, {dependent_row['pl_point_altered_y']})")
        
        return row, selected_df
    
    def apply_ccw_no_ext(self, row, selected_df):
        # Example logic for Counter-clockwise No Extension
        return row, selected_df

    def apply_xy_coordinate_adjustment(self, row, selected_df):
        """
        Applies an XY movement alteration to the current row by adjusting X and Y coordinates.
        This method modifies the original X and Y coordinates based on movement percentages.
        It will update multiple rows in the selected DataFrame, including non-MTM points 
        that fall between the first and last altered MTM points. It tracks altered points to avoid 
        applying the movement multiple times to the same points.
        
        :param row: The row containing point information and movement data.
        :param selected_df: The DataFrame containing all points for the current alteration rule.
        :param altered_points: A set of altered points' indexes to prevent multiple alterations.
        :return: A tuple of (row, selected_df, altered_points) with updated coordinates.
        """
        try:
            mtm_point = row['mtm points']
            mtm_dependent = row['mtm_dependent']

            logging.info(f"Applying XY Move on {mtm_point}")

            # Case 1: Individual row alteration (mtm_dependent == mtm_point)
            if mtm_dependent == mtm_point:

                # Use altered coordinates if available, otherwise use original ones
                current_x = row['pl_point_altered_x'] if pd.notna(row['pl_point_altered_x']) and row['pl_point_altered_x'] != "" else row['pl_point_x']
                current_y = row['pl_point_altered_y'] if pd.notna(row['pl_point_altered_y']) and row['pl_point_altered_y'] != "" else row['pl_point_y']

                # Apply the movement based on the current altered coordinates
                row['pl_point_altered_x'] = current_x + (self.alteration_movement * row['movement_x'])
                row['pl_point_altered_y'] = current_y + (self.alteration_movement * row['movement_y'])

                # Update the altered values in the DataFrame
                selected_df.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_x'] = row['pl_point_altered_x']
                selected_df.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_y'] = row['pl_point_altered_y']

                # Update Counter
                self.xy_move_step_counter +=1

                logging.info(f"Altered MTM point {mtm_point} to new coordinates: "
                            f"({row['pl_point_altered_x']}, {row['pl_point_altered_y']})")

                return row, selected_df

            # Case 2: Multiple row alteration (mtm_dependent != mtm_point)
            else:
                selected_df_copy = selected_df.copy()

                # Now, continue with operations on selected_df_copy
                p1 = selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_point, ['pl_point_x', 'pl_point_y']].values[0]
                p2 = selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_dependent, ['pl_point_x', 'pl_point_y']].values[0]
                dependent_row = selected_df_copy[selected_df_copy['mtm points'] == mtm_dependent].iloc[0]

                movement_x = row['movement_x']  
                movement_y = row['movement_y']
                
                print(f"Movement X: {movement_x}")
                print(f"Movement Y: {movement_y}")

                # Get the point_order for mtm_point
                start_point_order = selected_df_copy[selected_df_copy['mtm points'] == mtm_point]["point_order"].values[0]

                # Get the point_order for mtm_dependent
                end_point_order = selected_df_copy[selected_df_copy['mtm points'] == mtm_dependent]["point_order"].values[0]

                # Print the results
                logging.info(f"Start Point Order: {start_point_order}, End Point Order: {end_point_order}")

                # Make sure to capture points in either ascending or descending order
                if start_point_order > end_point_order:
                    points_in_range = selected_df_copy[
                        (selected_df_copy['point_order'] <= start_point_order) &
                        (selected_df_copy['point_order'] >= end_point_order)
                    ]
                else:
                    points_in_range = selected_df_copy[
                        (selected_df_copy['point_order'] >= start_point_order) &
                        (selected_df_copy['point_order'] <= end_point_order)
                    ]

                # Debug
                selected_df_copy.to_csv("data/selected_df_" + str(self.xy_move_step_counter) + ".csv")
                points_in_range.to_csv("data/points_in_range_" + str(self.xy_move_step_counter) + ".csv")
                
                # Apply XY movement to points between mtm_point and mtm_dependent
                for idx, point in points_in_range.iterrows():
                    # Use the index of the row in selected_df_copy, not the idx from points_in_range
                    copy_idx = point.name  # This gets the correct index for selected_df_copy
                    
                    # Use altered coordinates if available, otherwise use original ones
                    current_x = point['pl_point_altered_x'] if pd.notna(point['pl_point_altered_x']) and point['pl_point_altered_x'] != "" else point['pl_point_x']
                    current_y = point['pl_point_altered_y'] if pd.notna(point['pl_point_altered_y']) and point['pl_point_altered_y'] != "" else point['pl_point_y']

                    # Apply the movement
                    altered_x = current_x + (self.alteration_movement * movement_x)
                    altered_y = current_y + (self.alteration_movement * movement_y)

                    # Update these values directly in selected_df_copy using the correct index
                    selected_df_copy.loc[copy_idx, 'pl_point_altered_x'] = altered_x
                    selected_df_copy.loc[copy_idx, 'pl_point_altered_y'] = altered_y
                
                selected_df_copy.to_csv("data/altered_df_" + str(self.xy_move_step_counter) + ".csv")

                # Update counter
                self.xy_move_step_counter +=1

                # Apply alteration to the mtm_point itself
                row['pl_point_altered_x'] = p1[0] + (self.alteration_movement * row['movement_x'])
                row['pl_point_altered_y'] = p1[1] + (self.alteration_movement * row['movement_y'])
                
                # Update the mtm_point in selected_df
                selected_df_copy.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_x'] = row['pl_point_altered_x']
                selected_df_copy.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_y'] = row['pl_point_altered_y']

                logging.info(f"Altered MTM point {mtm_point} to new coordinates: "
                            f"({row['pl_point_altered_x']}, {row['pl_point_altered_y']})")

                # Apply alteration to the mtm_dependent point itself
                dependent_row['pl_point_altered_x'] = p2[0] + (self.alteration_movement * row['movement_x'])
                dependent_row['pl_point_altered_y'] = p2[1] + (self.alteration_movement * row['movement_y'])

                # Update the mtm_dependent in selected_df
                selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_dependent, 'pl_point_altered_x'] = dependent_row['pl_point_altered_x']
                selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_dependent, 'pl_point_altered_y'] = dependent_row['pl_point_altered_y']

                logging.info(f"Altered dependent MTM point {mtm_dependent} to new coordinates: "
                            f"({dependent_row['pl_point_altered_x']}, {dependent_row['pl_point_altered_y']})")
                
                return row, selected_df_copy

        except Exception as e:
            logging.error(f"Failed to apply XY adjustment: {e}")
            return row, selected_df
            
    def apply_mtm_correction(self, selected_df):
        """
        Post-processing.
        Applies MTM correction by identifying the rows where 'mtm_dependent' is equal to 'mtm points',
        both values are not NaN, and the 'alteration_type' is "X Y MOVE". Additionally, it finds the
        adjacent MTM points (both left and right), and applies movement adjustments to the points
        that fall between them using the movement values of the current MTM point.
        
        :param selected_df: The DataFrame containing all points, ordered as required.
        :return: A DataFrame filtered where 'mtm_dependent' is equal to 'mtm points', both are not NaN,
                and 'alteration_type' is "X Y MOVE", including adjacent MTM points and intermediate pl_points.
        """
        try:
            # Ensure the relevant columns are numeric
            selected_df['pl_point_x'] = pd.to_numeric(selected_df['pl_point_x'], errors='coerce')
            selected_df['pl_point_y'] = pd.to_numeric(selected_df['pl_point_y'], errors='coerce')
            selected_df['pl_point_altered_x'] = pd.to_numeric(selected_df['pl_point_altered_x'], errors='coerce')
            selected_df['pl_point_altered_y'] = pd.to_numeric(selected_df['pl_point_altered_y'], errors='coerce')
            selected_df['movement_x'] = pd.to_numeric(selected_df['movement_x'], errors='coerce')
            selected_df['movement_y'] = pd.to_numeric(selected_df['movement_y'], errors='coerce')

            # Filter DataFrame where 'mtm_dependent' equals 'mtm points', neither is NaN,
            # and 'alteration_type' is "X Y MOVE"
            filtered_df = selected_df[
                (selected_df['mtm_dependent'] == selected_df['mtm points']) &
                selected_df['mtm_dependent'].notna() &
                selected_df['mtm points'].notna() &
                (selected_df['alteration_type'] == "X Y MOVE")
            ]

            # Sort the DataFrame by 'mtm points' to ensure proper ordering
            #selected_df = selected_df.sort_values(by='mtm points').reset_index(drop=True)

            # Loop through the filtered rows and apply the correction
            for idx, row in filtered_df.iterrows():
                mtm_point = row['mtm points']
                movement_x = row['movement_x']
                movement_y = row['movement_y']

                # Log the movement values for the current MTM point
                logging.info(f"MTM Point: {mtm_point}, Applying Movement X: {movement_x}, Movement Y: {movement_y}")

                # Find the previous MTM point (largest MTM point smaller than the current one)
                previous_point = selected_df[selected_df['mtm points'] < mtm_point].iloc[-1] if not selected_df[selected_df['mtm points'] < mtm_point].empty else None
                
                # Find the next MTM point (smallest MTM point larger than the current one)
                next_candidates = selected_df[selected_df['mtm points'] > mtm_point]
                next_point = next_candidates[next_candidates['mtm points'] == mtm_point + 1].iloc[0] if not next_candidates[next_candidates['mtm points'] == mtm_point + 1].empty else next_candidates.iloc[0] if not next_candidates.empty else None

                # Get the points between previous and next points (excluding the MTM points themselves)
                if previous_point is not None and next_point is not None:
                    points_in_range = selected_df[
                        (selected_df['pl_point_x'] > previous_point['pl_point_x']) &  # Strict inequality to exclude previous_point
                        (selected_df['pl_point_x'] < next_point['pl_point_x'])  # Strict inequality to exclude next_point
                    ]
                else:
                    points_in_range = pd.DataFrame()  # No points if previous or next are None

                # Apply movement to the points in between (excluding previous and next MTM points)
                for point_idx, point in points_in_range.iterrows():
                    current_x = point['pl_point_altered_x'] if pd.notna(point['pl_point_altered_x']) else point['pl_point_x']
                    current_y = point['pl_point_altered_y'] if pd.notna(point['pl_point_altered_y']) else point['pl_point_y']

                    # Apply the movement using the movement_x and movement_y from the current MTM point
                    altered_x = current_x + movement_x
                    altered_y = current_y + movement_y

                    # Update these values directly in selected_df
                    selected_df.loc[point_idx, 'pl_point_altered_x'] = altered_x
                    selected_df.loc[point_idx, 'pl_point_altered_y'] = altered_y

                    # Log the movement applied to each in-between point
                    logging.info(f"Applied to Point {point_idx}: New X: {altered_x}, New Y: {altered_y}")

                # Check if points_in_range is not empty before logging
                if not points_in_range.empty and 'pl_point_x' in points_in_range.columns and 'pl_point_y' in points_in_range.columns:
                    logging.info(f"Points in Range Updated: {points_in_range[['pl_point_x', 'pl_point_y']]}")
                else:
                    logging.info("No points in range for this MTM point.")

            return selected_df

        except Exception as e:
            logging.error(f"Failed to apply MTM correction: {e}")
            return None

    def get_notch_points(self, perimeter_threshold=0.1575, tolerance=0.50, angle_threshold=60.0, small_displacement_threshold=0.1):
        """
        Identifies notch points from the list of vertices.
        A notch is defined as three points forming a perimeter approximately equal to the given threshold
        and with sharp angles. Returns a list of identified notch points without modifying the DataFrame.

        :param perimeter_threshold: The perimeter threshold in inches (default is 0.1575 inches for 4mm).
        :param tolerance: The allowed tolerance when comparing perimeters.
        :param angle_threshold: The angle threshold to detect sharp turns representing notches.
        :param small_displacement_threshold: The minimum threshold for the displacement to avoid retaining points with
                                            very small changes in coordinates.
        :return: List of identified notch points (each entry is a tuple of the three points forming a notch).
        """
        logging.info(f"Starting notch point detection process.")
        notch_points = []  # To store identified notch points
        total_notches_detected = 0

        for index, row in self.vertices_df.iterrows():
            vertex_list = ast.literal_eval(row['vertices'])
            if len(vertex_list) < 3:
                continue

            # Iterate through the vertex list and analyze points for proximity, angles, and dips
            for i in range(1, len(vertex_list) - 1):
                p1, p2, p3 = vertex_list[i - 1], vertex_list[i], vertex_list[i + 1]

                # Calculate distances
                d12 = np.linalg.norm(np.array(p1) - np.array(p2))
                d23 = np.linalg.norm(np.array(p2) - np.array(p3))
                d31 = np.linalg.norm(np.array(p3) - np.array(p1))

                # Calculate the angle between the points
                angle = self.processing_utils.calculate_angle(p1, p2, p3)

                # Calculate the perimeter (sum of the distances)
                perimeter = d12 + d23 + d31

                # Check for sharp angle (notch) and perimeter size
                if abs(angle) < angle_threshold and perimeter < (perimeter_threshold + tolerance):
                    logging.info(f"Notch detected at points: {p1}, {p2}, {p3} with angle: {angle:.2f} degrees")
                    total_notches_detected += 1
                    notch_points.append((p1, p2, p3))  # Store the identified notch points

                # Additional check for small displacement points
                if self.is_small_displacement(p1, p2, p3, small_displacement_threshold):
                    logging.info(f"Small displacement detected at point: {p2}.")
                    notch_points.append((p1, p2, p3))  # Store the small displacement points as notches

        logging.info(f"Total notches detected: {total_notches_detected}")
        return notch_points

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

    def process_and_save_vertices(self):
        """
        Prepares the vertices, reduces them, flattens, and saves the processed vertices.
        """
        logging.info(f"\nProcessing Vertices")

        ##### Processing Functions #####
        # Prepare vertices by removing notch points
        notch_points = self.get_notch_points()

        # Prepare vertices by reducing them (Old Method: Does not maintain the right dimensions)
        # When i remove this, it causes problemos
        #self.vertices_df = self.vertices_df.copy().apply(self.reduce_vertices, axis=1)
        #########

        # Save processed vertices
        self.save_processed_vertices()

        # Process vertices into a flattened list and remove duplicates
        vertices_nested_list = self.extract_and_flatten_vertices(self.vertices_df['vertices'].tolist())
        self.processed_vertices_list = self.processing_utils.remove_duplicates_preserve_order(vertices_nested_list)
        logging.info("Processed Vertices Done")

        return notch_points

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

    #piece_table_path = "data/staging/alteration_by_piece/combined_table_LGFG-V2-BC1-SH-08.csv"
    #vertices_table_path = "data/staging/vertices/vertices_LGFG-V2-BC1-SH-08.csv"
    
    # Debug: Check by Alteration Rule
    #debug_alteration_rule = "7F-SHPOINT"
    #debug_alteration_rule = "7F-ERECT"
    #debug_alteration_rule = "4-WAIST"
    debug_alteration_rule = "1LTH-FULL"
    #debug_alteration_rule = "1LTH-FSLV"
    #debug_alteration_rule = "1LTH-BACK"
    #debug_alteration_rule = "2ARMHOLEDN"
    #debug_alteration_rule = "2ARMHOLEIN"
    #debug_alteration_rule = "4-CHEST"
    #debug_alteration_rule = "3-COLLAR"
    #debug_alteration_rule = "FRT-HEIGHT"
    #debug_alteration_rule = "2SL-BICEP"


    alteration_movement = 2 # INCHES (can be positive or negative)
    
    make_alteration = PieceAlterationProcessor(piece_table_path=piece_table_path,
                                               vertices_table_path=vertices_table_path,
                                               debug_alteration_rule=debug_alteration_rule, 
                                               alteration_movement = alteration_movement)
    make_alteration.process_alterations()
    #make_alteration.log_info(debug_alteration_rule)

