import pandas as pd
import ast
import numpy as np
from app.smoothing import SmoothingFunctions  # Import the SmoothingFunctions class
from utils.data_processing_utils import DataProcessingUtils
import os
from functools import partial
from itertools import combinations, groupby  # Added groupby import
import networkx as nx

pd.set_option('future.no_silent_downcasting', True)

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

        # Adjustment points
        self.cw_adjustment_points = pd.DataFrame()  # Initialize as empty DataFrame
        self.ccw_adjustment_points = pd.DataFrame() 

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

        # Separate rows with an alteration_type and rows without an alteration_type
        rows_with_alteration = combined_df[combined_df['alteration_type'].notna()]
        rows_without_alteration = combined_df[combined_df['alteration_type'].isna()]

        # Drop rows from rows_without_alteration where the coordinates (pl_point_x, pl_point_y) exist in rows_with_alteration
        rows_to_drop = rows_without_alteration[
            rows_without_alteration.set_index(['pl_point_x', 'pl_point_y']).index.isin(
                rows_with_alteration.set_index(['pl_point_x', 'pl_point_y']).index
            )
        ]

        # Drop these rows from the combined DataFrame
        combined_df = combined_df.drop(rows_to_drop.index)

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
        
        return df.drop_duplicates(subset=['alteration_type', 'mtm points', 'pl_point_x', 'pl_point_y'])
    
    def mark_notch_points(self, notch_points, processed_df):
        """
        Marks the identified notch points in the altered DataFrame by creating a new column 
        that labels each notch point with a unique identifier.

        :param notch_points: List of identified notch points (each entry is a tuple of the three points forming a notch).
        :return: DataFrame with a new column 'notch_labels' indicating the identified notch points.
        """
        df = processed_df
        df = df.copy()  # Make an explicit copy to avoid the warning

        # Add a new column for marking notch points if it doesn't exist
        if 'notch_labels' not in df.columns:
            df.loc[:, 'notch_labels'] = ''

        logging.info(f"Starting to mark notch points. Total notch points identified: {len(notch_points)}")

        # Iterate through the identified notch points
        for notch in notch_points:
            p1, p2, p3 = notch

            #logging.info(f"Marking notch: ({p1}, {p2}, {p3})")

            # Find and mark each of the three points in the DataFrame
            for point in [p1, p2, p3]:
                # Locate the rows that match the notch point coordinates
                match_condition = (df['pl_point_x'] == point[0]) & (df['pl_point_y'] == point[1])
                if not df[match_condition].empty:
                    # Check if 'notch' is already in the 'notch_labels' column for the point
                    if 'notch' not in df.loc[match_condition, 'notch_labels'].values[0]:
                        df.loc[match_condition, 'notch_labels'] += 'notch'
                    #    logging.info(f"Marked notch point: {point} in DataFrame.")
                    #else:
                    #    logging.info(f"Duplicate notch point: {point} already marked, skipping.")
                #else:
                #    logging.warning(f"Notch point {point} not found in DataFrame.")

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
        df.loc[:, :] = df.replace("", np.nan)

        # Step 2: Check if each element in a row (excluding 'piece_name') is either NaN or zero
        is_nan_or_zero = df.drop(columns=['piece_name']).map(lambda x: pd.isna(x) or x == 0)

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

            # Mark notch points
            processed_df = self.mark_notch_points(notch_points, processed_df)

            self.processing_utils.save_csv(processed_df, "data/processed_df.csv")

            # Assuming you have your DataFrame loaded as `df`
            alteration_type_counts = processed_df['alteration_type'].value_counts()
            self.xy_move_count = alteration_type_counts.get('X Y MOVE', 0)  # Defaults to 0 if 'X Y MOVE' doesn't exist
            self.cw_ext_count = alteration_type_counts.get('CW Ext', 0)
            self.ccw_ext_count = alteration_type_counts.get('CCW Ext', 0)
            self.ccw_no_ext_count = alteration_type_counts.get('CCW No Ext', 0)

            # Initiate alteration order to keep track of alterations
            processed_df = processed_df.copy()
            processed_df.loc[:, "alteration_order"] = np.nan
            order_count = 0

            # Apply alteration rule row-by-row
            for index, row in processed_df.iterrows():

                # Only apply alterations where types exist
                if not pd.isna(row['alteration_type']):
                    logging.info(f"Processing Alteration Type: {row['alteration_type']}")

                    if row['alteration_type'] == 'CW Ext':
                        # Apply CW Ext (MTM point only)
                        row, updated_df = self.apply_extension(row, processed_df, extension_type="CW")
                        self.cw_ext_count += 1

                    elif row['alteration_type'] == 'CCW Ext':
                        # Apply CCW Ext (MTM point only)
                        row, updated_df = self.apply_extension(row, processed_df, extension_type="CCW")
                        self.ccw_ext_count += 1
                    else:
                        # For other alterations (X Y Move, CCW No Ext), apply normally
                        row, updated_df = self.apply_alteration_to_row(row, processed_df)

                    # Update alteration order in the row
                    row["alteration_order"] = int(order_count)
                
                    # Update the row back into processed_df (in case it's an individual row alteration)
                    row = row.apply(lambda x: np.nan if x == "" else x)
                    processed_df.loc[index] = row
                
                    # Ensure any DataFrame-wide changes are applied to processed_df
                    processed_df.update(updated_df)
                    order_count += 1

            # Check if CW Ext or CCW 

            # Check for further alteration points
            if self.xy_move_step_counter == self.xy_move_step_counter:
                processed_df = self.xy_move_correction(processed_df)
            
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
                self.altered_df = self.re_adjust_points(self.altered_df)
                self.processing_utils.save_csv(self.altered_df, save_filepath)
            else:
                logging.warning(f"Debug alteratio n type '{self.debug_alteration_rule}' not found in requested alterations.")
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
            #'CW Ext': self.apply_cw_ext,
            #'CCW Ext': self.apply_ccw_ext,
            'CCW No Ext': self.apply_ccw_no_ext,
        }

        alteration_type = row['alteration_type']
        logging.info(f"Attempting to apply alteration: {alteration_type}")
        
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
    
    def re_adjust_points(self, selected_df):
        """
        After applying all CW Ext and CCW Ext, adjust all points that were marked for adjustment.
        Points that are in the `cw_adjustment_points` and `ccw_adjustment_points` are updated here.
        
        :param selected_df: The DataFrame with all altered points.
        :return: The updated DataFrame after re-adjustments.
        """
        try:
            # First, apply adjustments to CW Ext adjustment points
            if not self.cw_adjustment_points.empty:
                logging.info("Applying final adjustments to CW Ext adjustment points.")
                self.cw_adjustment_points = self.apply_adjustment_to_points(
                    "CW Ext",  # Pass the alteration type for CW Ext
                    self.cw_adjustment_points,
                    selected_df
                )
                selected_df.update(self.cw_adjustment_points)

            # Then, apply adjustments to CCW Ext adjustment points
            if not self.ccw_adjustment_points.empty:
                logging.info("Applying final adjustments to CCW Ext adjustment points.")
                self.ccw_adjustment_points = self.apply_adjustment_to_points(
                    "CCW Ext",  # Pass the alteration type for CCW Ext
                    self.ccw_adjustment_points,
                    selected_df
                )
                selected_df.update(self.ccw_adjustment_points)

            return selected_df

        except Exception as e:
            logging.error(f"Error during re-adjustments: {e}")
            return selected_df

    def apply_adjustment_to_points(self, 
                                   alteration_type, 
                                   adjustment_points, 
                                   selected_df,
                                   notch_point_proximity_margin = 0.2):
        """
        Apply movement adjustments to the given adjustment points using a non-linear scaling function (e.g., polynomial).
        
        :param alteration_type: The type of alteration being applied (CW Ext or CCW Ext).
        :param adjustment_points: DataFrame of points that need to be adjusted.
        :param selected_df: The DataFrame containing all altered points for comparison.
        """
        # Find the MTM point where the specific alteration (CW Ext or CCW Ext) exists
        mtm_row = adjustment_points[adjustment_points['alteration_type'] == alteration_type].iloc[0]
        mtm_point = mtm_row['mtm points']
        mtm_dependent = mtm_row['mtm_dependent']

        # Get the most recent MTM coordinates
        mtm_coords = (
            mtm_row[['pl_point_altered_x', 'pl_point_altered_y']].fillna(mtm_row[['pl_point_x', 'pl_point_y']]).to_numpy()
        )

        # Find if there is a more recent update for the same MTM point in the selected_df (from either CW or CCW)
        more_recent_update = selected_df[(selected_df['mtm points'] == mtm_point) &
                                        (selected_df['pl_point_altered_x'].notna()) &
                                        (selected_df['pl_point_altered_y'].notna())]

        if not more_recent_update.empty:
            logging.info(f"Found more recent update for MTM point {mtm_point}")
            mtm_coords = more_recent_update[['pl_point_altered_x', 'pl_point_altered_y']].to_numpy()[0]

        # Get the dependent point coordinates (use altered if available, otherwise original)
        mtm_dependent_row = adjustment_points[adjustment_points['mtm points'] == mtm_dependent].iloc[0]
        mtm_dependent_coords = (
            mtm_dependent_row[['pl_point_x', 'pl_point_y']].to_numpy()
        )
        movement_x = mtm_row['movement_x']
        movement_y = mtm_row['movement_y']

        total_distance = np.linalg.norm(mtm_dependent_coords - mtm_coords)

        # Debug prints
        print("MTM Point:", mtm_point)
        print("MTM Coords:", mtm_coords)
        print("MTM Dependent Point:", mtm_dependent)
        print("MTM Dependent Coords:", mtm_dependent_coords)

        # Ensure the MTM point is not updated again during re-adjustment
        adjustment_points = adjustment_points[adjustment_points['mtm points'] != mtm_point]

        for idx, point in adjustment_points.iterrows():
            # Identify if this point is a notch point
            is_notch = 'notch' in str(point.get('notch_labels', ''))

            # Use altered coordinates if available, otherwise use original ones
            current_x = point['pl_point_altered_x'] if pd.notna(point['pl_point_altered_x']) else point['pl_point_x']
            current_y = point['pl_point_altered_y'] if pd.notna(point['pl_point_altered_y']) else point['pl_point_y']

            # Calculate the distance of the current point from the MTM point
            point_coords = np.array([current_x, current_y], dtype=np.float64)

            # Dummy: Calculate temporary "new" point coords based on the altered distance 

            distance_to_mtm = np.linalg.norm(point_coords - mtm_coords)
            distance_to_mtm_original = np.linalg.norm(point_coords - mtm_row[['pl_point_x', 'pl_point_y']].to_numpy())

            # Handle notch points that are near the MTM point (within the proximity_margin)
            if is_notch and distance_to_mtm_original <= notch_point_proximity_margin:
                logging.info(f"Notch point found near MTM point {mtm_point}. Moving it directly by the same movement.")

                # Does this handle both positive and negative?
                diff_coords = np.abs(point_coords - mtm_row[['pl_point_x', 'pl_point_y']])

                # Convert to NumPy arrays explicitly to avoid any FutureWarning
                mtm_coords = np.array(mtm_coords)
                diff_coords = np.array(diff_coords)
                
                # Move the notch point by the same movement as the MTM point
                altered_x = mtm_coords[0] - diff_coords[0]
                altered_y = mtm_coords[1] - diff_coords[1]

            else:
                # Example for cubic scaling
                # adjustment_factor = 1 - (distance_to_mtm / total_distance) ** 3
                
                # Apply exponential decay or other scaling function
                lambda_param = 1.0  # Adjust this for different decay rates
                adjustment_factor = np.exp(-lambda_param * (distance_to_mtm / total_distance))
                
                # Ensure the dependent point stays fixed
                if np.isclose(distance_to_mtm, total_distance):
                    adjustment_factor = 0  # Fix the last point (dependent point)
                
                # Example for logarithmic scaling
                #adjustment_factor = 1 - np.log(distance_to_mtm + 1) / np.log(total_distance + 1)
                
                # Example for sine scaling
                # adjustment_factor = np.sin(np.pi / 2 * (distance_to_mtm / total_distance))

                # Example for inverse proportional scaling
                # k = 1  # You can adjust this value to control the slope
                # adjustment_factor = 1 / (1 + k * (distance_to_mtm / total_distance))
                
                # Example for sigmoid scaling
                # a, b = 10, 0.5  # Tuning parameters
                # adjustment_factor = 1 / (1 + np.exp(a * (distance_to_mtm / total_distance - b)))

                if movement_x != 0 and movement_y == 0:
                    altered_x = current_x + adjustment_factor
                    altered_y = current_y 

                elif movement_y != 0 and movement_x == 0:
                    altered_x = current_x 
                    altered_y = current_y + adjustment_factor
                else:
                    altered_x = current_x  + adjustment_factor
                    altered_y = current_y + adjustment_factor

            # Update the altered values in the DataFrame
            adjustment_points.loc[idx, 'pl_point_altered_x'] = altered_x
            adjustment_points.loc[idx, 'pl_point_altered_y'] = altered_y

        # Save debug information for adjustment points
        adjustment_points.to_csv(f"data/adjustment_points_{alteration_type}.csv")

        return adjustment_points

    def apply_ccw_no_ext(self, row, selected_df, tolerance = 0):
        # Example logic for Counter-clockwise No Extension
        return row, selected_df
    
    def apply_extension(self, row, selected_df, extension_type="CW", tolerance=0):
        """
        Apply CW or CCW Ext alteration to the MTM point itself, and record surrounding points for adjustment later.
        """
        try:
            mtm_point = row['mtm points']
            mtm_dependent = row['mtm_dependent']

            selected_df_copy = selected_df.copy()
            p1, p2 = self._get_point_coordinates(mtm_point, mtm_dependent, selected_df_copy)
            movement_x, movement_y = row['movement_x'], row['movement_y']

            # Ensure any empty strings are converted to NaN
            selected_df_copy = selected_df_copy.replace("", np.nan).infer_objects()

            # Calculate point orders
            start_point_order = self._get_point_order(mtm_point, selected_df_copy)
            end_point_order = self._get_point_order(mtm_dependent, selected_df_copy)

            # Add distance calculations to DataFrame for proximity
            selected_df_copy = self._add_distance_to_points(selected_df_copy, p1, p2)

            # Get the altered or original coordinates of the mtm_point and mtm_dependent
            mtm_point_coords = selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_point,
                                                    ['pl_point_altered_x', 'pl_point_altered_y', 'pl_point_x', 'pl_point_y']].iloc[0]

            mtm_dependent_coords = selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_dependent,
                                                        ['pl_point_altered_x', 'pl_point_altered_y', 'pl_point_x', 'pl_point_y']].iloc[0]

            # Use altered coordinates if available, else fallback to original ones
            mtm_point_coords = [mtm_point_coords['pl_point_altered_x'], mtm_point_coords['pl_point_altered_y']] \
                if pd.notna(mtm_point_coords['pl_point_altered_x']) and pd.notna(mtm_point_coords['pl_point_altered_y']) \
                else [mtm_point_coords['pl_point_x'], mtm_point_coords['pl_point_y']]

            mtm_dependent_coords = [mtm_dependent_coords['pl_point_altered_x'], mtm_dependent_coords['pl_point_altered_y']] \
                if pd.notna(mtm_dependent_coords['pl_point_altered_x']) and pd.notna(mtm_dependent_coords['pl_point_altered_y']) \
                else [mtm_dependent_coords['pl_point_x'], mtm_dependent_coords['pl_point_y']]
            
            # Apply movement to MTM point
            new_x = mtm_point_coords[0] + (self.alteration_movement * movement_x)
            new_y = mtm_point_coords[1] + (self.alteration_movement * movement_y)

            # Update the MTM point in the DataFrame
            selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_point, 'pl_point_altered_x'] = new_x
            selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_point, 'pl_point_altered_y'] = new_y

            # Log the movement applied
            logging.info(f"{extension_type} Ext applied to MTM point {mtm_point}: New coordinates ({new_x}, {new_y})")

            # Total distance between the mtm point and the mtm dependent
            mtm_point_coords = np.array(mtm_point_coords, dtype=np.float64)
            mtm_dependent_coords = np.array(mtm_dependent_coords, dtype=np.float64)
            total_distance = np.linalg.norm(mtm_dependent_coords - mtm_point_coords)

            # Check if its okay that alteration point is updated twice?

            # Save surrounding points for adjustment after both CW and CCW are applied
            if extension_type == "CW":
                logging.info(f"CW Ext (movement_x, movement_y): {(movement_x, movement_y)}")
                self.cw_adjustment_points = self._get_points_in_range(selected_df_copy, start_point_order, end_point_order, tolerance, mtm_point, mtm_dependent)

            elif extension_type == "CCW":
                logging.info(f"CCW Ext (movement_x, movement_y): {(movement_x, movement_y)}")
                self.ccw_adjustment_points = self._get_points_in_range(selected_df_copy, start_point_order, end_point_order, tolerance, mtm_point, mtm_dependent) 

            return row, selected_df_copy

        except Exception as e:
            logging.error(f"Failed to apply {extension_type} Ext alteration: {e}")
            return row, selected_df_copy

    def apply_xy_coordinate_adjustment(self, row, selected_df, tolerance = 0):
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
                self.xy_move_step_counter +=1
                return self._apply_single_point_move(row, selected_df, mtm_point)

            # Case 2: Multiple row alteration (mtm_dependent != mtm_point)
            else:
                selected_df.to_csv("data/selected_df_xy_move.csv")     
                selected_df_copy = selected_df.copy()
                p1, p2 = self._get_point_coordinates(mtm_point, mtm_dependent, selected_df_copy)
                movement_x, movement_y = row['movement_x'], row['movement_y']
                dependent_row = selected_df_copy[selected_df_copy['mtm points'] == mtm_dependent].iloc[0]

                # Calculate point orders
                start_point_order = self._get_point_order(mtm_point, selected_df_copy)
                end_point_order = self._get_point_order(mtm_dependent, selected_df_copy)

                # Add distance calculations to DataFrame for proximity
                selected_df_copy = self._add_distance_to_points(selected_df_copy, p1, p2)

                # Capture points within range
                points_in_range = self._get_points_in_range(selected_df_copy, start_point_order, end_point_order, tolerance, mtm_point, mtm_dependent)           
                
                logging.info(f"XY Move (movement_x, movement_y): {(movement_x, movement_y)}")
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

                # Update counter
                self.xy_move_step_counter +=1

                return row, selected_df_copy

        except Exception as e:
            logging.error(f"Failed to apply XY adjustment: {e}")
            return row, selected_df
        
    def _apply_single_point_move(self, row, selected_df, mtm_point):
        # Use altered coordinates if available, otherwise use original ones
        current_x = row['pl_point_altered_x'] if pd.notna(row['pl_point_altered_x']) and row['pl_point_altered_x'] != "" else row['pl_point_x']
        current_y = row['pl_point_altered_y'] if pd.notna(row['pl_point_altered_y']) and row['pl_point_altered_y'] != "" else row['pl_point_y']

        # Apply the movement based on the current altered coordinates
        row['pl_point_altered_x'] = current_x + (self.alteration_movement * row['movement_x'])
        row['pl_point_altered_y'] = current_y + (self.alteration_movement * row['movement_y'])

        logging.info(f"XY Move (movement_x, movement_y): {(row['movement_x'], row['movement_y'])}")

        # Update the altered values in the DataFrame
        selected_df.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_x'] = row['pl_point_altered_x']
        selected_df.loc[selected_df['mtm points'] == mtm_point, 'pl_point_altered_y'] = row['pl_point_altered_y']

        logging.info(f"XY Move: Altered MTM point {mtm_point} to new coordinates: "
                    f"({row['pl_point_altered_x']}, {row['pl_point_altered_y']})")

        return row, selected_df
    
    def _get_point_coordinates(self, mtm_point, mtm_dependent, df):
        """Returns the coordinates for mtm_point and mtm_dependent."""
        p1 = df.loc[df['mtm points'] == mtm_point, ['pl_point_x', 'pl_point_y']].values[0]
        p2 = df.loc[df['mtm points'] == mtm_dependent, ['pl_point_x', 'pl_point_y']].values[0]
        return p1, p2
    
    def _get_point_order(self, point, df):
        """Returns the point_order for a given MTM point."""
        return df.loc[df['mtm points'] == point, 'point_order'].values[0]
    
    def _add_distance_to_points(self, df, p1, p2):
        """Adds the Euclidean distance from p1 and p2 to each point in the DataFrame."""
        df['dist_to_p1'] = np.sqrt((df['pl_point_x'] - p1[0]) ** 2 + (df['pl_point_y'] - p1[1]) ** 2)
        df['dist_to_p2'] = np.sqrt((df['pl_point_x'] - p2[0]) ** 2 + (df['pl_point_y'] - p2[1]) ** 2)
        return df
    
    def _get_points_in_range(self, df, start_order, end_order, tolerance, mtm_point, mtm_dependent):
        """Returns the points between the start and end orders, considering proximity tolerance."""
        if start_order > end_order:
            # Descending Order
            points_in_range = df[
                (
                    (df['point_order'] <= start_order) & (df['point_order'] >= end_order)
                ) | (
                    (df['dist_to_p1'] < tolerance) | (df['dist_to_p2'] < tolerance)
                ) & (df['mtm points'] != mtm_point) & (df['mtm points'] != mtm_dependent)
            ]
        else:
            # Ascending Order
            points_in_range = df[
                (
                    (df['point_order'] >= start_order) & (df['point_order'] <= end_order)
                ) | (
                    (df['dist_to_p1'] < tolerance) | (df['dist_to_p2'] < tolerance)
                ) & (df['mtm points'] != mtm_point) & (df['mtm points'] != mtm_dependent)
            ]
        
        # Ensure that point_order is numeric
        points_in_range.loc[:, 'point_order'] = pd.to_numeric(points_in_range['point_order'], errors='coerce')
        return points_in_range.sort_values(by="point_order")

    def xy_move_correction(self, selected_df):
        try:
            selected_df_copy = selected_df.copy()

            # Ensure the relevant columns are numeric
            numeric_columns = ['pl_point_x', 'pl_point_y', 'pl_point_altered_x', 'pl_point_altered_y', 'movement_x', 'movement_y']
            for col in numeric_columns:
                selected_df_copy[col] = pd.to_numeric(selected_df_copy[col], errors='coerce')

            # Filter DataFrame for the X Y MOVE alteration
            filtered_df = selected_df_copy[
                (selected_df_copy['mtm_dependent'] == selected_df_copy['mtm points']) &
                selected_df_copy['mtm_dependent'].notna() &
                selected_df_copy['mtm points'].notna() &
                (selected_df_copy['alteration_type'] == "X Y MOVE")
            ]

            xy_move_count = filtered_df['alteration_type'].value_counts().get('X Y MOVE', 0)

            if xy_move_count == 1:
                logging.info(f"1 instance of 'X Y MOVE' found. Proceeding with processing.")

                mtm_point = filtered_df['mtm points'].iloc[0]
                mtm_coords = filtered_df[['pl_point_altered_x', 'pl_point_altered_y']].fillna(filtered_df[['pl_point_x', 'pl_point_y']]).iloc[0].to_numpy()

                # Find the previous and next MTM points
                previous_mtm = selected_df_copy[selected_df_copy['mtm points'].notna() & (selected_df_copy['mtm points'] < mtm_point)]['mtm points'].max()
                next_mtm = selected_df_copy[selected_df_copy['mtm points'].notna() & (selected_df_copy['mtm points'] > mtm_point)]['mtm points'].min()

                if pd.isna(previous_mtm) or pd.isna(next_mtm):
                    logging.warning("Either the previous or the next MTM point is missing.")
                    return selected_df_copy

                logging.info(f"Current MTM: {mtm_point}")
                logging.info(f"Previous MTM: {previous_mtm}")
                logging.info(f"Next MTM: {next_mtm}")

                # Get all points between previous and next MTM points
                points_in_range = selected_df_copy[
                    (selected_df_copy['point_order'] > selected_df_copy[selected_df_copy['mtm points'] == previous_mtm]['point_order'].iloc[0]) &
                    (selected_df_copy['point_order'] < selected_df_copy[selected_df_copy['mtm points'] == next_mtm]['point_order'].iloc[0])
                ]

                # Split points into before and after the MTM point
                mtm_index = points_in_range[points_in_range['mtm points'] == mtm_point].index[0]
                points_before = points_in_range.loc[:mtm_index]
                points_after = points_in_range.loc[mtm_index:]

                # Create graphs and apply force-directed algorithm for both sections
                G_before = self._create_point_graph(points_before, mtm_point, mtm_coords)
                G_after = self._create_point_graph(points_after, mtm_point, mtm_coords)

                print("Applying force-directed algorithm to points before MTM")
                new_positions_before = self._apply_force_directed_algorithm(G_before, mtm_point, mtm_coords, iterations=50, k=0.1)
                
                print("Applying force-directed algorithm to points after MTM")
                new_positions_after = self._apply_force_directed_algorithm(G_after, mtm_point, mtm_coords, iterations=50, k=0.1)

                # Combine new positions
                new_positions = {**new_positions_before, **new_positions_after}

                print("New positions applied")
                print(new_positions)

                if new_positions is None:
                    logging.warning("Force-directed algorithm failed to produce new positions. Skipping position updates.")
                else:
                    # Update positions in the DataFrame
                    for node, pos in new_positions.items():
                        if node != mtm_point:  # Don't update the MTM point
                            old_pos = (selected_df_copy.loc[selected_df_copy['point_order'] == node, 'pl_point_x'].iloc[0],
                                       selected_df_copy.loc[selected_df_copy['point_order'] == node, 'pl_point_y'].iloc[0])
                            selected_df_copy.loc[selected_df_copy['point_order'] == node, 'pl_point_altered_x'] = pos[0]
                            selected_df_copy.loc[selected_df_copy['point_order'] == node, 'pl_point_altered_y'] = pos[1]
                            if G_before.nodes.get(node, {}).get('is_notch', False) or G_after.nodes.get(node, {}).get('is_notch', False):
                                logging.info(f"Notch point {node} final movement: {old_pos} -> {pos}")

                # Save for debugging
                selected_df_copy.to_csv("data/selected_df_force_directed.csv")

                return selected_df_copy

            elif xy_move_count > 1:
                logging.warning(f"More than 1 instance of 'X Y MOVE' found: {xy_move_count} instances.")
                # Handle multiple X Y MOVE instances if needed

        except Exception as e:
            logging.error(f"Failed to apply XY Move correction: {e}")
            return selected_df

    def _create_point_graph(self, df, mtm_point, mtm_coords):
        G = nx.Graph()
        for _, row in df.iterrows():
            if row['mtm points'] == mtm_point:
                # Use altered coordinates for the altered MTM point
                pos = tuple(mtm_coords)
            else:
                # Use original coordinates for all other points
                pos = (row['pl_point_x'], row['pl_point_y'])
            
            G.add_node(row['point_order'], pos=pos, is_notch=pd.notna(row.get('notch_labels', pd.NA)), is_mtm=pd.notna(row.get('mtm points', pd.NA)))
        
        # Connect adjacent points
        sorted_points = sorted(G.nodes())
        for i in range(len(sorted_points) - 1):
            G.add_edge(sorted_points[i], sorted_points[i+1])
        
        return G

    def _apply_force_directed_algorithm(self, G, fixed_point, fixed_pos, iterations=50, k=0.1):
        try:
            # Set initial positions for all nodes
            pos = nx.get_node_attributes(G, 'pos')
            
            # Log initial notch positions
            self._log_notch_positions(G, pos, "Initial notch positions")
            
            # Identify all MTM points and the altered point
            fixed_nodes = [node for node, data in G.nodes(data=True) if data['is_mtm'] or node == fixed_point]
            
            # Group notch points
            notch_groups = self._group_notch_points(G)
            
            # Custom spring layout function
            def custom_spring_layout(G, k=0.1, iterations=50):
                pos_array = np.array([pos[node] for node in G.nodes()])
                for _ in range(iterations):
                    for i, node in enumerate(G.nodes()):
                        if node not in fixed_nodes:
                            # Calculate force
                            force = np.zeros(2)
                            for neighbor in G.neighbors(node):
                                diff = pos_array[list(G.nodes()).index(neighbor)] - pos_array[i]
                                distance = np.linalg.norm(diff)
                                if distance > 0:
                                    force += 0.1 * diff / distance  # Attractive force
                            
                            # Apply force with damping
                            pos_array[i] += force * k
                            
                            # Limit displacement
                            original_pos = np.array(pos[node])
                            max_displacement = 0.1  # Limit maximum displacement
                            displacement = pos_array[i] - original_pos
                            if np.linalg.norm(displacement) > max_displacement:
                                pos_array[i] = original_pos + displacement / np.linalg.norm(displacement) * max_displacement
                    
                    # Move notch groups as rigid units
                    self._move_notch_groups(notch_groups, pos_array, G)
                
                return {node: tuple(pos) for node, pos in zip(G.nodes(), pos_array)}

            # Apply custom spring layout
            new_pos = custom_spring_layout(G, k=k, iterations=iterations)
            
            # Ensure fixed nodes maintain their original positions
            for node in fixed_nodes:
                new_pos[node] = pos[node]

            # Log final notch positions
            self._log_notch_positions(G, new_pos, "Final notch positions")

            return new_pos

        except Exception as e:
            logging.error(f"Error in force-directed algorithm: {str(e)}")
            return None

    def _group_notch_points(self, G):
        notch_groups = []
        notch_nodes = [node for node, data in G.nodes(data=True) if data['is_notch']]
        
        for k, g in groupby(enumerate(notch_nodes), lambda ix: ix[0] - ix[1]):
            group = list(map(lambda x: x[1], g))
            if len(group) >= 3:  # Assuming a notch consists of at least 3 points
                notch_groups.append(group)
        
        return notch_groups

    def _move_notch_groups(self, notch_groups, pos_array, G):
        for group in notch_groups:
            # Calculate the average movement for the notch group
            avg_movement = np.zeros(2)
            for node in group:
                node_index = list(G.nodes()).index(node)
                original_pos = np.array(G.nodes[node]['pos'])
                current_pos = pos_array[node_index]
                avg_movement += current_pos - original_pos
            avg_movement /= len(group)
            
            logging.info(f"Moving notch group {group} by {avg_movement}")
            
            # Apply the average movement to all points in the notch group
            for node in group:
                node_index = list(G.nodes()).index(node)
                original_pos = np.array(G.nodes[node]['pos'])
                pos_array[node_index] = original_pos + avg_movement
                logging.info(f"Notch point {node} moved from {original_pos} to {pos_array[node_index]}")

    def _log_notch_positions(self, G, positions, message):
        notch_nodes = [node for node, data in G.nodes(data=True) if data['is_notch']]
        logging.info(f"{message}:")
        for node in notch_nodes:
            logging.info(f"Notch point {node}: {positions[node]}")

    def _adjust_notch_points(self, notch_points, df, new_positions):
        notch_groups = self._group_notch_points(notch_points)
        for group in notch_groups:
            self._move_notch_group(group, df, new_positions)

    def _move_notch_group(self, group, df, new_positions):
        # Calculate the average movement of nearby non-notch points
        nearby_points = self._find_nearby_points(group, df, new_positions)
        avg_movement = self._calculate_average_movement(nearby_points, new_positions)
        
        # Move the entire notch group by this average movement
        for _, point in group.iterrows():
            new_x = point['pl_point_x'] + avg_movement[0]
            new_y = point['pl_point_y'] + avg_movement[1]
            df.loc[df['point_order'] == point['point_order'], 'pl_point_altered_x'] = new_x
            df.loc[df['point_order'] == point['point_order'], 'pl_point_altered_y'] = new_y

    def _find_nearby_points(self, group, df, new_positions):
        group_center = group[['pl_point_x', 'pl_point_y']].mean()
        distances = df.apply(lambda row: np.linalg.norm(row[['pl_point_x', 'pl_point_y']] - group_center), axis=1)
        nearby_indices = distances.nsmallest(5).index
        return df.loc[nearby_indices]

    def _calculate_average_movement(self, nearby_points, new_positions):
        total_movement = [0, 0]
        count = 0
        for _, point in nearby_points.iterrows():
            if point['point_order'] in new_positions:
                old_pos = np.array([point['pl_point_x'], point['pl_point_y']])
                new_pos = np.array(new_positions[point['point_order']])
                movement = new_pos - old_pos
                total_movement += movement
                count += 1
        return total_movement / count if count > 0 else [0, 0]

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
                    #logging.info(f"Notch detected at points: {p1}, {p2}, {p3} with angle: {angle:.2f} degrees")
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


    alteration_movement = 5 # INCHES (can be positive or negative)
    
    make_alteration = PieceAlterationProcessor(piece_table_path=piece_table_path,
                                               vertices_table_path=vertices_table_path,
                                               debug_alteration_rule=debug_alteration_rule, 
                                               alteration_movement = alteration_movement)
    make_alteration.process_alterations()
    #make_alteration.log_info(debug_alteration_rule)

