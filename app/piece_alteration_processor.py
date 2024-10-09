import pandas as pd
import ast
import numpy as np
from utils.data_processing_utils import DataProcessingUtils
import os
from itertools import groupby 
from functools import partial
import networkx as nx

pd.set_option('future.no_silent_downcasting', True)

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

# TODO: Go through the alterations we have done currently
# TODO: Start with 1LTH-FULL

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
        Applies alteration rules to each DataFrame and processes vertices.
        Handles both debug mode and full alteration process.
        """
        # Prepare vertices and get notch points
        notch_points = self.process_and_save_vertices()

        def apply_single_alteration_rule(alteration_rule):
            selected_df = self.alteration_dfs[alteration_rule]
            unselected_df = self.get_unselected_rows(selected_df)

            # Combine selected and unselected rows into a single DataFrame
            combined_df = self.combine_and_clean_data(selected_df, unselected_df)
            
            # Prepare DataFrame by converting columns and dropping unnecessary ones
            processed_df = self.prepare_dataframe(combined_df)

            # Mark notch points
            processed_df = self.mark_notch_points(notch_points, processed_df)

            processed_df.to_csv("data/processed_marked_notch_points.csv", index=False)

            # Count alteration types
            alteration_type_counts = processed_df['alteration_type'].value_counts()
            self.xy_move_count = alteration_type_counts.get('X Y MOVE', 0)
            self.cw_ext_count = alteration_type_counts.get('CW Ext', 0)
            self.ccw_ext_count = alteration_type_counts.get('CCW Ext', 0)
            self.ccw_no_ext_count = alteration_type_counts.get('CCW No Ext', 0)

            # Initialize alteration order
            processed_df['alteration_order'] = np.nan
            order_count = 0

            # Define alteration type map
            alteration_type_map = {
                'X Y MOVE': self.apply_xy_coordinate_adjustment,
                'CW Ext': partial(self.apply_extension, extension_type="CW"),
                'CCW Ext': partial(self.apply_extension, extension_type="CCW"),
                'CCW No Ext': partial(self.apply_no_extension, extension_type="CCW"),
            }

            # Apply alteration rule row-by-row
            for index, row in processed_df.iterrows():
                if not pd.isna(row['alteration_type']):
                    alteration_type = row['alteration_type']
                    logging.info(f"Processing Alteration Type: {alteration_type}")

                    func = alteration_type_map.get(alteration_type)
                    if func:
                        _, processed_df = func(row, processed_df)  # Use the returned DataFrame
                        processed_df.loc[index, 'alteration_order'] = order_count
                        order_count += 1
                    else:
                        logging.warning(f"No viable alteration types found for {alteration_type}")

            # Update the row in processed_df
            processed_df = processed_df.map(lambda x: np.nan if x == "" else x)
            processed_df.to_csv("data/processed_df.csv", index=False)

            # Check for further alteration points
            if self.xy_move_step_counter > 0 and self.xy_move_step_counter == self.xy_move_count:
                #processed_df = self.xy_move_correction(processed_df)
                processed_df = self.post_alteration_point_shift(processed_df)
            
            processed_df = self.remove_empty_rows(processed_df)
            processed_df = self.re_adjust_points(processed_df)

            return processed_df

        # Debug Mode
        if not pd.isna(self.debug_alteration_rule):
            debug_savefolder = "data/staging_processed/debug/"
            save_filepath = f"{debug_savefolder}{self.piece_name}_{self.debug_alteration_rule}{self.save_file_format}"
            os.makedirs(debug_savefolder, exist_ok=True)

            if self.debug_alteration_rule in self.alteration_rules:
                logging.info(f"Running Alteration in Debug Mode: {self.debug_alteration_rule}")
                self.altered_df = apply_single_alteration_rule(self.debug_alteration_rule)
                self.processing_utils.save_csv(self.altered_df, save_filepath)
            else:
                logging.warning(f"Debug alteration type '{self.debug_alteration_rule}' not found in requested alterations.")
            return  # Early return after processing the debug alteration
        
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
        try:
            logging.info(f"CW adjustment points: {len(self.cw_adjustment_points)}")
            logging.info(f"CCW adjustment points: {len(self.ccw_adjustment_points)}")

            if not self.cw_adjustment_points.empty and not self.ccw_adjustment_points.empty:
                # Both CW and CCW extensions occurred
                logging.info("Applying final adjustments to CW and CCW Ext adjustment points separately.")
                
                # Find the common MTM point
                cw_mtm = self.cw_adjustment_points[self.cw_adjustment_points['alteration_type'] == 'CW Ext'].iloc[0]
                ccw_mtm = self.ccw_adjustment_points[self.ccw_adjustment_points['alteration_type'] == 'CCW Ext'].iloc[0]
                
                if cw_mtm['mtm points'] != ccw_mtm['mtm points']:
                    logging.warning("CW and CCW MTM points do not match. Using CW MTM point.")
                
                mtm_point = cw_mtm
                
                # Calculate total movement
                total_movement = (cw_mtm['movement_y'] + ccw_mtm['movement_y']) / 2
                
                # Adjust notch points first
                self.cw_adjustment_points = self.adjust_notch_points(self.cw_adjustment_points, mtm_point, total_movement)
                self.ccw_adjustment_points = self.adjust_notch_points(self.ccw_adjustment_points, mtm_point, total_movement)
                
                # Adjust points in range
                self.cw_adjustment_points = self.apply_adjustment_to_points_in_range(
                    alteration_type="CW Ext",
                    adjustment_points=self.cw_adjustment_points,
                    selected_df=selected_df,
                    total_movement=total_movement,
                    mtm_point=mtm_point
                )
                selected_df.update(self.cw_adjustment_points)
                
                self.ccw_adjustment_points = self.apply_adjustment_to_points_in_range(
                    alteration_type="CCW Ext",
                    adjustment_points=self.ccw_adjustment_points,
                    selected_df=selected_df,
                    total_movement=total_movement,
                    mtm_point=mtm_point
                )
                selected_df.update(self.ccw_adjustment_points)
            
            else:
                # If only one type of extension occurred, proceed as before
                if not self.cw_adjustment_points.empty:
                    logging.info("Applying final adjustments to CW Ext adjustment points.")
                    self.cw_adjustment_points = self.adjust_notch_points(self.cw_adjustment_points, None, None)
                    self.cw_adjustment_points = self.apply_adjustment_to_points_in_range(
                        alteration_type="CW Ext",
                        adjustment_points=self.cw_adjustment_points,
                        selected_df=selected_df
                    )
                    selected_df.update(self.cw_adjustment_points)

                if not self.ccw_adjustment_points.empty:
                    logging.info("Applying final adjustments to CCW Ext adjustment points.")
                    self.ccw_adjustment_points = self.adjust_notch_points(self.ccw_adjustment_points, None, None)
                    self.ccw_adjustment_points = self.apply_adjustment_to_points_in_range(
                        alteration_type="CCW Ext",
                        adjustment_points=self.ccw_adjustment_points,
                        selected_df=selected_df
                    )
                    selected_df.update(self.ccw_adjustment_points)

            return selected_df

        except Exception as e:
            logging.error(f"Error during re-adjustments: {e}")
            logging.error(f"CW adjustment points: {self.cw_adjustment_points}")
            logging.error(f"CCW adjustment points: {self.ccw_adjustment_points}")
            return selected_df

    def adjust_notch_points(self, adjustment_points, mtm_point=None, total_movement=None, notch_point_proximity_margin=0.2):
        if mtm_point is None or total_movement is None:
            mtm_point = adjustment_points[adjustment_points['alteration_type'].isin(['CW Ext', 'CCW Ext'])].iloc[0]
            total_movement = mtm_point['movement_y']

        mtm_coords_original = np.array([mtm_point['pl_point_x'], mtm_point['pl_point_y']])
        mtm_coords_altered = np.array([mtm_point['pl_point_altered_x'], mtm_point['pl_point_altered_y']])

        def distance_to_mtm(point):
            return np.linalg.norm(point - mtm_coords_original)

        for idx, point in adjustment_points.iterrows():
            current_coords = point[['pl_point_x', 'pl_point_y']].to_numpy()
            is_notch = 'notch' in str(point.get('notch_labels', ''))
            distance_to_mtm_original = distance_to_mtm(current_coords)

            if is_notch and distance_to_mtm_original <= notch_point_proximity_margin:
                logging.info(f"Notch point found near MTM point {mtm_point['mtm points']}. Moving it directly by the same movement.")

                diff_coords = np.abs(current_coords - mtm_coords_original)
                altered_coords = mtm_coords_altered - diff_coords

                adjustment_points.loc[idx, 'pl_point_altered_x'] = altered_coords[0]
                adjustment_points.loc[idx, 'pl_point_altered_y'] = altered_coords[1]

        return adjustment_points

    def apply_adjustment_to_points_in_range(self, 
                                            alteration_type, 
                                            adjustment_points, 
                                            selected_df,
                                            total_movement=None,
                                            mtm_point=None):
        logging.info(f"Applying adjustments for {alteration_type}")
        logging.info(f"Number of adjustment points: {len(adjustment_points)}")
        logging.info(f"Total movement: {total_movement}")

        if mtm_point is None or total_movement is None:
            mtm_point = adjustment_points[adjustment_points['alteration_type'].isin(['CW Ext', 'CCW Ext'])].iloc[0]
            total_movement = mtm_point['movement_y']

        mtm_coords_original = np.array([mtm_point['pl_point_x'], mtm_point['pl_point_y']])
        mtm_coords_altered = np.array([mtm_point['pl_point_altered_x'], mtm_point['pl_point_altered_y']])

        logging.info(f"MTM point original coordinates: {mtm_coords_original}")
        logging.info(f"MTM point altered coordinates: {mtm_coords_altered}")

        def distance_to_mtm(point):
            return np.linalg.norm(point - mtm_coords_original)

        max_distance = max(distance_to_mtm(point[['pl_point_x', 'pl_point_y']].to_numpy()) for _, point in adjustment_points.iterrows())
        logging.info(f"Maximum distance from MTM point: {max_distance}")

        def curved_adjustment(relative_distance, curvature=0.5):
            return 1 - (relative_distance ** curvature)

        def gaussian_smoothing(relative_distance, sigma=0.3):
            return np.exp(-0.5 * (relative_distance / sigma)**2)

        def cosine_smoothing(relative_distance):
            return 0.5 * (1 + np.cos(np.pi * relative_distance))
        
        def linear_smoothing(relative_distance):
            return 1 - relative_distance

        def smoothstep(relative_distance):
            x = np.clip(relative_distance, 0, 1)
            return 1 - (x * x * (3 - 2 * x))  # Reversed smoothstep
        
        def smoothing_function(relative_distance):
            #return gaussian_smoothing(relative_distance, sigma=0.3)
            #return cosine_smoothing(relative_distance)
            #return smoothstep(relative_distance)
            return linear_smoothing(relative_distance)

        for idx, point in adjustment_points.iterrows():
            logging.info(f"Processing point {idx}: {point.get('mtm points', 'nan')}")
            current_coords = point[['pl_point_x', 'pl_point_y']].to_numpy()
            logging.info(f"  Original coordinates: {current_coords}")

            is_mtm_dependent = not pd.isna(point.get('mtm points'))
            is_notch = 'notch' in str(point.get('notch_labels', ''))

            #if is_mtm_dependent:
            #    logging.info(f"  MTM-dependent point: Keeping original position.")
            #     continue
            if is_notch:
                logging.info(f"  Notch point: Skipping adjustment.")
                continue

            distance_to_mtm_original = distance_to_mtm(current_coords)
            relative_distance = distance_to_mtm_original / max_distance
            adjustment_factor = smoothing_function(relative_distance)

            logging.info(f"  Distance to MTM: {distance_to_mtm_original}")
            logging.info(f"  Relative distance: {relative_distance}")
            logging.info(f"  Adjustment factor: {adjustment_factor}")

            vector_to_point = current_coords - mtm_coords_original
            new_base_position = mtm_coords_altered + vector_to_point

            point_movement = (new_base_position - current_coords) * adjustment_factor
            altered_coords = current_coords + point_movement

            logging.info(f"  Vector to point: {vector_to_point}")
            logging.info(f"  New base position: {new_base_position}")
            logging.info(f"  Point movement: {point_movement}")
            logging.info(f"  Altered coordinates: {altered_coords}")

            adjustment_points.loc[idx, 'pl_point_altered_x'] = altered_coords[0]
            adjustment_points.loc[idx, 'pl_point_altered_y'] = altered_coords[1]

        return adjustment_points
    
    def check_alteration_limits(self, alteration_movement, max_pos, max_neg, min_pos, min_neg, mtm_point):
        """
        Check if the alteration movement is within the suggested limits.

        :param alteration_movement: The proposed alteration movement.
        :param max_pos: Maximum positive movement limit.
        :param max_neg: Maximum negative movement limit.
        :param min_pos: Minimum positive movement limit.
        :param min_neg: Minimum negative movement limit.
        :param mtm_point: The MTM point being altered.
        :raises ValueError: If the alteration movement exceeds the suggested limits.
        """
        if alteration_movement > 0:
            if alteration_movement > max_pos:
                raise ValueError(f"Alteration movement ({alteration_movement}) exceeds suggested maximum positive movement ({max_pos}) for point {mtm_point}")
            elif alteration_movement < min_pos:
                raise ValueError(f"Alteration movement ({alteration_movement}) is less than suggested minimum positive movement ({min_pos}) for point {mtm_point}")
        elif alteration_movement < 0:
            if abs(alteration_movement) > abs(max_neg):
                raise ValueError(f"Alteration movement ({alteration_movement}) exceeds suggested maximum negative movement ({max_neg}) for point {mtm_point}")
            elif abs(alteration_movement) < abs(min_neg):
                raise ValueError(f"Alteration movement ({alteration_movement}) is less than suggested minimum negative movement ({min_neg}) for point {mtm_point}")

    def apply_no_extension(self, row, selected_df, extension_type="CCW", tolerance=0):
        mtm_point = row['mtm points']
        mtm_dependent = row['mtm_dependent']

        # Get movement limits 
        max_pos = row['maximum_movement_inches_positive']
        max_neg = row['maximum_movement_inches_negative']
        min_pos = row['minimum_movement_inches_positive']
        min_neg = row['minimum_movement_inches_negative']   

        self.check_alteration_limits(self.alteration_movement, max_pos, max_neg, min_pos, min_neg, mtm_point)

        try:
            selected_df_copy = selected_df.copy()
            p1, p2 = self._get_point_coordinates(mtm_point, mtm_dependent, selected_df_copy)
            movement_x, movement_y = row['movement_x'], row['movement_y']

            # Calculate point orders
            start_point_order = self._get_point_order(mtm_point, selected_df_copy)
            end_point_order = self._get_point_order(mtm_dependent, selected_df_copy)

            # Add distance calculations to DataFrame for proximity
            selected_df_copy = self._add_distance_to_points(selected_df_copy, p1, p2)

            # Get points between mtm_point and mtm_dependent
            points_in_range = self._get_points_in_range(selected_df_copy, start_point_order, end_point_order, tolerance, mtm_point, mtm_dependent)

            # Move the MTM point
            new_mtm_x = p1[0] + (self.alteration_movement * movement_x)
            new_mtm_y = p1[1] + (self.alteration_movement * movement_y)

            selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_point, ['pl_point_altered_x', 'pl_point_altered_y']] = [new_mtm_x, new_mtm_y]
            logging.info(f"MTM point {mtm_point} moved from ({p1[0]}, {p1[1]}) to ({new_mtm_x}, {new_mtm_y})")

            # Calculate the movement vector
            movement_vector = np.array([new_mtm_x, new_mtm_y]) - np.array(p1)

            # Move all points except the MTM dependent
            for i, point in points_in_range.iterrows():
                if point['mtm points'] != mtm_dependent:
                    original_point = np.array([point['pl_point_x'], point['pl_point_y']])
                    new_point = original_point + movement_vector
                    selected_df_copy.loc[i, ['pl_point_altered_x', 'pl_point_altered_y']] = new_point

            logging.info(f"{extension_type} No Extension applied. MTM point {mtm_point} moved from ({p1[0]}, {p1[1]}) to ({new_mtm_x}, {new_mtm_y})")
            logging.info(f"MTM dependent {mtm_dependent} remains fixed at ({p2[0]}, {p2[1]})")

            selected_df_copy.to_csv("data/ccw_no_ext_df.csv", index=False)

            return row, selected_df_copy

        except Exception as e:
            logging.error(f"Failed to apply {extension_type} No Extension alteration: {e}")
            return row, selected_df  # Return original row and DataFrame without alteration

    def _add_distance_to_points(self, df, p1, p2):
        """Adds the Euclidean distance from p1 and p2 to each point in the DataFrame."""
        df['dist_to_p1'] = np.sqrt((df['pl_point_x'] - p1[0]) ** 2 + (df['pl_point_y'] - p1[1]) ** 2)
        df['dist_to_p2'] = np.sqrt((df['pl_point_x'] - p2[0]) ** 2 + (df['pl_point_y'] - p2[1]) ** 2)
        return df
    
    def apply_extension(self, row, selected_df, extension_type="CW", tolerance=0):
        """
        Apply CW or CCW Ext alteration to the MTM point itself, and record surrounding points for adjustment later.
        """
        mtm_point = row['mtm points']
        mtm_dependent = row['mtm_dependent']  

        # Get movement limits 
        max_pos = row['maximum_movement_inches_positive']
        max_neg = row['maximum_movement_inches_negative']
        min_pos = row['minimum_movement_inches_positive']
        min_neg = row['minimum_movement_inches_negative']   

        self.check_alteration_limits(self.alteration_movement, max_pos, max_neg, min_pos, min_neg, mtm_point)

        try:

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
        mtm_point = row['mtm points']
        mtm_dependent = row['mtm_dependent']

        # Get movement limits 
        max_pos = row['maximum_movement_inches_positive']
        max_neg = row['maximum_movement_inches_negative']
        min_pos = row['minimum_movement_inches_positive']
        min_neg = row['minimum_movement_inches_negative']   

        self.check_alteration_limits(self.alteration_movement, max_pos, max_neg, min_pos, min_neg, mtm_point)

        try:
            logging.info(f"Applying XY Move on {mtm_point}")

            # Case 1: Individual row alteration (mtm_dependent == mtm_point)
            if mtm_dependent == mtm_point:
                self.xy_move_step_counter +=1
                return self._apply_single_point_move(row, selected_df, mtm_point)

            # Case 2: Multiple row alteration (mtm_dependent != mtm_point)
            else:
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
        selected_df_copy = selected_df.copy()
        
        # Use altered coordinates if available, otherwise use original ones
        current_x = row['pl_point_altered_x'] if pd.notna(row['pl_point_altered_x']) and row['pl_point_altered_x'] != "" else row['pl_point_x']
        current_y = row['pl_point_altered_y'] if pd.notna(row['pl_point_altered_y']) and row['pl_point_altered_y'] != "" else row['pl_point_y']

        # Apply the movement based on the current altered coordinates
        new_x = current_x + (self.alteration_movement * row['movement_x'])
        new_y = current_y + (self.alteration_movement * row['movement_y'])

        logging.info(f"XY Move (movement_x, movement_y): {(row['movement_x'], row['movement_y'])}")

        # Update the altered values in the DataFrame copy
        mask = selected_df_copy['mtm points'] == mtm_point
        if mask.any():
            selected_df_copy.loc[mask, 'pl_point_altered_x'] = new_x
            selected_df_copy.loc[mask, 'pl_point_altered_y'] = new_y
            logging.info(f"XY Move: Altered MTM point {mtm_point} to new coordinates: ({new_x}, {new_y})")
        else:
            logging.warning(f"MTM point {mtm_point} not found in the DataFrame")

        # Check adjacent points and nearby notch points
        current_point_order = row['point_order']
        
        def find_nearest_notch(direction):
            if direction == "previous":
                nearby_points = selected_df_copy[selected_df_copy['point_order'] < current_point_order].sort_values('point_order', ascending=False)
            else:  # "next"
                nearby_points = selected_df_copy[selected_df_copy['point_order'] > current_point_order].sort_values('point_order')
            
            for _, point in nearby_points.iterrows():
                if 'notch' in str(point['notch_labels']):
                    return point
            return None

        def is_close_enough(point1, point2, threshold=0.2):
            return (abs(point1[0] - point2[0]) <= threshold and 
                    abs(point1[1] - point2[1]) <= threshold)

        def has_adjacent_notch(point):
            prev_point = selected_df_copy[selected_df_copy['point_order'] == point['point_order'] - 1]
            next_point = selected_df_copy[selected_df_copy['point_order'] == point['point_order'] + 1]
            return ('notch' in str(prev_point['notch_labels'].values[0]) if not prev_point.empty else False) or \
                   ('notch' in str(next_point['notch_labels'].values[0]) if not next_point.empty else False)

        def move_if_notch(adjacent_point, direction):
            if adjacent_point is not None:
                logging.info(f"Adjacent {direction} point: {adjacent_point['mtm points']}")
                if 'notch' in str(adjacent_point['notch_labels']):
                    adjacent_x = adjacent_point['pl_point_altered_x'] if pd.notna(adjacent_point['pl_point_altered_x']) and adjacent_point['pl_point_altered_x'] != "" else adjacent_point['pl_point_x']
                    adjacent_y = adjacent_point['pl_point_altered_y'] if pd.notna(adjacent_point['pl_point_altered_y']) and adjacent_point['pl_point_altered_y'] != "" else adjacent_point['pl_point_y']
                    
                    if is_close_enough((current_x, current_y), (adjacent_x, adjacent_y)) and not has_adjacent_notch(adjacent_point):
                        logging.info(f"Adjacent {direction} point is a close notch point without adjacent notches")
                        new_adjacent_x = adjacent_x + (self.alteration_movement * row['movement_x'])
                        new_adjacent_y = adjacent_y + (self.alteration_movement * row['movement_y'])
                        
                        mask = selected_df_copy['point_order'] == adjacent_point['point_order']
                        if mask.any():
                            selected_df_copy.loc[mask, 'pl_point_altered_x'] = new_adjacent_x
                            selected_df_copy.loc[mask, 'pl_point_altered_y'] = new_adjacent_y
                            logging.info(f"XY Move: Moved adjacent {direction} notch point to new coordinates: ({new_adjacent_x}, {new_adjacent_y})")
                        else:
                            logging.warning(f"Adjacent {direction} point not found in the DataFrame")
                    else:
                        if not is_close_enough((current_x, current_y), (adjacent_x, adjacent_y)):
                            logging.info(f"Adjacent {direction} notch point is not close enough to MTM point. Not moving.")
                        else:
                            logging.info(f"Adjacent {direction} notch point has another adjacent notch. Not moving.")

        prev_notch = find_nearest_notch("previous")
        next_notch = find_nearest_notch("next")

        move_if_notch(prev_notch, "previous")
        move_if_notch(next_notch, "next")

        return row, selected_df_copy
    
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

            # Get all MTM points
            all_mtm_points = selected_df_copy[selected_df_copy['mtm points'].notna()]['mtm points'].sort_values()

            # Filter DataFrame for the X Y MOVE alteration
            filtered_df = selected_df_copy[
                (selected_df_copy['mtm_dependent'] == selected_df_copy['mtm points']) &
                selected_df_copy['mtm_dependent'].notna() &
                selected_df_copy['mtm points'].notna() &
                (selected_df_copy['alteration_type'] == "X Y MOVE")
            ]

            xy_move_count = filtered_df['alteration_type'].value_counts().get('X Y MOVE', 0)

            if xy_move_count >= 1:
                logging.info(f"{xy_move_count} instance(s) of 'X Y MOVE' found. Processing each one.")

                for _, mtm_row in filtered_df.iterrows():
                    mtm_point = mtm_row['mtm points']
                    
                    # Get the coordinates of the current MTM point (altered if available, otherwise original)
                    mtm_coords = np.array([
                        mtm_row['pl_point_altered_x'] if pd.notna(mtm_row['pl_point_altered_x']) else mtm_row['pl_point_x'],
                        mtm_row['pl_point_altered_y'] if pd.notna(mtm_row['pl_point_altered_y']) else mtm_row['pl_point_y']
                    ])

                    # Get the movement of the MTM point
                    mtm_movement = np.array([mtm_row['movement_x'], mtm_row['movement_y']]) * self.alteration_movement

                    # Find the previous and next MTM points
                    previous_mtm = all_mtm_points[all_mtm_points < mtm_point].max()
                    next_mtm = all_mtm_points[all_mtm_points > mtm_point].min()

                    # Get all points between previous and next MTM points
                    points_in_range = selected_df_copy[
                        (selected_df_copy['point_order'] > selected_df_copy[selected_df_copy['mtm points'] == previous_mtm]['point_order'].iloc[0] if pd.notna(previous_mtm) else selected_df_copy['point_order'].min()) &
                        (selected_df_copy['point_order'] < selected_df_copy[selected_df_copy['mtm points'] == next_mtm]['point_order'].iloc[0] if pd.notna(next_mtm) else selected_df_copy['point_order'].max())
                    ]

                    # Get coordinates of previous and next MTM points
                    prev_mtm_coords = selected_df_copy[selected_df_copy['mtm points'] == previous_mtm][['pl_point_x', 'pl_point_y']].iloc[0].values if pd.notna(previous_mtm) else None
                    next_mtm_coords = selected_df_copy[selected_df_copy['mtm points'] == next_mtm][['pl_point_x', 'pl_point_y']].iloc[0].values if pd.notna(next_mtm) else None

                    # Apply movement to points with compression
                    for idx, point in points_in_range.iterrows():
                        # Skip points that have already been altered
                        if pd.notna(point['pl_point_altered_x']) and pd.notna(point['pl_point_altered_y']):
                            logging.info(f"Point {point['point_order']} already altered, skipping.")
                            continue

                        current_coords = np.array([point['pl_point_x'], point['pl_point_y']])
                        
                        if np.isnan(current_coords).any():
                            logging.warning(f"Point {point['point_order']} has NaN coordinates. Skipping.")
                            continue

                        # Calculate distance ratio for compression
                        if point['point_order'] < mtm_row['point_order']:
                            # Point is before the current MTM point
                            total_distance = np.linalg.norm(mtm_coords - prev_mtm_coords) if prev_mtm_coords is not None else np.linalg.norm(mtm_coords - current_coords)
                            point_distance = np.linalg.norm(mtm_coords - current_coords)
                        else:
                            # Point is after the current MTM point
                            total_distance = np.linalg.norm(next_mtm_coords - mtm_coords) if next_mtm_coords is not None else np.linalg.norm(current_coords - mtm_coords)
                            point_distance = np.linalg.norm(current_coords - mtm_coords)

                        distance_ratio = point_distance / total_distance if total_distance != 0 else 0

                        # Apply compressed movement
                        new_coords = current_coords + mtm_movement * (1 - distance_ratio)

                        # Check if the new position goes beyond the previous or next MTM point
                        if prev_mtm_coords is not None and point['point_order'] < mtm_row['point_order']:
                            if np.dot(new_coords - prev_mtm_coords, mtm_coords - prev_mtm_coords) < 0:
                                # Compress towards previous MTM point
                                new_coords = prev_mtm_coords + (mtm_coords - prev_mtm_coords) * distance_ratio
                        elif next_mtm_coords is not None and point['point_order'] > mtm_row['point_order']:
                            if np.dot(new_coords - mtm_coords, next_mtm_coords - mtm_coords) < 0:
                                # Compress towards next MTM point
                                new_coords = mtm_coords + (next_mtm_coords - mtm_coords) * distance_ratio

                        selected_df_copy.loc[idx, 'pl_point_altered_x'] = new_coords[0]
                        selected_df_copy.loc[idx, 'pl_point_altered_y'] = new_coords[1]

                        # Log the movement
                        movement = new_coords - current_coords
                        logging.info(f"Point {point['point_order']} moved by {movement}")

                    # Update the MTM point itself if it hasn't been altered
                    if pd.isna(selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_point, 'pl_point_altered_x'].iloc[0]):
                        new_mtm_coords = mtm_coords + mtm_movement
                        selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_point, 'pl_point_altered_x'] = new_mtm_coords[0]
                        selected_df_copy.loc[selected_df_copy['mtm points'] == mtm_point, 'pl_point_altered_y'] = new_mtm_coords[1]
                        logging.info(f"MTM point {mtm_point} moved to {new_mtm_coords}")
                    else:
                        logging.info(f"MTM point {mtm_point} already altered, skipping.")

            return selected_df_copy

        except Exception as e:
            logging.error(f"Failed to apply XY Move correction: {e}")
            return selected_df

    def _create_point_graph(self, df, mtm_point, mtm_coords, moved_notch_points):
        G = nx.Graph()
        for _, row in df.iterrows():
            if row['mtm points'] == mtm_point:
                pos = tuple(mtm_coords)
            elif row['point_order'] in moved_notch_points:
                pos = (row['pl_point_altered_x'], row['pl_point_altered_y'])
            else:
                pos = (row['pl_point_x'], row['pl_point_y'])
            
            if not np.isnan(pos).any():  # Only add nodes with valid positions
                G.add_node(row['point_order'], pos=pos, is_notch=pd.notna(row.get('notch_labels', pd.NA)), 
                           is_mtm=pd.notna(row.get('mtm points', pd.NA)), is_fixed_notch=row['point_order'] in moved_notch_points)
        
        # Connect adjacent points
        sorted_points = sorted(G.nodes())
        for i in range(len(sorted_points) - 1):
            G.add_edge(sorted_points[i], sorted_points[i+1])
        
        return G

    def _apply_force_directed_algorithm(self, G, mtm_point, mtm_coords, prev_mtm_coords, next_mtm_coords, moved_notch_points, iterations=50, k=0.1):
        if len(G) <= 1:
            logging.warning(f"Graph has {len(G)} nodes. Skipping force-directed algorithm.")
            return {node: data['pos'] for node, data in G.nodes(data=True)}

        try:
            # Initialize positions
            pos = {n: data['pos'] for n, data in G.nodes(data=True)}
            
            # Define fixed points
            fixed_points = [mtm_point]
            if prev_mtm_coords is not None:
                fixed_points.append(min(G.nodes()))  # Assume the first node is the previous MTM point
                pos[min(G.nodes())] = prev_mtm_coords
            if next_mtm_coords is not None:
                fixed_points.append(max(G.nodes()))  # Assume the last node is the next MTM point
                pos[max(G.nodes())] = next_mtm_coords

            for _ in range(iterations):
                for node in G.nodes():
                    if node not in fixed_points and node not in moved_notch_points:
                        # Calculate forces
                        force = np.zeros(2)
                        for neighbor in G.neighbors(node):
                            diff = np.array(pos[neighbor]) - np.array(pos[node])
                            distance = np.linalg.norm(diff)
                            if distance > 0:
                                force += k * diff / distance  # Spring force
                        
                        # Apply force
                        new_pos = np.array(pos[node]) + force
                        
                        # Ensure the point doesn't move below the previous MTM point
                        if prev_mtm_coords is not None:
                            new_pos[1] = max(new_pos[1], prev_mtm_coords[1])
                        
                        # Update position
                        pos[node] = tuple(new_pos)

                # Adjust positions to maintain string-like behavior
                sorted_nodes = sorted(G.nodes())
                for i in range(1, len(sorted_nodes) - 1):
                    prev_node = sorted_nodes[i-1]
                    curr_node = sorted_nodes[i]
                    next_node = sorted_nodes[i+1]
                    
                    prev_pos = np.array(pos[prev_node])
                    curr_pos = np.array(pos[curr_node])
                    next_pos = np.array(pos[next_node])
                    
                    # Calculate the ideal position based on neighbors
                    ideal_pos = (prev_pos + next_pos) / 2
                    
                    # Move the current point towards the ideal position
                    pos[curr_node] = tuple(curr_pos + 0.1 * (ideal_pos - curr_pos))

            # Handle NaN values
            nan_mask = np.array([np.isnan(pos[node]).any() for node in G.nodes()])
            if nan_mask.any():
                logging.warning(f"NaN values detected in {nan_mask.sum()} positions. Replacing with original positions.")
                for node in G.nodes():
                    if np.isnan(pos[node]).any():
                        pos[node] = G.nodes[node]['pos']

            return pos

        except Exception as e:
            logging.error(f"Error in force-directed algorithm: {str(e)}")
            return None

    def _group_notch_points(self, df):
        notch_groups = []
        notch_points = df[df['notch_labels'].str.contains('notch', na=False, case=False)].sort_values('point_order')
        
        if notch_points.empty:
            return notch_groups

        current_group = []
        prev_order = None

        for _, point in notch_points.iterrows():
            if prev_order is None or point['point_order'] - prev_order <= 1:
                current_group.append(point)
            else:
                if len(current_group) >= 3:  # Assuming a notch consists of at least 3 points
                    notch_groups.append(current_group)
                current_group = [point]
            prev_order = point['point_order']

        # Add the last group if it meets the criteria
        if len(current_group) >= 3:
            notch_groups.append(current_group)

        return notch_groups

    def post_alteration_point_shift(self, selected_df):
        logging.info("Starting post-alteration point shift")
        selected_df_copy = selected_df.copy()

        # Get all MTM points and their dependents
        all_key_points = selected_df_copy[
            (selected_df_copy['mtm points'].notna()) | 
            (selected_df_copy['mtm_dependent'].notna())
        ].sort_values('point_order')
        
        logging.info(f"All MTM points and dependents: {all_key_points['mtm points'].fillna(all_key_points['mtm_dependent']).tolist()}")

        for i in range(len(all_key_points) - 1):
            current_point = all_key_points.iloc[i]
            next_point = all_key_points.iloc[i+1]

            order1 = current_point['point_order']
            order2 = next_point['point_order']

            movement1_x = current_point['pl_point_altered_x'] - current_point['pl_point_x']
            movement1_y = current_point['pl_point_altered_y'] - current_point['pl_point_y']
            movement2_x = next_point['pl_point_altered_x'] - next_point['pl_point_x']
            movement2_y = next_point['pl_point_altered_y'] - next_point['pl_point_y']

            point1_label = current_point['mtm points'] if pd.notna(current_point['mtm points']) else current_point['mtm_dependent']
            point2_label = next_point['mtm points'] if pd.notna(next_point['mtm points']) else next_point['mtm_dependent']

            logging.info(f"Processing points between: {point1_label} and {point2_label}")
            logging.info(f"  Point order range: {order1} to {order2}")
            logging.info(f"  Movement at current point: ({movement1_x:.4f}, {movement1_y:.4f})")
            logging.info(f"  Movement at next point: ({movement2_x:.4f}, {movement2_y:.4f})")

            # Get points in range, including the endpoints
            points_in_range = selected_df_copy[
                (selected_df_copy['point_order'] >= order1) & 
                (selected_df_copy['point_order'] <= order2)
            ]

            for idx, point in points_in_range.iterrows():
                if pd.notna(point['mtm points']) or pd.notna(point['mtm_dependent']):
                    # This is an MTM point or dependent point, use its existing alteration
                    shift_x = point['pl_point_altered_x'] - point['pl_point_x']
                    shift_y = point['pl_point_altered_y'] - point['pl_point_y']
                elif 'notch' in str(point.get('notch_labels', '')).lower():
                    # This is a notch point, check if it was already moved
                    if pd.notna(point['pl_point_altered_x']) and pd.notna(point['pl_point_altered_y']):
                        # Notch point was already moved, keep its current position
                        shift_x = point['pl_point_altered_x'] - point['pl_point_x']
                        shift_y = point['pl_point_altered_y'] - point['pl_point_y']
                        logging.info(f"    Preserving previously moved notch point {point['point_order']}")
                    else:
                        # Notch point wasn't moved, interpolate its movement
                        t = (point['point_order'] - order1) / (order2 - order1)
                        shift_x = self.cubic_interpolation(t, movement1_x, movement2_x)
                        shift_y = self.cubic_interpolation(t, movement1_y, movement2_y)
                else:
                    # This is a regular point, interpolate its movement
                    t = (point['point_order'] - order1) / (order2 - order1)
                    shift_x = self.cubic_interpolation(t, movement1_x, movement2_x)
                    shift_y = self.cubic_interpolation(t, movement1_y, movement2_y)
                
                # Apply the shift
                selected_df_copy.loc[idx, 'pl_point_altered_x'] = point['pl_point_x'] + shift_x
                selected_df_copy.loc[idx, 'pl_point_altered_y'] = point['pl_point_y'] + shift_y
                
                point_type = 'MTM' if pd.notna(point['mtm points']) else 'Dependent' if pd.notna(point['mtm_dependent']) else 'Regular'
                logging.info(f"    Point order {point['point_order']} ({point_type}, Notch: {point['notch_labels']}): shift ({shift_x:.4f}, {shift_y:.4f})")

        # Verify changes
        changed_points = selected_df_copy[
            (selected_df_copy['pl_point_altered_x'] != selected_df_copy['pl_point_x']) |
            (selected_df_copy['pl_point_altered_y'] != selected_df_copy['pl_point_y'])
        ]
        logging.info(f"Total points changed: {len(changed_points)}")

        # Create debug DataFrame
        debug_columns = ['point_order', 'mtm points', 'mtm_dependent', 'notch_labels', 'pl_point_x', 'pl_point_y', 
                        'pl_point_altered_x', 'pl_point_altered_y']
        debug_df = selected_df_copy[debug_columns].copy()  # Create an explicit copy

        # Calculate movements using vectorized operations
        debug_df.loc[:, 'movement_x'] = debug_df['pl_point_altered_x'] - debug_df['pl_point_x']
        debug_df.loc[:, 'movement_y'] = debug_df['pl_point_altered_y'] - debug_df['pl_point_y']

        debug_df.to_csv('debug_post_alteration_shift.csv', index=False)
        logging.info("Debug information saved to 'debug_post_alteration_shift.csv'")

        return selected_df_copy

    def _move_notch_group(self, group, df, movement_x, movement_y):
        for point in group:
            idx = point.name
            df.loc[idx, 'pl_point_altered_x'] = point['pl_point_x'] + movement_x
            df.loc[idx, 'pl_point_altered_y'] = point['pl_point_y'] + movement_y
            df.loc[idx, 'movement_x'] = movement_x
            df.loc[idx, 'movement_y'] = movement_y
        
        logging.info(f"Moved notch group (size: {len(group)}) by ({movement_x:.4f}, {movement_y:.4f})")

    def _move_notch_group(self, group, df, movement_x, movement_y):
        for point in group:
            idx = point.name
            df.loc[idx, 'pl_point_altered_x'] = point['pl_point_x'] + movement_x
            df.loc[idx, 'pl_point_altered_y'] = point['pl_point_y'] + movement_y
        
        logging.info(f"Moved notch group (size: {len(group)}) by ({movement_x:.4f}, {movement_y:.4f})")

    def cubic_interpolation(self, t, start, end):
        # Cubic function: f(t) = -2t^3 + 3t^2
        # This gives a smooth S-curve interpolation
        t = -2 * t**3 + 3 * t**2
        return start + (end - start) * t

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
    
    #debug_alteration_rule = "1LTH-FSLV"
    #debug_alteration_rule = "FRT-HEIGHT"
    #debug_alteration_rule = "2SL-BICEP"

    ## LGFG-SH-01-CCB-FO
    #debug_alteration_rule = "1LTH-BACK"
    debug_alteration_rule = "1LTH-FRONT"
    #debug_alteration_rule = "1LTH-FULL"
    #debug_alteration_rule = "2ARMHOLEDN"
    #debug_alteration_rule = "2ARMHOLEIN"
    #debug_alteration_rule = "3-COLLAR"
    #debug_alteration_rule = "3-SHOULDER"
    #debug_alteration_rule = "4-CHEST"
    #debug_alteration_rule = "4-HIP"
    #debug_alteration_rule = "4-WAIST"
    #debug_alteration_rule = "4CHESTACRS" 
    #debug_alteration_rule = "5-DARTBACK"
    #debug_alteration_rule = "6-PLACKET"
    #debug_alteration_rule = "7F-BELLY"
    #debug_alteration_rule = "7F-ERECT"
    #debug_alteration_rule = "7F-SH-BKSL"
    #debug_alteration_rule = "7F-SHPOINT"
    #debug_alteration_rule = "7F-SHSLOPE"
    #debug_alteration_rule = "7F-SHSQUAR"
    #debug_alteration_rule = "7F-STOOPED"
    #debug_alteration_rule = "HIGH-CHEST"
    #debug_alteration_rule = "LONG-BACK"
    #debug_alteration_rule = "OPEN-CLR"
    #debug_alteration_rule = "ROUND-BACK"
    #debug_alteration_rule = "SHORT-BACK"
    #debug_alteration_rule = "WAISTSHAPE"
    #debug_alteration_rule = "WAISTSMOTH"


    alteration_movement = 0.75 # INCHES (can be positive or negative)
    
    make_alteration = PieceAlterationProcessor(piece_table_path=piece_table_path,
                                               vertices_table_path=vertices_table_path,
                                               debug_alteration_rule=debug_alteration_rule, 
                                               alteration_movement = alteration_movement)
    make_alteration.process_alterations()
    #make_alteration.log_info(debug_alteration_rule)