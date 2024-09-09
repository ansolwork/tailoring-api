import pandas as pd
import ast
import numpy as np
from app.smoothing import SmoothingFunctions  # Import the SmoothingFunctions class
from utils.data_processing_utils import DataProcessingUtils
import os

class MakeAlterationByPiece:
    """
    Handles the processing of alteration rules.

    :param alteration_df: DataFrame containing the alteration data.
    :param piece_name: Name of the piece being processed.
    :param alteration_rule: Rule for applying alterations.
    """

    def __init__(self, 
                staged_piece_table_path=None, 
                staged_vertices_table_path=None, 
                save_folder_processed_pieces="data/staging_processed/processed_alterations_by_piece/", 
                save_folder_processed_vertices="data/staging_processed/processed_vertices_by_piece/", 
                save_file_format=".csv"):
        
        self.processing_utils = DataProcessingUtils()

        # Load the piece and vertices data
        self.staged_piece_df = self.processing_utils.load_csv(staged_piece_table_path)
        self.staged_vertices_df = self.processing_utils.load_csv(staged_vertices_table_path)

        # Get the unique alteration rules
        self.alteration_rules = self.get_alteration_rules()

        # Dictionary to store DataFrames by alteration rule
        self.alteration_dfs = self.split_by_alteration_rule()

        # Piece name (assuming only one unique piece name)
        self.piece_name = self.get_piece_name()

        # Save Options
        self.save_folder_processed_pieces = save_folder_processed_pieces
        self.save_folder_processed_vertices = save_folder_processed_vertices
        self.save_file_format = save_file_format

    def get_alteration_rules(self):
        """
        Returns a list of unique alteration rules.
        """
        return self.staged_piece_df['alteration_rule'].dropna().unique().tolist()

    def get_piece_name(self):
        """
        Returns the unique piece name from the piece_name column.
        Assumes there is only one unique piece name.
        """
        piece_name = self.staged_piece_df['piece_name'].dropna().unique()
        if len(piece_name) == 1:
            return piece_name[0]
        else:
            raise ValueError("There should only be one unique piece name, but multiple were found.")

    def split_by_alteration_rule(self):
        """
        Splits the staged_piece_df into multiple DataFrames, organized by alteration rule.
        Rows without an alteration rule (NaN values) are included in each DataFrame.
        Returns a dictionary where the keys are alteration rules and the values are DataFrames.
        """
        # Separate rows where alteration_rule is NaN
        no_rule_df = self.staged_piece_df[self.staged_piece_df['alteration_rule'].isna()]

        # Create a dictionary to hold DataFrames split by alteration_rule
        alteration_dfs = {}

        # Group the DataFrame by alteration_rule
        for rule, group in self.staged_piece_df.groupby('alteration_rule'):
            # Combine rows with NaN alteration rules with the group
            combined_df = pd.concat([group, no_rule_df])
            alteration_dfs[rule] = combined_df
            #self.processing_utils.save_csv(combined_df, "data/debug/" + rule + ".csv")

        return alteration_dfs
    
    def prepare_dataframe(self, df):
        """Prepares the DataFrame by converting columns to the correct types and adding necessary columns."""
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
        return df
    
    def apply_alteration_rules(self):
        for alteration_rule in self.alteration_rules:
            selected_df = self.alteration_dfs[alteration_rule]
            selected_df = self.prepare_dataframe(selected_df)

            # Next: Run Other Functions

            # Debug
            break 
    
    def print_info(self):
        """
        Print Alteration Info (for Debugging)
        """
        print(f"\nMake alteration on Piece Name: {self.piece_name}")

        # Define the alteration type you're looking for
        alteration_rule = "7F-SHPOINT"

        # Check if the alteration type exists in the alteration DataFrame
        if alteration_rule in self.alteration_dfs:
            print(f"\nAlteration DFs on Alteration Type {alteration_rule}:\n")
            print(self.alteration_dfs[alteration_rule])
        else:
            print(f"\nAlteration type '{alteration_rule}' not found in the DataFrame.")


if __name__ == "__main__":

    staged_piece_table_path = "data/staging/alteration_by_piece/combined_table_LGFG-SH-01-CCB-FO.csv"
    staged_vertices_table_path = "data/staging/vertices/vertices_LGFG-SH-01-CCB-FO.csv"

    make_alteration = MakeAlterationByPiece(staged_piece_table_path=staged_piece_table_path,
                                            staged_vertices_table_path=staged_vertices_table_path)
    make_alteration.apply_alteration_rules()
    make_alteration.print_info()

