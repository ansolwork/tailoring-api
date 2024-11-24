import pandas as pd
import os
import numpy as np
import re
from tqdm import tqdm
import shutil
import logging

# TODO: Fix issue why we are not getting enough Alteration Rules in the output
# TODO: Mark Line Points
# Join with point specification Table 
# IMplement sorting of MTM Points

class CreateTable:
    def __init__(self, alteration_filepath, combined_entities_folder, item, is_graded=False, debug=False):
        self.alteration_filepath = alteration_filepath
        self.combined_entities_folder = combined_entities_folder
        self.is_graded = is_graded
        self.debug = debug  # Add debug flag
        self.df_dict = self.load_table()  # Loading all sheets as a dictionary of DataFrames
        self.alteration_joined_df = pd.DataFrame()
        self.combined_entities_joined_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.item = item
        
        # Set output directories based on is_graded flag
        if self.is_graded:
            self.output_dir = f"data/staging/graded/{item}"
            self.output_table_path = f"data/staging/graded/{item}/combined_alteration_tables"
            self.output_table_path_by_piece = f"data/staging/graded/{item}/alteration_by_piece"
        else:
            self.output_dir = f"data/staging/base/{item}"
            self.output_table_path = f"data/staging/base/{item}/combined_alteration_tables"
            self.output_table_path_by_piece = f"data/staging/base/{item}/alteration_by_piece"

        # Add more sheets if necessary
        self.sheet_list = ["SHIRT-FRONT", "SHIRT-BACK", "SHIRT-YOKE", "SHIRT-SLEEVE", 
                           "SHIRT-COLLAR", "SHIRT-COLLAR BAND", "SHIRT-CUFF", "SQUARE-PIECE", "CIRCLE-PIECE"]

    def load_table(self):
        # Load all sheets into a dictionary of DataFrames
        df_dict = pd.read_excel(self.alteration_filepath, sheet_name=None)
        return df_dict
        
    @staticmethod
    def load_csv(filepath):
        df_csv = pd.read_csv(filepath)
        return df_csv
    
    def join_mtm_details(self):
        # Load MTM point details from the dictionary
        mtm_point_details = self.df_dict["MTM POINT NUMBERS"]

        # Ensure 'mtm points' are floats, and drop NaNs or invalid values before processing
        self.alteration_joined_df = self.alteration_joined_df[self.alteration_joined_df['mtm points'].notna()]

        # Convert the 'mtm points' to integers by extracting the whole number part
        self.alteration_joined_df['mtm points'] = self.alteration_joined_df['mtm points'].astype(int)

        # Now extract the first two digits of 'mtm points' to match with 'mtm_group'
        self.alteration_joined_df['mtm_group'] = self.alteration_joined_df['mtm points'].astype(str).str[:2].astype(int)

        # Merge MTM point details with alteration_joined_df based on 'mtm_group'
        self.alteration_joined_df = pd.merge(
            self.alteration_joined_df, 
            mtm_point_details, 
            on='mtm_group', 
            how='left'
        )
    
    def process_table(self):
        # Filter the sheets dictionary to only include the specified sheets
        selected_sheets = {name: self.df_dict[name] for name in self.sheet_list if name in self.df_dict}

        # Concatenate the DataFrames into a single DataFrame
        big_df = pd.concat(selected_sheets.values(), ignore_index=True)
        big_df = big_df.rename(columns={'mtm_alteration': 'mtm points'})

        self.alteration_joined_df = big_df
        self.join_mtm_details()
    
    def process_combined_entities(self):
        combined_df_list = []
        
        # Get list of files first
        files = [f for f in os.listdir(self.combined_entities_folder) if f.endswith(".xlsx")]
        
        # Create progress bar
        file_type = "graded" if self.is_graded else "base"
        for filename in tqdm(files, desc=f"Processing {file_type} files"):
            filepath = os.path.join(self.combined_entities_folder, filename)
            if self.debug:
                print(f"\nProcessing file: {filepath}")
            
            combined_df = pd.read_excel(filepath)
            
            if 'Filename' in combined_df.columns:
                combined_df = combined_df.rename(columns={'Filename': 'piece_name'})
            
            if self.debug:
                print("Original piece_name values:", combined_df['piece_name'].unique())
                print(f"Initial columns: {combined_df.columns.tolist()}")
                print(f"Unique sizes before processing: {combined_df['size'].unique()}")
            
            # Initialize size column
            combined_df['size'] = None
            
            if self.is_graded:
                # First try to get size from filename
                combined_df['size'] = combined_df['piece_name'].str.extract(r'-(\d+)\.dxf$').astype(str)
                
                # If size not found in filename, check text entities
                text_rows = combined_df[combined_df['Type'] == 'TEXT']
                if not text_rows.empty:
                    for _, row in text_rows.iterrows():
                        if 'Text' in row and pd.notna(row['Text']):
                            # Look for pattern like "30 - 62 (39)"
                            size_match = re.search(r'\((\d+)\)', str(row['Text']))
                            if size_match:
                                size = size_match.group(1)
                                # Update size for all rows with same piece_name
                                piece_mask = combined_df['piece_name'] == row['piece_name']
                                combined_df.loc[piece_mask, 'size'] = size
                
                # Extract base piece name (remove only size and .dxf)
                combined_df['piece_name'] = combined_df['piece_name'].str.replace(r'-\d+\.dxf$', '', regex=True)
                
                # Ensure we're not losing any part of the piece name
                combined_df['piece_name'] = combined_df['piece_name'].astype(str)
                
                if self.debug:      
                    print("After processing graded files:")
                    print("Piece names:", combined_df['piece_name'].unique())
                    print("Sizes:", combined_df['size'].unique())
            else:
                # For base files, extract size from the filename
                base_piece_name = filename.replace('_combined_entities_labeled.xlsx', '')
                # Modified regex to capture size correctly
                size_match = re.search(r'-(\d+)(?:\.dxf)?$', base_piece_name)
                if size_match:
                    size = size_match.group(1)
                    # Remove only the size number from the end
                    piece_name = re.sub(r'-\d+(?:\.dxf)?$', '', base_piece_name)
                    combined_df['size'] = str(size)  # Ensure size is string
                    combined_df['piece_name'] = piece_name
                else:
                    piece_name = base_piece_name.replace('_combined_entities', '')
                    combined_df['piece_name'] = piece_name
            
            # After size extraction but before merging alteration rules
            if self.debug and '39' in combined_df['size'].unique():
                print("\nFound size 39 data:")
                size_39_data = combined_df[combined_df['size'] == '39']
                print(f"Number of size 39 rows: {len(size_39_data)}")
                print(f"Available columns for size 39: {size_39_data.columns[size_39_data.notna().any()].tolist()}")
            
            # Join with alteration rules
            if 'alteration_rules' in self.df_dict:
                combined_df = pd.merge(
                    combined_df,
                    self.df_dict['alteration_rules'],
                    on=['size', 'piece_name'],  # Make sure we're joining on correct keys
                    how='left'
                )
                
                if self.debug and '39' in combined_df['size'].unique():
                    print("\nAfter joining alteration rules:")
                    size_39_data = combined_df[combined_df['size'] == '39']
                    print(f"Size 39 rows with rules: {len(size_39_data[size_39_data['alteration_name'].notna()])}")
            
            # Ensure piece_name is string type before concatenation
            combined_df['piece_name'] = combined_df['piece_name'].astype(str)
            if self.debug:      
                print("Final piece_name values:", combined_df['piece_name'].unique())
            combined_df_list.append(combined_df)
        
        # Combine all the DataFrames
        final_combined_df = pd.concat(combined_df_list, ignore_index=True) if combined_df_list else pd.DataFrame()
        
        # Ensure piece_name is preserved after concatenation
        final_combined_df['piece_name'] = final_combined_df['piece_name'].astype(str)
        final_combined_df.columns = final_combined_df.columns.str.lower()
        
        # Drop unnecessary columns
        columns_to_drop = ['color', 'type', 'layer', 'color', 'point_x', 'point_y', 'height', 
                          'style', 'text', 'block', 'line_start_x', 'line_start_y', 'line_end_x', 'line_end_y',
                          'vertex_index', 'point label', 'vertex label']
        
        final_combined_df = final_combined_df.drop(columns=[col for col in columns_to_drop if col in final_combined_df.columns])
        
        if self.debug:  
            print("\nDEBUG after processing:")
            print(f"Columns in final_combined_df: {final_combined_df.columns.tolist()}")
            print(f"Sample sizes in final df: {final_combined_df['size'].unique()[:5]}")
            print(f"Sample piece names in final df: {final_combined_df['piece_name'].unique()[:5]}")
        
        self.combined_entities_joined_df = final_combined_df
    
    def merge_tables(self):
        # Your existing merge code
        self.merged_df = pd.merge(
            self.alteration_joined_df, 
            self.combined_entities_joined_df, 
            on=['piece_name', 'mtm points'], 
            how='right'
        )
            
        # Keep only rows that had a size
        has_size = self.merged_df['size'].notna()
        self.merged_df = self.merged_df[has_size]
        
        # Sort by piece_name and size
        self.merged_df = self.merged_df.sort_values(['piece_name', 'size'])

        self.print_alteration_rule_counts()

    def print_alteration_rule_counts(self):
        if self.debug:      
            print("\nAlteration Rule Counts Matrix (Size × Rule):")
        
        # Create pivot table
        pivot = pd.pivot_table(
            self.merged_df, 
            values='mtm points',
            index='alteration_rule',
            columns='size',
            aggfunc='count',
            fill_value=0
        )
        
        # Sort columns (sizes) numerically
        pivot = pivot.reindex(sorted(pivot.columns, key=int), axis=1)
        
        # Print the matrix
        if self.debug:  
            print(pivot)

    def save_table_csv_by_alteration_rule(self, output_filename_prefix="combined_table"):
        output_table_path = self.output_table_path
        os.makedirs(output_table_path, exist_ok=True)

        self.merged_df['piece_name'] = self.merged_df['piece_name'].str.replace(".dxf", "", regex=False)

        # Group the DataFrame by 'alteration_rule'
        grouped = self.merged_df.groupby('alteration_rule')
        
        for alteration_rule, group_df in grouped:
            # Convert all column headers to lowercase
            group_df.columns = group_df.columns.str.lower()
            
            # Sort by size (convert to integer for proper numerical sorting)
            group_df['size'] = pd.to_numeric(group_df['size'], errors='coerce')
            group_df = group_df.sort_values('size')
                        
            # Create a sanitized file name for each alteration_rule
            safe_alteration_rule = str(alteration_rule).replace(" ", "_").replace("/", "_")
            output_file_path = os.path.join(output_table_path, f"{output_filename_prefix}_{safe_alteration_rule}.csv")
            
            # Save the group as a CSV file
            group_df.to_csv(output_file_path, index=False)

            if self.debug:  
                print(f"CSV files saved to {output_table_path}")

    def save_table_csv_by_piece_name(self, output_filename_prefix="combined_table"):
        output_table_path = self.output_table_path_by_piece
        os.makedirs(output_table_path, exist_ok=True)

        grouping_column = 'base_piece_name' if 'base_piece_name' in self.merged_df.columns else 'piece_name'
        grouped = self.merged_df.groupby(grouping_column)

        for base_piece_name, group_df in grouped:
            # Convert all column headers to lowercase
            group_df.columns = group_df.columns.str.lower()

            # Convert size to numeric for proper sorting
            group_df['size'] = pd.to_numeric(group_df['size'], errors='coerce')
            
            # Sort by size, then alteration_rule, then mtm points
            group_df = group_df.sort_values(['size', 'alteration_rule', 'mtm points'])

            safe_piece_name = str(base_piece_name).replace(" ", "_").replace("/", "_")
            output_file_path = os.path.join(output_table_path, f"{output_filename_prefix}_{safe_piece_name}.csv")

            group_df = group_df.drop('vertices', axis=1)
            
            if 'base_piece_name' in group_df.columns:
                group_df = group_df.drop('base_piece_name', axis=1)

            # Drop duplicate rows
            group_df = group_df.dropna(how='all', subset=[col for col in group_df.columns if col != 'piece_name'])

            # Reset the index and add point_order starting from 0
            group_df = group_df.reset_index(drop=True)
            group_df['point_order'] = group_df.index
            
            # Save the group as a CSV file
            group_df.to_csv(output_file_path, index=False)

            if self.debug:  
                print(f"CSV file saved to {output_file_path}")

    def add_other_mtm_points(self):
        for filename in os.listdir(self.output_table_path):
            if filename.endswith(".csv"):
                # Load the CSV file
                load_path = os.path.join(self.output_table_path, filename)
                df = self.load_csv(load_path)

                # Iterate over each unique piece_name in the DataFrame
                unique_piece_names = df['piece_name'].unique()

                for piece_name in unique_piece_names:
                    # Find matching rows in the combined_entities_joined_df for the current piece_name
                    matching_rows = self.combined_entities_joined_df[self.combined_entities_joined_df['piece_name'] == piece_name]

                    # Concatenate the DataFrames
                    df = pd.concat([df, matching_rows], axis=0)

                # Ensure 'vertices' column is dropped
                if 'vertices' in df.columns:
                    df.drop(columns=['vertices'], inplace=True)

                # Remove rows where all columns except for specific ones (like 'piece_name') are NaN
                df.dropna(how='all', subset=[col for col in df.columns if col != 'piece_name'], inplace=True)

                # Save the updated DataFrame back to CSV
                df.to_csv(load_path, index=False)

               #print(f"Updated DataFrame saved to {load_path}")

    def create_vertices_df(self):
        # Base directory where all piece_name directories will be created
        base_dir = os.path.join(self.output_dir, "vertices")
        os.makedirs(base_dir, exist_ok=True)

        if self.debug:
            print("\nDEBUG create_vertices_df:")
            print(f"Available columns: {self.combined_entities_joined_df.columns.tolist()}")
            print(f"Total rows in combined_entities: {len(self.combined_entities_joined_df)}")

        # Get the unique piece names
        unique_piece_names = self.combined_entities_joined_df['piece_name'].unique()

        # Loop through each unique piece_name
        for piece_name in unique_piece_names:
            # Remove ".dxf" extension from piece_name
            sanitized_piece_name = piece_name.replace(".dxf", "")

            # Filter rows where piece_name matches the current piece_name
            matching_rows = self.combined_entities_joined_df[self.combined_entities_joined_df['piece_name'] == piece_name]
            
            vertices_data = []
            
            # Process each row individually to ensure we don't miss any vertices
            for _, row in matching_rows.iterrows():
                if 'vertices' in row and pd.notna(row['vertices']):
                    size = row['size'] if 'size' in row and pd.notna(row['size']) else 'base'
                    vertices_data.append({
                        'piece_name': sanitized_piece_name,
                        'size': size,
                        'vertices': row['vertices']
                    })

            # Create DataFrame from vertices data
            if vertices_data:  # Only create DataFrame if we have data
                vertices_df = pd.DataFrame(vertices_data)
                
                if self.debug:
                    print(f"\nProcessed {piece_name}:")
                    print(f"Found {len(vertices_data)} total vertices")
                    if 'size' in vertices_df.columns:
                        print("\nVertices per size before deduplication:")
                        size_counts = vertices_df.groupby('size').size()
                        for size, count in size_counts.items():
                            print(f"Size {size}: {count} vertices")
                
                # Sort by size if it exists and has multiple values
                if 'size' in vertices_df.columns and vertices_df['size'].nunique() > 1:
                    vertices_df['size'] = vertices_df['size'].astype(str)
                    vertices_df = vertices_df.sort_values('size')
                    
                    # Remove duplicates within each size group
                    vertices_df = vertices_df.drop_duplicates(subset=['size', 'vertices'])
                
                if self.debug:
                    print("\nVertices per size after deduplication:")
                    size_counts = vertices_df.groupby('size').size()
                    for size, count in size_counts.items():
                        print(f"Size {size}: {count} vertices")
                
                # Save the DataFrame to a CSV file
                output_path = os.path.join(base_dir, f'vertices_{sanitized_piece_name}.csv')
                vertices_df.to_csv(output_path, index=False)

    def clean_staging_folder(self):
        """Clean the staging folder before saving new data"""
        folders_to_clean = [
            self.output_dir,
            self.output_table_path,
            self.output_table_path_by_piece
        ]
        
        for folder in folders_to_clean:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    if self.debug:
                        print(f"Cleaned directory: {folder}")
                except Exception as e:
                    logging.error(f"Error cleaning directory {folder}: {str(e)}")

if __name__ == "__main__":
    alteration_filepath = "data/input/mtm_points.xlsx"
    item = "shirt"
    debug = False  # Set debug flag here
    
    # Process base files
    print("\nProcessing base files...")
    combined_entities_folder = f"data/input/mtm_combined_entities_labeled/{item}/"
    create_table = CreateTable(
        alteration_filepath, 
        combined_entities_folder,
        item,
        is_graded=False,
        debug=debug
    )
    
    # Clean staging folder before processing
    create_table.clean_staging_folder()
    
    # Process the sheets and get the combined DataFrame
    create_table.process_table()
    create_table.process_combined_entities()
    create_table.create_vertices_df()
    create_table.merge_tables()
    create_table.save_table_csv_by_alteration_rule()
    create_table.save_table_csv_by_piece_name()
    create_table.add_other_mtm_points()

    # Process graded files
    print("\nProcessing graded files...")
    graded_entities_folder = f"data/input/graded_mtm_combined_entities_labeled/{item}/all_sizes_merged/"
    create_table_graded = CreateTable(
        alteration_filepath, 
        graded_entities_folder,
        item,
        is_graded=True,
        debug=debug
    )
    
    # Clean staging folder before processing graded files
    create_table_graded.clean_staging_folder()
    
    # Process the sheets and get the combined DataFrame for graded files
    create_table_graded.process_table()
    create_table_graded.process_combined_entities()
    create_table_graded.create_vertices_df()
    create_table_graded.merge_tables()
    create_table_graded.save_table_csv_by_alteration_rule()
    create_table_graded.save_table_csv_by_piece_name()
    create_table_graded.add_other_mtm_points()

    # DEBUG
    #print("\n# Debug: All Unique Processed Piece Names. A common error is if they do not match the Actual piece name. \nCheck for extra spaces, incomplete names or unwanted extensions (e.g. .dxf)\n")
    #print(f"Processed Piece Names: {create_table.combined_entities_joined_df['piece_name'].unique()}\n")
    #print(f"Combined Entities (They should match): {create_table.alteration_joined_df['piece_name'].unique()}")
    #print("\nNOTE: If there are pieces in the Processed Piece Names that do not exist in the Combined entities or vice versa, then that table needs to be created")