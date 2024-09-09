import pandas as pd
import os
import numpy as np

class CreateTable:
    def __init__(self, alteration_filepath, combined_entities_folder):
        self.alteration_filepath = alteration_filepath
        self.combined_entities_folder = combined_entities_folder
        self.df_dict = self.load_table()  # Loading all sheets as a dictionary of DataFrames
        self.alteration_joined_df = pd.DataFrame()
        self.combined_entities_joined_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.output_dir = "data/staging"
        self.output_table_path = "data/staging/combined_alteration_tables"
        self.output_table_path_by_piece = "data/staging/alteration_by_piece"

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
    
    def process_table(self):
        # Filter the sheets dictionary to only include the specified sheets
        selected_sheets = {name: self.df_dict[name] for name in self.sheet_list if name in self.df_dict}
        
        # Concatenate the DataFrames into a single DataFrame
        big_df = pd.concat(selected_sheets.values(), ignore_index=True)
        big_df = big_df.rename(columns={'mtm_alteration': 'mtm points'})

        self.alteration_joined_df = big_df
    
    def process_combined_entities(self):
        combined_df_list = []
        
        # Iterate over each Excel file in the combined entities folder
        for filename in os.listdir(self.combined_entities_folder):
            if filename.endswith(".xlsx"):
                filepath = os.path.join(self.combined_entities_folder, filename)
                #print(f"Processing file: {filepath}")
                
                # Load the current Excel file
                combined_df = pd.read_excel(filepath)
                
                # Rename the 'Filename' column to 'piece_name' and remove the '.dxf' extension
                if 'Filename' in combined_df.columns:
                    combined_df = combined_df.rename(columns={'Filename': 'piece_name'})
                    
                    # Remove the '.dxf' extension and strip leading/trailing whitespace
                    combined_df['piece_name'] = combined_df['piece_name'].str.replace(r'\.dxf$', '', regex=True).str.strip()
                    
                    # Remove only trailing spaces and numbers, ensuring valid names remain intact
                    combined_df['piece_name'] = combined_df['piece_name'].str.replace(r'\s+\d*$', '', regex=True).str.strip()

                combined_df_list.append(combined_df)
        
        # Combine all the DataFrames from different files into one
        final_combined_df = pd.concat(combined_df_list, ignore_index=True) if combined_df_list else pd.DataFrame()
        final_combined_df.columns = final_combined_df.columns.str.lower()
        
        # Drop DXF Native entities and other not needed for further processing
        columns_to_drop = ['color', 'type', 'layer', 'color', 'point_x', 'point_y', 'height', 
                           'style', 'text', 'block', 'line_start_x', 'line_start_y', 'line_end_x', 'line_end_y',
                           'vertex_index', 'point label', 'vertex label']
        
        final_combined_df.drop(columns=columns_to_drop, inplace=True)
        self.combined_entities_joined_df = final_combined_df
    
    def merge_tables(self):
        self.alteration_joined_df['piece_name'] = self.alteration_joined_df['piece_name'].fillna('')
        self.combined_entities_joined_df['piece_name'] = self.combined_entities_joined_df['piece_name'].fillna('')

        self.merged_df = pd.merge(
            self.alteration_joined_df, 
            self.combined_entities_joined_df, 
            on=['piece_name', 'mtm points'], 
            how='right'
        )

        #self.merged_df.to_csv("data/staging/all_combined_tables.csv")

    def save_table_csv_by_alteration_rule(self, output_filename_prefix="combined_table"):
        
        output_table_path = self.output_table_path

        # Ensure the output directory exists
        os.makedirs(output_table_path, exist_ok=True)

        self.merged_df['piece_name'] = self.merged_df['piece_name'].str.replace(".dxf", "", regex=False)

        # Group the DataFrame by 'alteration_rule'
        grouped = self.merged_df.groupby('alteration_rule')
        
        for alteration_rule, group_df in grouped:
            # Convert all column headers to lowercase
            group_df.columns = group_df.columns.str.lower()
                        
            # Create a sanitized file name for each alteration_rule
            safe_alteration_rule = str(alteration_rule).replace(" ", "_").replace("/", "_")  # Ensure no illegal filename characters
            output_file_path = os.path.join(output_table_path, f"{output_filename_prefix}_{safe_alteration_rule}.csv")
            
            # Save the group as a CSV file
            group_df.to_csv(output_file_path, index=False)

            print(f"CSV files saved to {output_table_path}")

    def save_table_csv_by_piece_name(self, output_filename_prefix="combined_table"):

        output_table_path = self.output_table_path_by_piece

        # Ensure the output directory exists
        os.makedirs(output_table_path, exist_ok=True)

        self.merged_df['piece_name'] = self.merged_df['piece_name'].str.replace(".dxf", "", regex=False)

        # Group the DataFrame by 'alteration_rule'
        grouped = self.merged_df.groupby('piece_name')

        for piece_name, group_df in grouped:
            # Convert all column headers to lowercase
            group_df.columns = group_df.columns.str.lower()

            safe_piece_name = str(piece_name).replace(" ", "_").replace("/", "_")  # Ensure no illegal filename characters
            output_file_path = os.path.join(output_table_path, f"{output_filename_prefix}_{safe_piece_name}.csv")
            
            # Save the group as a CSV file
            group_df.to_csv(output_file_path, index=False)

            print(f"CSV files saved to {output_table_path}")        

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

                print(f"Updated DataFrame saved to {load_path}")


    def create_vertices_df(self):
        # Base directory where all piece_name directories will be created
        base_dir = os.path.join(self.output_dir, "vertices")
        os.makedirs(base_dir, exist_ok=True)

        # Get the unique piece names
        unique_piece_names = self.combined_entities_joined_df['piece_name'].unique()

        # Loop through each unique piece_name
        for piece_name in unique_piece_names:
            # Remove ".dxf" extension from piece_name
            sanitized_piece_name = piece_name.replace(".dxf", "")

            # Filter rows where piece_name matches the current piece_name
            matching_rows = self.combined_entities_joined_df[self.combined_entities_joined_df['piece_name'] == piece_name]
            
            # Extract unique vertices for this piece_name
            unique_vertices = matching_rows['vertices'].unique()

            # Create a DataFrame with piece_name and its corresponding unique vertices
            vertices_df = pd.DataFrame({'piece_name': sanitized_piece_name, 'vertices': unique_vertices})
            
            # Drop rows where 'vertices' is empty or NaN
            vertices_df = vertices_df.dropna(subset=['vertices'])
            
            # Save the DataFrame to a CSV file in the corresponding directory
            output_path = os.path.join(base_dir, f'vertices_{sanitized_piece_name}.csv')
            vertices_df.to_csv(output_path, index=False)
            
            # Print the DataFrame (optional)
            #print(f'Saved {output_path}')

if __name__ == "__main__":
    alteration_filepath = "data/input/mtm_points.xlsx"
    combined_entities_folder = "data/input/mtm-combined-entities/"
    create_table = CreateTable(alteration_filepath, combined_entities_folder)
    
    # Process the sheets and get the combined DataFrame
    create_table.process_table()
    create_table.process_combined_entities()

    # Create Vertices DF
    create_table.create_vertices_df()

    # Join tables
    create_table.merge_tables()
    
    # Save the combined DataFrame as CSV files 
    create_table.save_table_csv_by_alteration_rule()
    create_table.save_table_csv_by_piece_name()
    create_table.add_other_mtm_points()

    # DEBUG
    #print("\n# Debug: All Unique Processed Piece Names. A common error is if they do not match the Actual piece name. \nCheck for extra spaces, incomplete names or unwanted extensions (e.g. .dxf)\n")
    #print(f"Processed Piece Names: {create_table.combined_entities_joined_df['piece_name'].unique()}\n")
    #print(f"Combined Entities (They should match): {create_table.alteration_joined_df['piece_name'].unique()}")
    #print("\nNOTE: If there are pieces in the Processed Piece Names that do not exist in the Combined entities or vice versa, then that table needs to be created")
