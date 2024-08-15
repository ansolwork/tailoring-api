import pandas as pd
import os

class CreateTable:
    def __init__(self, alteration_filepath, combined_entities_folder):
        self.alteration_filepath = alteration_filepath
        self.combined_entities_folder = combined_entities_folder
        self.df_dict = self.load_table()  # Loading all sheets as a dictionary of DataFrames
        self.alteration_joined_df = pd.DataFrame()
        self.combined_entities_joined_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.output_dir = "../data/output_tables"
        self.output_table_path = "../data/output_tables/combined_alteration_tables"

    def load_table(self):
        # Load all sheets into a dictionary of DataFrames
        df_dict = pd.read_excel(self.alteration_filepath, sheet_name=None)
        return df_dict
    
    @staticmethod
    def load_csv(filepath):
        df_csv = pd.read_csv(filepath)
        return df_csv
    
    def process_table(self, sheets):
        # Filter the sheets dictionary to only include the specified sheets
        selected_sheets = {name: self.df_dict[name] for name in sheets if name in self.df_dict}
        
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
                print(f"Processing file: {filepath}")
                
                # Load the current Excel file
                combined_df = pd.read_excel(filepath)
                
                # Rename the 'Filename' column to 'piece_name' and remove the '.dxf' extension
                if 'Filename' in combined_df.columns:
                    combined_df = combined_df.rename(columns={'Filename': 'piece_name'})
                    combined_df['piece_name'] = combined_df['piece_name'].str.replace(r'\d+\.dxf$', '', regex=True).str.strip()
                
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
        # This will keep only the pieces we have combined entities for
        self.merged_df = pd.merge(self.alteration_joined_df, self.combined_entities_joined_df, on=['piece_name', 'mtm points'], how='right')
    
    def save_table_csv(self, output_filename_prefix="combined_table"):
        
        output_table_path = self.output_table_path

        # Ensure the output directory exists
        os.makedirs(output_table_path, exist_ok=True)
        
        # Group the DataFrame by 'alteration_rule'
        grouped = self.merged_df.groupby('alteration_rule')
        
        for alteration_rule, group_df in grouped:
            # Convert all column headers to lowercase
            group_df.columns = group_df.columns.str.lower()
            
            #print(f"Piece Names:\n{self.merged_df['piece_name'].unique()}")
            
            # Create a sanitized file name for each alteration_rule
            safe_alteration_rule = str(alteration_rule).replace(" ", "_").replace("/", "_")  # Ensure no illegal filename characters
            output_file_path = os.path.join(output_table_path, f"{output_filename_prefix}_{safe_alteration_rule}.csv")
            
            # Save the group as a CSV file
            group_df.to_csv(output_file_path, index=False)
        
        print(f"CSV files saved to {output_table_path}")

    def add_other_mtm_points(self):
        for filename in os.listdir(self.output_table_path):
            if filename.endswith(".csv"):
                # Load the CSV file
                load_path = os.path.join(self.output_table_path, filename)
                df = self.load_csv(load_path)

                # Do operation here
                join_col = df['piece_name'].iloc[0]  # Use .iloc[0] to safely get the first element
                matching_rows = self.combined_entities_joined_df[self.combined_entities_joined_df['piece_name'] == join_col]

                # Concatenate the DataFrames
                df = pd.concat([df, matching_rows], axis=0)
                
                # Ensure 'vertices' is dropped
                df.drop(columns=['vertices'], inplace=True)

                # Save the updated DataFrame back to CSV
                df.to_csv(load_path, index=False)

    def create_vertices_df(self):
        # Base directory where all piece_name directories will be created
        base_dir = os.path.join(self.output_dir, "vertices")
        os.makedirs(base_dir, exist_ok=True)

        # Get the unique piece names
        unique_piece_names = self.combined_entities_joined_df['piece_name'].unique()

        # Loop through each unique piece_name
        for piece_name in unique_piece_names:
            # Filter rows where piece_name matches the current piece_name
            matching_rows = self.combined_entities_joined_df[self.combined_entities_joined_df['piece_name'] == piece_name]
            
            # Extract unique vertices for this piece_name
            unique_vertices = matching_rows['vertices'].unique()
            
            # Create a DataFrame with piece_name and its corresponding unique vertices
            vertices_df = pd.DataFrame({'piece_name': piece_name, 'vertices': unique_vertices})
            
            # Drop rows where 'vertices' is empty or NaN
            vertices_df = vertices_df.dropna(subset=['vertices'])
            
            # Save the DataFrame to a CSV file in the corresponding directory
            output_path = os.path.join(base_dir, f'{piece_name}_vertices.csv')
            vertices_df.to_csv(output_path, index=False)
            
            # Print the DataFrame (optional)
            print(f'Saved {output_path}')


if __name__ == "__main__":
    alteration_filepath = "../data/input/MTM-POINTS.xlsx"
    combined_entities_folder = "../data/input/mtm-combined-entities/"
    create_table = CreateTable(alteration_filepath, combined_entities_folder)

    # Included sheets
    sheet_list = ["SHIRT-FRONT", "SHIRT-BACK", "SHIRT-YOKE", "SHIRT-SLEEVE", "SHIRT-COLLAR", "SHIRT-COLLAR-BAND", "SHIRT-CUFF"]
    
    # Process the sheets and get the combined DataFrame
    create_table.process_table(sheet_list)
    create_table.process_combined_entities()

    # Create Vertices DF
    create_table.create_vertices_df()

    # Join tables
    create_table.merge_tables()
    
    # Save the combined DataFrame as CSV files 
    create_table.save_table_csv()
    create_table.add_other_mtm_points()
