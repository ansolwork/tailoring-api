import pandas as pd
import os

class CreateTable:
    def __init__(self, input_filepath):
        self.input_filepath = input_filepath
        self.df_dict = self.load_table()  # Loading all sheets as a dictionary of DataFrames

    def load_table(self):
        # Load all sheets into a dictionary of DataFrames
        df_dict = pd.read_excel(self.input_filepath, sheet_name=None)
        return df_dict
    
    def process_table(self, sheets):
        # Filter the sheets dictionary to only include the specified sheets
        selected_sheets = {name: self.df_dict[name] for name in sheets if name in self.df_dict}
        
        # Concatenate the DataFrames into a single DataFrame
        big_df = pd.concat(selected_sheets.values(), ignore_index=True)
        
        return big_df
    
    def save_table(self, big_df, output_table_path="../data/output_tables", output_filename="combined_table.xlsx"):
        # Ensure the output directory exists
        os.makedirs(output_table_path, exist_ok=True)
        
        # Create the full file path
        output_file_path = os.path.join(output_table_path, output_filename)
        
        # Save the DataFrame to an Excel file
        big_df.to_excel(output_file_path, index=False)
        
        print(f"Table saved to {output_file_path}")

if __name__ == "__main__":
    input_filepath = "../data/input/MTM-POINTS.xlsx"
    create_table = CreateTable(input_filepath)

    # Included sheets
    sheet_list = ["SHIRT-FRONT"]
    
    # Process the sheets and get the combined DataFrame
    combined_df = create_table.process_table(sheet_list)
    
    # Save the combined DataFrame as an Excel file
    create_table.save_table(combined_df)
