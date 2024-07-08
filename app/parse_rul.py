import pandas as pd
import os

# Define the directory containing RUL files
input_directory = "../data/02-07-2024-dxf-files/"
output_directory = "../data/output_tables/"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
files = os.listdir(input_directory)

# Filter out the RUL files
rul_files = [file for file in files if file.endswith(".RUL")]

# Process each RUL file
for filename in rul_files:
    file_path = os.path.join(input_directory, filename)
    
    try:
        # Read the file into a DataFrame (modify 'sep' parameter based on your file's delimiter)
        df = pd.read_csv(file_path)
        
        # Display the DataFrame
        print(f"Contents of {filename}:")
        print(df)
        
        # Save the DataFrame to an Excel file
        excel_file_path = os.path.join(output_directory, filename.replace('.RUL', '_RUL.xlsx'))
        df.to_excel(excel_file_path, index=False)
        
        print(f"Data from {filename} saved to {excel_file_path}")
    except Exception as e:
        print(f"Failed to process {filename}: {e}")
