import os
import pandas as pd
from matplotlib import pyplot as plt
import zipfile
import ast

class ApplyAlteration:
    def __init__(self, alteration_rules_file=None, scale_factor=5):
        if alteration_rules_file:
            self.alteration_rules_df = pd.read_excel(alteration_rules_file)
        self.scale_factor = scale_factor
        self.coordinates_tables = {}
        self.filtered_tables = {}

    def load_coordinates_tables(self, path):
        if os.path.isfile(path):
            print(f"Loading single file: {path}")
            self.coordinates_tables[os.path.basename(path)] = pd.read_excel(path)
        elif os.path.isdir(path):
            print(f"Loading directory: {path}")
            # List all Excel files in the directory
            excel_files = [file for file in os.listdir(path) if file.endswith('.xlsx')]
            # Load all the Excel files into a dictionary of DataFrames
            self.coordinates_tables = {file: pd.read_excel(os.path.join(path, file)) for file in excel_files}
        else:
            print(f"{path} is not a valid file or directory. Please check the path.")
            return

    def remove_nan_mtm_points(self):
        for file, df in self.coordinates_tables.items():
            # Strip any leading/trailing whitespace from the column names and make them case-insensitive
            df.columns = df.columns.str.strip().str.lower()
            print(f"Columns in {file}: {df.columns.tolist()}")
            # Filter rows with non-NaN MTM Points
            if 'mtm points' in df.columns:
                filtered_df = df[df['mtm points'].notna()].copy()
                # Convert MTM Points to integers
                filtered_df['mtm points'] = filtered_df['mtm points'].astype(int)
                self.filtered_tables[file] = filtered_df
            else:
                print(f"Column 'mtm points' not found in {file}")

    def display_filtered_coordinates_tables(self):
        for file, df in self.filtered_tables.items():
            print(f"Contents of {file}:")
            print(df)
            print("\n")

    def plot_points(self, output_dir="../data/output_graphs/mtm_labels/"):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        output_files = []

        for file, df in self.coordinates_tables.items():
            df.columns = df.columns.str.strip().str.lower()
            print(f"Full DataFrame columns for {file}: {df.columns.tolist()}")

            if 'vertices' not in df.columns:
                print(f"Required column 'vertices' not found in {file}. Skipping this file.")
                continue

            plt.figure(figsize=(12, 9))

            # Plot the vertices
            vertices = df['vertices'].dropna().unique()
            for vert_str in vertices:
                try:
                    vertices_list = ast.literal_eval(vert_str)
                    x_coords = [v[0] for v in vertices_list]
                    y_coords = [v[1] for v in vertices_list]
                    plt.plot(x_coords, y_coords, c='blue', marker='o', linestyle='-', linewidth=1, markersize=4)
                except (ValueError, SyntaxError):
                    print(f"Invalid format in 'vertices' column for file: {file}")

            # Overlay MTM Points
            df_non_nan = df[df['mtm points'].notna()].copy()
            df_non_nan['mtm points'] = df_non_nan['mtm points'].astype(int)
            for _, row in df_non_nan.iterrows():
                plt.text(row['pl_point_x'], row['pl_point_y'], f"{int(row['mtm points'])}", fontsize=9, color='red', ha='right')
                plt.plot(row['pl_point_x'], row['pl_point_y'], 'ro')  # Highlight point in red

            plt.xlabel('X Coordinate', fontsize=12)
            plt.ylabel('Y Coordinate', fontsize=12)
            plt.title(f'Plot of Points for {file}', fontsize=14)
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.png")
            print(f"Saving plot to {output_path}")
            plt.savefig(output_path)
            plt.close()

            if os.path.exists(output_path):
                print(f"Plot saved successfully: {output_path}")
            else:
                print(f"Failed to save plot: {output_path}")

            output_files.append(output_path)
        
        if len(output_files) > 1:
            zip_path = os.path.join(output_dir, "mtm_points_graphs.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in output_files:
                    zipf.write(file, os.path.basename(file))
            print(f"Saved zip file to {zip_path}")
            return zip_path
        else:
            if output_files and os.path.exists(output_files[0]):
                return output_files[0]
            else:
                print("No output files were generated.")
                return None

if __name__ == "__main__":
    alteration_rules_file = "../data/output_tables/alteration_rules.xlsx"
    # coordinates_table_path = "../data/mtm-points-coordinates/"  # Could be a file or a directory
    coordinates_table_path = "../data/mtm-test-points"

    apply_alteration = ApplyAlteration(alteration_rules_file)
    apply_alteration.load_coordinates_tables(coordinates_table_path)
    apply_alteration.remove_nan_mtm_points()
    apply_alteration.display_filtered_coordinates_tables()
    output_path = apply_alteration.plot_points()
    print(f"Graphs saved in: {output_path}")

    # Print filtered DataFrames
    print("\nFiltered DataFrames with non-NaN MTM Points:")
    for file, df in apply_alteration.filtered_tables.items():
        print(f"\nContents of {file}:")
        print(df)
        print("\n")
