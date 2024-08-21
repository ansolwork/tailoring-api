import os
import pandas as pd
from matplotlib import pyplot as plt
import zipfile
import ast

class ApplyAlteration:
    def __init__(self, alteration_rules_file=None, scale_factor=5):
        if alteration_rules_file:
            self.alteration_rules_df = pd.read_excel(alteration_rules_file)
            self.alteration_rules_df.columns = self.alteration_rules_df.columns.str.strip().str.lower()  # Ensure columns are lowercase
        self.scale_factor = scale_factor
        self.coordinates_tables = {}
        self.filtered_tables = {}
        self.merged_tables = {}
        self.modified_tables = {}

    def load_coordinates_tables(self, path):
        if os.path.isfile(path):
            print(f"Loading single file: {path}")
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip().str.lower()
            self.coordinates_tables[os.path.basename(path)] = df
        elif os.path.isdir(path):
            print(f"Loading directory: {path}")
            # List all Excel files in the directory
            excel_files = [file for file in os.listdir(path) if file.endswith('.xlsx')]
            # Load all the Excel files into a dictionary of DataFrames
            for file in excel_files:
                df = pd.read_excel(os.path.join(path, file))
                df.columns = df.columns.str.strip().str.lower()
                self.coordinates_tables[file] = df
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
    
    def merge_alteration_rules(self):
        # Ensure alteration_rules_df is loaded
        if self.alteration_rules_df is None:
            print("Alteration rules DataFrame not loaded.")
            return

        print("Merging alteration rules:")
        print(self.alteration_rules_df)

        # Prepare to store the merged DataFrames
        merged_tables = {}

        # Iterate through the coordinate tables and perform the merge
        for file, coordinates_df in self.coordinates_tables.items():
            # Ensure both DataFrames have matching column names for the merge
            coordinates_df.columns = coordinates_df.columns.str.strip().str.lower()
            self.alteration_rules_df.columns = self.alteration_rules_df.columns.str.strip().str.lower()

            # Merge on First PT
            merged_first_pt = pd.merge(coordinates_df, self.alteration_rules_df, left_on='mtm points', right_on='first pt', how='inner')

            # Merge on Second PT
            merged_second_pt = pd.merge(coordinates_df, self.alteration_rules_df, left_on='mtm points', right_on='second pt', how='inner')

            # Combine the results, removing duplicates
            merged_df = pd.concat([merged_first_pt, merged_second_pt]).drop_duplicates().reset_index(drop=True)

            # Store the merged DataFrame
            merged_tables[file] = merged_df

            print(f"Merged DataFrame for {file}:")
            print(merged_df)
            print("\n")

        # Optionally, you can save or further process the merged DataFrames
        self.merged_tables = merged_tables

        # Display the merged tables
        self.display_merged_tables()

    def display_merged_tables(self):
        for file, df in self.merged_tables.items():
            print(f"Contents of merged {file}:")
            print(df)
            print("\n")

    def export_to_excel(self, output_file):
        with pd.ExcelWriter(output_file) as writer:
            for file, df in self.merged_tables.items():
                # Using the base file name (without extension) as the sheet name
                sheet_name = os.path.splitext(file)[0]
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Merged tables saved to {output_file}")

    def apply_alteration_rules(self):
        # Ensure the merged tables are available
        if not self.merged_tables:
            print("Merged tables are not available. Please run merge_alteration_rules() first.")
            return

        # Prepare to store the modified DataFrames
        modified_tables = {}

        # Iterate through the merged tables and apply the movements
        for file, df in self.merged_tables.items():
            # Ensure numeric conversion
            df['pl_point_x'] = pd.to_numeric(df['pl_point_x'], errors='coerce').fillna(0)
            df['pl_point_y'] = pd.to_numeric(df['pl_point_y'], errors='coerce').fillna(0)
            
            # Remove percentage signs and convert to numeric
            df['movement x'] = df['movement x'].str.replace('%', '').astype(float)
            df['movement y'] = df['movement y'].str.replace('%', '').astype(float)
            
            print(df['movement y'])  # Debug print to see the values after conversion

            # Apply the percentage movements
            df['pl_point_x_modified'] = df['pl_point_x'] * (1 + df['movement x'] / 100.0)
            df['pl_point_y_modified'] = df['pl_point_y'] * (1 + df['movement y'] / 100.0)
            modified_tables[file] = df

            print(f"Modified DataFrame for {file}:")
            print(df)
            print("\n")

        self.modified_tables = modified_tables

        # Save the modified tables to an Excel file
        self.export_modified_to_excel("../data/output_tables/modified_tables.xlsx")

    def export_modified_to_excel(self, output_file):
        with pd.ExcelWriter(output_file) as writer:
            for file, df in self.modified_tables.items():
                # Using the base file name (without extension) as the sheet name
                sheet_name = os.path.splitext(file)[0]
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Modified tables saved to {output_file}")

    def replace_with_modified_points(self):
        for file, df in self.coordinates_tables.items():
            if file in self.modified_tables:
                modified_df = self.modified_tables[file]
                for idx, row in modified_df.iterrows():
                    point_index = row['mtm points']
                    df.loc[df['mtm points'] == point_index, 'pl_point_x'] = row['pl_point_x_modified']
                    df.loc[df['mtm points'] == point_index, 'pl_point_y'] = row['pl_point_y_modified']
                self.coordinates_tables[file] = df

    def plot_full_graph_with_modifications(self, output_dir="../data/output_graphs/full_graph_with_modifications/"):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for file, df in self.coordinates_tables.items():
            plt.figure(figsize=(12, 9))

            # Plot the vertices
            vertices = df['vertices'].dropna().unique()
            for vert_str in vertices:
                try:
                    vertices_list = ast.literal_eval(vert_str)
                    x_coords = [v[0] for v in vertices_list]
                    y_coords = [v[1] for v in vertices_list]
                    plt.plot(x_coords, y_coords, 'bo-', label='Original Vertices', linewidth=1, markersize=4)
                except (ValueError, SyntaxError):
                    print(f"Invalid format in 'vertices' column for file: {file}")

            # Overlay MTM Points with modifications
            df_non_nan = df[df['mtm points'].notna()].copy()
            for _, row in df_non_nan.iterrows():
                plt.text(row['pl_point_x'], row['pl_point_y'], f"{int(row['mtm points'])}", fontsize=9, color='red', ha='right')
                plt.plot(row['pl_point_x'], row['pl_point_y'], 'ro')  # Highlight modified point in red

            plt.xlabel('X Coordinate', fontsize=12)
            plt.ylabel('Y Coordinate', fontsize=12)
            plt.title(f'Full Graph with Modifications for {file}', fontsize=14)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_full_modified.png")
            print(f"Saving plot to {output_path}")
            plt.savefig(output_path)
            plt.close()

            if os.path.exists(output_path):
                print(f"Plot saved successfully: {output_path}")
            else:
                print(f"Failed to save plot: {output_path}")

    def get_alteration_rule(self, rule):
        if self.alteration_rules_df is not None:
            print("Alteration Rules DataFrame columns:")
            print(self.alteration_rules_df.columns)
            rule_subset = self.alteration_rules_df[self.alteration_rules_df['rule name'].str.upper() == rule.upper()]
            return rule_subset
        else:
            print("Alteration rules DataFrame not loaded.")
            return None

    def merge_with_rule_subset(self, rule):
        rule_subset = self.get_alteration_rule(rule)
        if rule_subset is not None:
            for file, df in self.coordinates_tables.items():
                # Merge the coordinate table with the rule subset on 'mtm points' and 'first pt'
                merged_df_first = pd.merge(df, rule_subset, left_on='mtm points', right_on='first pt', how='left')
                
                # Merge the coordinate table with the rule subset on 'mtm points' and 'second pt'
                merged_df_second = pd.merge(df, rule_subset, left_on='mtm points', right_on='second pt', how='left')
                
                # Combine the results, removing duplicates
                merged_df = pd.concat([merged_df_first, merged_df_second]).drop_duplicates().reset_index(drop=True)
                
                self.merged_tables[file] = merged_df

                print(f"Merged DataFrame for {file} with rule '{rule}':")
                print(merged_df)
                print("\n")
        else:
            print("No rule subset found for the specified rule.")

    def export_merged_with_rule_subset_to_excel(self, output_file):
        with pd.ExcelWriter(output_file) as writer:
            for file, df in self.merged_tables.items():
                # Using the base file name (without extension) as the sheet name
                sheet_name = os.path.splitext(file)[0]
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Merged tables with rule subset saved to {output_file}")

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
    
    apply_alteration.merge_alteration_rules()
    apply_alteration.display_merged_tables()

    # Save merged tables to an Excel file
    output_excel_file = "../data/output_tables/merged_tables.xlsx"
    apply_alteration.export_to_excel(output_excel_file)

    # Apply alteration rules
    apply_alteration.apply_alteration_rules()

    # Save modified tables to an Excel file
    output_modified_excel_file = "../data/output_tables/modified_tables.xlsx"
    apply_alteration.export_modified_to_excel(output_modified_excel_file)

    # Replace original points with modified points in the coordinates table
    apply_alteration.replace_with_modified_points()

    # Plot the full graph with modifications
    apply_alteration.plot_full_graph_with_modifications()

    # Merge with specific alteration rule
    apply_alteration.load_coordinates_tables(coordinates_table_path)
    apply_alteration.merge_with_rule_subset("4-WAIST")

    # Save the merged DataFrame with rule subset to an Excel file
    output_merged_with_rule_subset_file = "../data/output_tables/merged_with_rule_subset.xlsx"
    apply_alteration.export_merged_with_rule_subset_to_excel(output_merged_with_rule_subset_file)
