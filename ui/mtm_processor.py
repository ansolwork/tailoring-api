import pandas as pd
from matplotlib import pyplot as plt
import os
import zipfile
import ast

class MTMProcessor:
    def __init__(self, scale_factor=5):
        self.scale_factor = scale_factor
        self.coordinates_tables = {}
        self.filtered_tables = {}
        self.merged_tables = {}
        self.modified_tables = {}

    ### Data Loading and Filtering Functions ###

    def load_coordinates_tables(self, path):
        """Load Excel files containing coordinate tables into a dictionary of DataFrames."""
        if os.path.isfile(path):
            self._load_single_file(path)
        elif os.path.isdir(path):
            self._load_directory(path)
        else:
            print(f"{path} is not a valid file or directory. Please check the path.")
    
    def _load_single_file(self, file_path):
        """Helper to load a single Excel file."""
        print(f"Loading single file: {file_path}")
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip().str.lower()
        self.coordinates_tables[os.path.basename(file_path)] = df

    def _load_directory(self, dir_path):
        """Helper to load all Excel files from a directory."""
        print(f"Loading directory: {dir_path}")
        excel_files = [file for file in os.listdir(dir_path) if file.endswith('.xlsx')]
        for file in excel_files:
            df = pd.read_excel(os.path.join(dir_path, file))
            df.columns = df.columns.str.strip().str.lower()
            self.coordinates_tables[file] = df

    def remove_nan_mtm_points(self):
        """Filter out rows where 'MTM Points' is NaN and store filtered tables."""
        for file, df in self.coordinates_tables.items():
            if 'mtm points' in df.columns:
                filtered_df = df[df['mtm points'].notna()].copy()
                filtered_df['mtm points'] = filtered_df['mtm points'].astype(int)
                self.filtered_tables[file] = filtered_df
            else:
                print(f"Column 'mtm points' not found in {file}")

    def display_filtered_coordinates_tables(self):
        """Display the filtered coordinate tables."""
        for file, df in self.filtered_tables.items():
            print(f"Contents of {file}:")
            print(df)
            print("\n")

    ### Visualization Functions ###
    def plot_points(self, output_dir="ui/static/plots/", rename=None):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate a plot for the first file in coordinates_tables
        for file, df in self.coordinates_tables.items():
            plt.figure(figsize=(20, 16))

            self.plot_vertices(df)
            self.overlay_mtm_points(df)

            # Save the plot as a PNG file to the local `static/plots` directory
            if rename:
                plot_filename = rename
                plot_filepath = os.path.join(output_dir, f"{plot_filename}_plot.png")
                self.configure_plot(rename)
            else:
                plot_filename = os.path.splitext(file)[0]
                plot_filepath = os.path.join(output_dir, f"{plot_filename}_plot.png")
                self.configure_plot(file)
        
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.savefig(plot_filepath)
            plt.close()

            # Return the plot file path (for local use) and the filename
            return plot_filename, plot_filepath

    def plot_vertices(self, df):
        """Plot vertices from the dataframe."""
        vertices = df['vertices'].dropna().unique()
        for vert_str in vertices:
            try:
                vertices_list = ast.literal_eval(vert_str)
                x_coords = [v[0] for v in vertices_list]
                y_coords = [v[1] for v in vertices_list]
                plt.plot(x_coords, y_coords, c='blue', marker='o', linestyle='-', linewidth=1, markersize=4)
            except (ValueError, SyntaxError):
                print("Invalid format in 'vertices' column")

    def overlay_mtm_points(self, df):
        """Overlay MTM points on the plot."""
        df_non_nan = df[df['mtm points'].notna()].copy()
        df_non_nan['mtm points'] = df_non_nan['mtm points'].astype(int)
        for _, row in df_non_nan.iterrows():
            plt.text(row['pl_point_x'], row['pl_point_y'], f"{int(row['mtm points'])}", fontsize=9, color='red', ha='right')
            plt.plot(row['pl_point_x'], row['pl_point_y'], 'ro')

    def configure_plot(self, file):
        """Set plot labels, title, and grid."""
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(f'Plot of Points for {file}', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
