import os
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

class ExcelFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_frame = None
        self.filtered_data_frame = None

    def read_file(self):
        if not os.path.exists(self.file_path):
            print(f"File {self.file_path} does not exist.")
            return None
        
        try:
            self.data_frame = pd.read_excel(self.file_path)
            print("File read successfully")
        except Exception as e:
            print(f"Error reading file: {e}")
            self.data_frame = None
        return self.data_frame

    def display_contents(self):
        if self.data_frame is not None:
            print("Displaying first few rows of the data frame:")
            print(self.data_frame.head())
        else:
            print("Data frame is empty. Please read the file first.")

    def filter_by_column_value(self, column_name, values):
        if self.data_frame is not None:
            if column_name in self.data_frame.columns:
                self.filtered_data_frame = self.data_frame[self.data_frame[column_name].isin(values)]
                if not self.filtered_data_frame.empty:
                    print(f"Filtered data (where {column_name} in {values}):")
                    print(self.filtered_data_frame.head())
                else:
                    print(f"No rows found with {column_name} in {values}")
            else:
                print(f"Column {column_name} not found in the data frame.")
        else:
            print("Data frame is empty. Please read the file first.")

    def display_filtered_contents(self):
        if self.filtered_data_frame is not None and not self.filtered_data_frame.empty:
            print("Displaying filtered data frame:")
            print(self.filtered_data_frame.head())
        else:
            print("Filtered data frame is empty. Please apply a filter first.")

    def draw_lines_and_polylines(self, output_directory):
        if self.filtered_data_frame is not None and not self.filtered_data_frame.empty:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            grouped = self.filtered_data_frame.groupby('Filename')
            for name, group in grouped:
                plt.figure(figsize=(16, 10))  # Increase figure size

                # Draw lines
                lines = group[group['Type'] == 'LINE']
                for _, row in lines.iterrows():
                    if pd.notna(row['Line_Start_X']) and pd.notna(row['Line_End_X']) and pd.notna(row['Line_Start_Y']) and pd.notna(row['Line_End_Y']):
                        plt.plot([row['Line_Start_X'], row['Line_End_X']], [row['Line_Start_Y'], row['Line_End_Y']], marker='o', linewidth=0.5, markersize=5)
                        plt.text(row['Line_Start_X'], row['Line_Start_Y'], f"({row['Line_Start_X']}, {row['Line_Start_Y']})", fontsize=10, ha='right', va='bottom')
                        plt.text(row['Line_End_X'], row['Line_End_Y'], f"({row['Line_End_X']}, {row['Line_End_Y']})", fontsize=10, ha='right', va='bottom')

                # Draw polylines
                polylines = group[group['Type'].isin(['POLYLINE', 'LWPOLYLINE'])]
                unique_points = polylines.drop_duplicates(subset=['PL_POINT_X', 'PL_POINT_Y'])
                for vertex_label in polylines['Vertex Label'].unique():
                    vertex_group = polylines[polylines['Vertex Label'] == vertex_label]
                    xs = vertex_group['PL_POINT_X'].tolist()
                    ys = vertex_group['PL_POINT_Y'].tolist()
                    plt.plot(xs, ys, marker='o', linewidth=0.5, markersize=5)

                # Annotate unique points
                for x, y, point_label in zip(unique_points['PL_POINT_X'], unique_points['PL_POINT_Y'], unique_points['Point Label']):
                    plt.text(x, y, f'{point_label}', fontsize=10, ha='right', va='bottom')

                plt.title(f'Polyline Plot for {name}', fontsize=16)
                plt.xlabel('X Coordinate', fontsize=14)
                plt.ylabel('Y Coordinate', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.grid(True)
                plt.tight_layout()  # Improve layout

                output_path = os.path.join(output_directory, f"polyline_plot_{name}.png")
                plt.savefig(output_path, dpi=300)  # Increase DPI for better quality
                plt.close()

                print(f"Polyline plot for {name} saved to {output_path}")

                # Create and save the table of coordinates
                all_xs = unique_points['PL_POINT_X'].dropna().tolist()
                all_ys = unique_points['PL_POINT_Y'].dropna().tolist()
                point_labels = unique_points['Point Label'].dropna().tolist()
                coordinates_df = pd.DataFrame({'Point': point_labels, 'X Coordinate': all_xs, 'Y Coordinate': all_ys})
                
                coordinates_table_path_csv = os.path.join(output_directory, f"coordinates_table_{name}.csv")
                coordinates_df.to_csv(coordinates_table_path_csv, index=False)

                coordinates_table_path_excel = os.path.join(output_directory, f"coordinates_table_{name}.xlsx")
                coordinates_df.to_excel(coordinates_table_path_excel, index=False)

                print(f"Coordinates table for {name} saved to {coordinates_table_path_csv} and {coordinates_table_path_excel}")
        else:
            print("Filtered data frame is empty. Please apply a filter first.")

if __name__ == "__main__":
    table_directory = "../data/output_tables/"
    filename = "_combined_entities.xlsx"
    output_dir = "../data/output_graphs/dxf_plots/"  # Specify the output directory

    filepath = os.path.join(table_directory, filename)
    
    file_reader = ExcelFileReader(filepath)
    df = file_reader.read_file()
    
    if df is not None:
        file_reader.display_contents()

        column_name = 'Type'
        values = ['LINE', 'POLYLINE', 'LWPOLYLINE']  # List of entity types to filter by
        file_reader.filter_by_column_value(column_name, values)

        file_reader.display_filtered_contents()
        file_reader.draw_lines_and_polylines(output_dir)
