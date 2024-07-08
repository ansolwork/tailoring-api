import pandas as pd
import os
from matplotlib import pyplot as plt
import ast

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

    def filter_by_column_value(self, column_name, value):
        if self.data_frame is not None:
            if column_name in self.data_frame.columns:
                self.filtered_data_frame = self.data_frame[self.data_frame[column_name] == value]
                if not self.filtered_data_frame.empty:
                    print(f"Filtered data (where {column_name} == {value}):")
                    print(self.filtered_data_frame)
                else:
                    print(f"No rows found with {column_name} == {value}")
            else:
                print(f"Column {column_name} not found in the data frame.")
        else:
            print("Data frame is empty. Please read the file first.")

    def display_filtered_contents(self):
        if self.filtered_data_frame is not None and not self.filtered_data_frame.empty:
            print("Displaying filtered data frame:")
            print(self.filtered_data_frame.to_string(index=False))
        else:
            print("Filtered data frame is empty. Please apply a filter first.")

    def draw_lines(self, output_directory):
        if self.filtered_data_frame is not None and not self.filtered_data_frame.empty:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            grouped = self.filtered_data_frame.groupby('Filename')
            for name, group in grouped:
                plt.figure(figsize=(10, 6))
                for _, row in group.iterrows():
                    plt.plot([row['Start_X'], row['End_X']], [row['Start_Y'], row['End_Y']], marker='o')

                plt.title(f'Line Plot for {name}')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.grid(True)

                output_path = os.path.join(output_directory, f"line_plot_{name}.png")
                plt.savefig(output_path)
                plt.show()

                print(f"Line plot for {name} saved to {output_path}")
        else:
            print("Filtered data frame is empty. Please apply a filter first.")

    def draw_polylines(self, output_directory, entity_value):
        if self.filtered_data_frame is not None and not self.filtered_data_frame.empty:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            grouped = self.filtered_data_frame.groupby('Filename')
            for name, group in grouped:
                plt.figure(figsize=(10, 6))
                for _, row in group.iterrows():
                    vertices = ast.literal_eval(row['Vertices'])
                    xs, ys = zip(*vertices)
                    plt.plot(xs, ys, marker='o')
                    
                    # Annotate each point with a number
                    for i, (x, y) in enumerate(zip(xs, ys), start=1):
                        plt.text(x, y, f'{i}', fontsize=8, ha='right')

                plt.title(f'Polyline Plot for {entity_value} - {name}')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.grid(True)

                # Save the plot
                output_path = os.path.join(output_directory, f"polyline_plot_{entity_value}_{name}.png")
                plt.savefig(output_path)
                plt.show()

                print(f"Polyline plot for {name} saved to {output_path}")

                # Create and save the table of coordinates
                coordinates_df = pd.DataFrame({'Point': range(1, len(xs)+1), 'X Coordinate': xs, 'Y Coordinate': ys})
                
                coordinates_table_path_csv = os.path.join(output_directory, f"coordinates_table_{entity_value}_{name}.csv")
                coordinates_df.to_csv(coordinates_table_path_csv, index=False)

                coordinates_table_path_excel = os.path.join(output_directory, f"coordinates_table_{entity_value}_{name}.xlsx")
                coordinates_df.to_excel(coordinates_table_path_excel, index=False)

                print(f"Coordinates table for {name} saved to {coordinates_table_path_csv} and {coordinates_table_path_excel}")
        else:
            print("Filtered data frame is empty. Please apply a filter first.")

if __name__ == "__main__":
    table_directory = "../data/output_tables/"
    filename = "_combined_entities.xlsx"
    output_dir = "../data/output_graphs/"  # Specify the output directory

    filepath = os.path.join(table_directory, filename)
    
    file_reader = ExcelFileReader(filepath)
    df = file_reader.read_file()
    
    if df is not None:
        file_reader.display_contents()

        column_name = 'Type'  # Replace with actual column name
        values = ['LINE', 'POLYLINE']  # Replace with actual value to filter by
        for value in values:
            file_reader.filter_by_column_value(column_name, value)

            if value == 'LINE' and file_reader.filtered_data_frame is not None:
                file_reader.filtered_data_frame = file_reader.filtered_data_frame[['Filename','Type', 'Block', 'Start_X', 'Start_Y', 'End_X', 'End_Y']]
                file_reader.display_filtered_contents()
                file_reader.draw_lines(output_dir)

            if value == 'POLYLINE':
                file_reader.filtered_data_frame = file_reader.filtered_data_frame[['Filename','Type', 'Block', 'Vertices']]
                file_reader.display_filtered_contents()
                file_reader.draw_polylines(output_dir, value)
