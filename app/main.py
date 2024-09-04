from app.create_table import CreateTable
from app.make_alteration import MakeAlteration  # Assuming this is in a separate file named 'make_alteration.py'
from app.visualize_alteration import VisualizeAlteration  # Import the visualization class
import os
import pandas as pd

class Main:
    def __init__(self, alteration_filepath, combined_entities_folder, 
                 staging_1_alteration, staging_1_vertices,
                 staging_2_alteration, staging_2_vertices
                 ):
        self.create_table = CreateTable(alteration_filepath, combined_entities_folder)
        self.make_alteration = MakeAlteration(input_folder_alteration=staging_1_alteration, input_folder_vertices=staging_1_vertices)
        #self.visualize_alteration = VisualizeAlteration(processed_alterations_path, processed_vertices_path)

    def create_tables(self):
        # Process the sheets and get the combined DataFrame
        self.create_table.process_table()
        self.create_table.process_combined_entities()

        # Create Vertices DataFrame
        self.create_table.create_vertices_df()

        # Join tables
        self.create_table.merge_tables()
        
        # Save the combined DataFrame as CSV files 
        self.create_table.save_table_csv()
        self.create_table.add_other_mtm_points()

    def apply_alterations(self):
        self.make_alteration.alter_all()

    def visualize_results(self):
        # Prepare plot data
        self.visualize_alteration.prepare_plot_data()
        # Plot polylines table
        self.visualize_alteration.plot_polylines_table()
        # Plot the alteration table
        self.visualize_alteration.plot_alteration_table()

    def run(self):
        # Run the complete table creation process
        self.create_tables()

        # Apply alterations after tables are created
        self.apply_alterations()

        # Visualize the results after applying alterations
        #self.visualize_results()

if __name__ == "__main__":
    # Input 1
    alteration_filepath = "../data/input/mtm_points.xlsx"
    combined_entities_folder = "../data/input/mtm-combined-entities/"

    # Input 2 - After processing (Staging 1)
    alteration_staging_1 = "../data/staging_1/combined_alteration_tables/"
    vertices_staging_1 = "../data/staging_1/vertices/"

    # Input 3: Visualization
    processed_alterations_path = "../data/staging_2/processed_alterations/"
    processed_vertices_path = "../data/staging_2/processed_vertices/"

    # Initialize the Main class with the required file paths
    main_process = Main(alteration_filepath, combined_entities_folder, 
                        alteration_staging_1, vertices_staging_1,
                        processed_alterations_path, processed_vertices_path)
    
    # Execute the process
    main_process.run()
