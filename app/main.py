from create_table import CreateTable
from make_alteration import MakeAlteration  # Assuming this is in a separate file named 'make_alteration.py'
from visualize_alteration import VisualizeAlteration  # Import the visualization class

class Main:
    def __init__(self, alteration_filepath, combined_entities_folder, 
                 preprocessed_table_path, input_vertices_path,
                 processed_alterations_path, processed_vertices_path
                 ):
        self.create_table = CreateTable(alteration_filepath, combined_entities_folder)
        self.make_alteration = MakeAlteration(preprocessed_table_path, input_vertices_path)
        self.visualize_alteration = VisualizeAlteration(processed_alterations_path, processed_vertices_path)

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
        # Apply alteration rules
        self.make_alteration.apply_alteration_rules(custom_alteration=False)

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
        self.visualize_results()

if __name__ == "__main__":
    # Input 1
    alteration_filepath = "../data/input/MTM-POINTS.xlsx"
    combined_entities_folder = "../data/input/mtm-combined-entities/"

    # Input 2 - After processing
    preprocessed_table_path = "../data/output_tables/combined_alteration_tables/combined_table_4-WAIST.csv"
    input_vertices_path = "../data/output_tables/vertices/LGFG-SH-01-CCB-FO_vertices.csv"

    # Input 3: Visualization
    processed_alterations_path = "../data/output_tables/processed_alterations_2.xlsx"
    processed_vertices_path = "../data/output_tables/vertices_df.xlsx"

    # Initialize the Main class with the required file paths
    main_process = Main(alteration_filepath, combined_entities_folder, 
                        preprocessed_table_path, input_vertices_path,
                        processed_alterations_path, processed_vertices_path)
    
    # Execute the process
    main_process.run()
