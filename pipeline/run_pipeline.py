import os
import argparse
import yaml
from app.create_table import CreateTable


class PipelineRunner:
    def __init__(self, config_path):
        # Load config from YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Initialize paths from config
        self.mtm_dir_path = config['AWS_MTM_DIR_PATH']
        self.mtm_graded_dir_path = config['AWS_MTM_GRADED_DIR_PATH']
        self.mtm_dir_path_labeled = config['AWS_MTM_DIR_PATH_LABELED']
        self.mtm_graded_dir_path_labeled = config['AWS_MTM_GRADED_DIR_PATH_LABELED']
        self.plot_dir_base = config['AWS_PLOT_DIR_BASE']
        self.output_dir_path = config['AWS_OUTPUT_DIR_PATH']
        self.alteration_dir_path = config['AWS_MTM_ALTERATION_DIR_PATH']

    def concatenate_item_subdirectories(self, path, item):
        return os.path.join(path, item)

    def create_tables(self, alteration_path, mtm_path):
        # Create CreateTable instance
        create_table = CreateTable(alteration_path, mtm_path)
        # Process the sheets and get the combined DataFrame
        create_table.process_table()
        create_table.process_combined_entities()

        # Create Vertices DataFrame
        create_table.create_vertices_df()

        # Join tables
        create_table.merge_tables()

        # Save the combined DataFrame as CSV files
        create_table.save_table_csv()
        create_table.add_other_mtm_points()

    def run_pipeline(self, item):
        # Create all necessary paths using the item argument
        mtm_path = self.concatenate_item_subdirectories(self.mtm_dir_path, item)
        mtm_graded_path = self.concatenate_item_subdirectories(self.mtm_graded_dir_path, item)
        mtm_labeled_path = self.concatenate_item_subdirectories(self.mtm_dir_path_labeled, item)
        mtm_graded_labeled_path = self.concatenate_item_subdirectories(self.mtm_graded_dir_path_labeled, item)
        plot_path = self.concatenate_item_subdirectories(self.plot_dir_base, item)
        output_path = self.concatenate_item_subdirectories(self.output_dir_path, item)
        alteration_path = self.concatenate_item_subdirectories(self.alteration_dir_path, item)

        self.create_tables(alteration_path=alteration_path, mtm_path=mtm_path)
        # TODO: Add instances to process the tables and rest of the pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the pipeline with a specific item')
    parser.add_argument('--item', required=True, help='Item to process in the pipeline')
    return parser.parse_args()


# Can be used for local testing and debugging and not for API
def main(item):
    if not item:
        raise ValueError("Item parameter is required")

    pipeline = PipelineRunner('tailoring_api_config.yml')
    return pipeline.run_pipeline(item)


# Used for API
def process_pipeline_request(request_data):
    """
    Handle pipeline requests from API
    """
    if 'item' not in request_data:
        raise ValueError("Request must include 'item' parameter")

    if not request_data['item']:
        raise ValueError("Item parameter is required")

    pipeline = PipelineRunner('tailoring_api_config.yml')
    return pipeline.run_pipeline(request_data['item'])


if __name__ == "__main__":
    args = parse_arguments()
    main(args.item)
