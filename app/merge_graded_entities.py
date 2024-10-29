import pandas as pd
import os
import numpy as np
import logging

class MergeGradedEntities:
    def __init__(self, graded_folder, labeled_folder):
        self.graded_folder = graded_folder
        self.labeled_folder = labeled_folder
        self.output_folder = "data/input/merged_graded_labeled_entities"
        self.graded_data = {}
        self.labeled_data = {}

    def load_graded_data(self):
        for root, dirs, files in os.walk(self.graded_folder):
            for file in files:
                if file.endswith('.xlsx'):
                    file_path = os.path.join(root, file)
                    piece_name = os.path.basename(os.path.dirname(file_path))
                    df = pd.read_excel(file_path)
                    if piece_name not in self.graded_data:
                        self.graded_data[piece_name] = []
                    self.graded_data[piece_name].append(df)

    def load_labeled_data(self):
        for file in os.listdir(self.labeled_folder):
            if file.endswith('.xlsx'):
                file_path = os.path.join(self.labeled_folder, file)
                df = pd.read_excel(file_path)
                # Extract non-NaN rows from 'MTM Points' column
                labeled_df = df[df['MTM Points'].notna()]
                
                # Associate labeled data with all matching piece names
                for piece_name in self.graded_data.keys():
                    if piece_name in file:
                        if piece_name not in self.labeled_data:
                            self.labeled_data[piece_name] = []
                        self.labeled_data[piece_name].append(labeled_df)

    def extract_and_insert_mtm_points(self):
        for piece_name, labeled_dfs in self.labeled_data.items():
            if piece_name in self.graded_data:
                # Get the mapping of Point Labels to MTM Points from labeled data
                mtm_mapping = {}
                for labeled_df in labeled_dfs:
                    mtm_points = labeled_df[labeled_df['MTM Points'].notna()]
                    for _, row in mtm_points.iterrows():
                        mtm_mapping[row['Point Label']] = row['MTM Points']
                
                logging.info(f"Found MTM points at Point Labels: {list(mtm_mapping.keys())}")
                
                # Apply MTM points to graded files using Point Label positions
                for graded_df in self.graded_data[piece_name]:
                    for point_label, mtm_point in mtm_mapping.items():
                        mask = graded_df['Point Label'] == point_label
                        graded_df.loc[mask, 'MTM Points'] = mtm_point
                    
                    logging.info(f"Applied MTM points to graded file")

    def save_merged_data(self):
        for piece_name, dfs in self.graded_data.items():
            piece_folder = os.path.join(self.output_folder, piece_name)
            os.makedirs(piece_folder, exist_ok=True)
            output_path = os.path.join(piece_folder, f'{piece_name}_graded_combined_entities_labeled.xlsx')
            
            # Concatenate all DataFrames for this piece
            merged_df = pd.concat(dfs, ignore_index=True)
            
            merged_df.to_excel(output_path, index=False)
            print(f"Merged data for {piece_name} saved to {output_path}")

    def process(self):
        self.load_graded_data()
        self.load_labeled_data()
        
        # Debug information
        logging.info(f"Graded data pieces: {list(self.graded_data.keys())}")
        logging.info(f"Labeled data pieces: {list(self.labeled_data.keys())}")
        
        for piece, dfs in self.labeled_data.items():
            total_mtm_points = sum(len(df) for df in dfs)
            logging.info(f"Piece {piece} has {total_mtm_points} non-null MTM points in labeled data")
        
        self.extract_and_insert_mtm_points()
        self.save_merged_data()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    graded_folder = "data/input/graded_mtm_combined_entities"
    labeled_folder = "data/input/mtm_combined_entities_labeled"
    
    merger = MergeGradedEntities(graded_folder, labeled_folder)
    merger.process()
