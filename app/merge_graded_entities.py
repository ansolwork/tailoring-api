import pandas as pd
import os
import numpy as np
import logging
import re

class MergeGradedEntities:
    def __init__(self, graded_folder, labeled_folder, item):
        self.graded_folder = graded_folder
        self.labeled_folder = labeled_folder
        self.output_folder = os.path.join("data/input/graded_mtm_combined_entities_labeled", item)
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
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = os.path.join(self.output_folder, f'{piece_name}_graded_combined_entities_labeled.xlsx')
            
            # Concatenate all DataFrames for this piece
            merged_df = pd.concat(dfs, ignore_index=True)
            
            if 'Filename' in merged_df.columns:
                # First try to get size from TEXT field
                text_sizes = merged_df[merged_df['Type'] == 'TEXT']['Text'].apply(lambda x: get_sizes_from_text(x))
                text_sizes = text_sizes[text_sizes.apply(len) > 0].apply(lambda x: x[0])  # Take first valid size
                
                # For rows where text size is missing, fallback to filename
                filename_sizes = merged_df['Filename'].str.extract(r'-(\d+)\.dxf')[0]
                
                # Combine both size sources, prioritizing text sizes this time
                merged_df['Size'] = text_sizes.fillna(filename_sizes).astype(str)
                
                # Log input sizes
                input_sizes = sorted(merged_df['Size'].dropna().unique())
                logging.info(f"\nPiece {piece_name} input sizes: {input_sizes}")
                
                # Keep only rows where size was found
                merged_df = merged_df[merged_df['Size'].notna()]
                
                # Log remaining sizes after filtering
                final_sizes = sorted(merged_df['Size'].unique())
                logging.info(f"Piece {piece_name} final sizes after filtering: {final_sizes}")
                
                if len(merged_df) == 0:
                    logging.warning(f"Warning: No rows left after filtering for {piece_name}")
                    continue
            
            merged_df.to_excel(output_path, index=False)
            logging.info(f"Merged data for {piece_name} saved to {output_path}")

    def process(self):
        self.load_graded_data()
        self.load_labeled_data()
        
        # Debug information
        logging.info(f"Graded data pieces: {list(self.graded_data.keys())}")
        logging.info(f"Labeled data pieces: {list(self.labeled_data.keys())}")
        
        for piece, dfs in self.labeled_data.items():
            total_mtm_points = sum(len(df) for df in dfs)
            logging.info(f"Piece {piece} has {total_mtm_points} non-null MTM points in labeled data")
        
        # Add detailed logging for each piece's files
        for piece_name, dfs_list in self.graded_data.items():
            logging.info(f"\nProcessing files for piece {piece_name}:")
            for df in dfs_list:
                if 'Filename' in df.columns:
                    filenames = df['Filename'].unique()
                    for filename in filenames:
                        # Add logging for text elements to debug size extraction
                        text_rows = df[df['Type'] == 'TEXT']
                        logging.info(f"- Found file: {filename}")
                        logging.info(f"  Text elements: {text_rows['Text'].tolist()}")
                        size = self.process_file(filename, piece_name)
                        if size:
                            logging.info(f"  Extracted size: {size}")
        
        self.extract_and_insert_mtm_points()
        self.save_merged_data()

    def process_file(self, filepath, piece_name):
        # Since we already have the text elements in the DataFrame
        # We don't need to read the file again
        if isinstance(filepath, str):
            # Extract size from filename if it matches pattern
            match = re.search(r'-(\d+)\.dxf$', filepath)
            if match:
                return match.group(1)
        return None

    def extract_size_from_text(self, text_elements):
        """Extract size from text elements that match pattern '30 - 62 (XX)'"""
        for text in text_elements:
            # Look for pattern XX - XX (size)
            match = re.search(r'\d+\s*-\s*\d+\s*\((\d+)\)', text)
            if match:
                size = match.group(1)
                return size
        return None

def get_sizes_from_text(text, min_size=30, max_size=62):
    """
    Extract sizes from text, prioritizing parenthetical sizes and filename sizes
    Args:
        text: String to extract sizes from
        min_size: Minimum valid size (inclusive)
        max_size: Maximum valid size (inclusive)
    """
    if not isinstance(text, str):
        return []
    
    sizes = set()
    
    # First priority: Extract size from parentheses (32) from text like "30 - 62 (32)"
    paren_match = re.search(r'\((\d+)\)', text)
    if paren_match:
        size = int(paren_match.group(1))
        if min_size <= size <= max_size:
            sizes.add(str(size))
    
    # Second priority: Extract size from filename pattern like "-30.dxf"
    filename_match = re.search(r'-(\d{2})\.dxf$', text)
    if filename_match:
        size = int(filename_match.group(1))
        if min_size <= size <= max_size:
            sizes.add(str(size))
    
    return sorted(list(sizes))

if __name__ == "__main__":
    # Enable debug logging
    logging.basicConfig(
        level=logging.DEBUG,  # Change to DEBUG to see more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    item = "shirt"

    graded_folder = f"data/input/graded_mtm_combined_entities/{item}"
    labeled_folder = f"data/input/mtm_combined_entities_labeled/{item}"
    
    merger = MergeGradedEntities(graded_folder, labeled_folder, item)
    merger.process()
