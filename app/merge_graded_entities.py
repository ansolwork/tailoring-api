import pandas as pd
import os
import numpy as np
import logging
import re

class MergeGradedEntities:
    def __init__(self, graded_folder, labeled_folder, item, min_size=30, max_size=62):
        self.graded_folder = graded_folder
        self.labeled_folder = labeled_folder
        self.output_folder = os.path.join("data/input/graded_mtm_combined_entities_labeled", item)
        self.graded_data = {}
        self.labeled_data = {}
        self.min_size = min_size
        self.max_size = max_size

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
                    # Extract sizes from both filename and text fields
                    sizes = set()
                    if 'Filename' in graded_df.columns:
                        sizes.update(int(size) for size in get_sizes_from_text(graded_df['Filename'].iloc[0], self.min_size, self.max_size))
                    if 'Text' in graded_df.columns:
                        for text in graded_df['Text'].dropna().unique():
                            sizes.update(int(size) for size in get_sizes_from_text(text, self.min_size, self.max_size))
                    
                    logging.info(f"Found sizes for piece {piece_name}: {sorted(list(sizes))}")
                    
                    # Apply MTM points
                    for point_label, mtm_point in mtm_mapping.items():
                        mask = graded_df['Point Label'] == point_label
                        graded_df.loc[mask, 'MTM Points'] = mtm_point
                        # Add size information
                        graded_df.loc[mask, 'size'] = ','.join(str(size) for size in sorted(sizes))
                    
                    logging.info(f"Applied MTM points to graded file")

    def save_merged_data(self):
        for piece_name, dfs in self.graded_data.items():
            os.makedirs(self.output_folder, exist_ok=True)
            all_size_dfs = []
            
            for df in dfs:
                if 'Filename' in df.columns:
                    # Get base filename without extension
                    original_filename = df['Filename'].iloc[0]
                    base_filename = re.sub(r'\.dxf$', '', original_filename)  # Remove .dxf first
                    base_filename = re.sub(r'-\d+$', '', base_filename)      # Then remove any trailing size
                    
                    # Extract sizes from Text field
                    text_sizes = set()
                    if 'Text' in df.columns:
                        for text in df['Text'].dropna().unique():
                            found_sizes = get_sizes_from_text(text)
                            text_sizes.update(found_sizes)
                    
                    # If sizes found in Text field, create a copy of df for each size
                    if text_sizes:
                        for size in text_sizes:
                            size_df = df.copy()
                            size_df['Filename'] = f"{base_filename}-{size}.dxf"  # Proper filename construction
                            size_df['size'] = size
                            all_size_dfs.append(size_df)
                    else:
                        # Check for size in filename
                        size_match = re.search(r'-(\d+)\.dxf$', original_filename)
                        if size_match:
                            df['size'] = size_match.group(1)
                            all_size_dfs.append(df)
                        else:
                            # Keep original if no size found
                            all_size_dfs.append(df)
                else:
                    # Keep original if no Filename column
                    all_size_dfs.append(df)
            
            if all_size_dfs:
                final_df = pd.concat(all_size_dfs, ignore_index=True)
                output_path = os.path.join(self.output_folder, f'{piece_name}_graded_combined_entities_labeled.xlsx')
                final_df.to_excel(output_path, index=False)
                print(f"Saved combined data for {piece_name} with all sizes to {output_path}")

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
