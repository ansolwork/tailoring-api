import pandas as pd
import os
import numpy as np
import logging
import re
import shutil
from tqdm import tqdm
import ast  # To safely evaluate string representations of lists
from concurrent.futures import ThreadPoolExecutor

class ExcludeFontDebugFilter(logging.Filter):
    def filter(self, record):
        # Exclude messages that contain 'findfont'
        return 'findfont' not in record.getMessage()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress debug messages from matplotlib's font manager
font_manager_logger = logging.getLogger('matplotlib.font_manager')
font_manager_logger.setLevel(logging.WARNING)

# Get the matplotlib logger and add the filter
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.addFilter(ExcludeFontDebugFilter())

class MergeGradedEntities:
    def __init__(self, graded_folder, labeled_folder, item):
        self.graded_folder = graded_folder
        self.labeled_folder = labeled_folder
        self.item = item
        
        # Initialize other instance variables
        self.base_files = {}
        self.graded_files = {}
        self.graded_data = {}    
        self.labeled_data = {}   
        self.min_size = 30      
        self.max_size = 62      
        self.output_folder = f"data/input/graded_mtm_combined_entities_labeled/{self.item}"  
        
        # Configure logging to show DEBUG level and above
        logging.getLogger().setLevel(logging.DEBUG)
        
        class ExcludeFilter(logging.Filter):
            def filter(self, record):
                message = record.getMessage()
                # List of patterns to exclude
                exclude_patterns = [
                    "Applied MTM points",
                    "Reference MTM sequence:",
                    "Text elements:",
                    "Found file:",
                    "Extracted size:",
                    "Processing files for piece"
                ]
                return not any(pattern in message for pattern in exclude_patterns)
        
        # Add the filter to the root logger
        logging.getLogger().addFilter(ExcludeFilter())
        
        # Then call cleanup
        self.cleanup_output_folders()

    def cleanup_output_folders(self):
        """Clean up the entire graded_mtm_combined_entities_labeled folder before processing"""
        try:
            # Main output directory to clean
            output_base = "data/input/graded_mtm_combined_entities_labeled"
            
            if os.path.exists(output_base):
                logging.info(f"Removing entire output directory: {output_base}")
                shutil.rmtree(output_base, ignore_errors=True)
            
            # Recreate the main item directory
            item_dir = os.path.join(output_base, self.item)
            os.makedirs(item_dir, exist_ok=True)
            logging.info(f"Created fresh output directory: {item_dir}")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            raise

    def load_graded_data(self):
        expected_columns = ['Filename', 'Type', 'Layer', 'Color', 'Text', 'Point_X', 'Point_Y', 'Height', 'Style', 'Block', 
                           'Line_Start_X', 'Line_Start_Y', 'Line_End_X', 'Line_End_Y', 'Vertices', 'PL_POINT_X', 'PL_POINT_Y', 
                           'Vertex_Index', 'Point Label', 'Vertex Label', 'MTM Points']
        
        for root, dirs, files in os.walk(self.graded_folder):
            for file in files:
                if file.endswith('.xlsx'):
                    # Skip GRADE-RULE files
                    if 'GRADE-RULE' in file:
                        logging.debug(f"Skipping grade rule file: {file}")
                        continue
                    
                    file_path = os.path.join(root, file)
                    piece_name = os.path.basename(os.path.dirname(file_path))
                    df = pd.read_excel(file_path)
                    
                    # Only log if the DataFrame doesn't have the expected columns
                    if not all(col in df.columns for col in expected_columns):
                        logging.error(f"Unexpected columns in file: {file_path}")
                        logging.error(f"Found columns: {df.columns.tolist()}")
                        logging.error(f"Missing columns: {[col for col in expected_columns if col not in df.columns]}")
                    
                    # Store in graded_data
                    if piece_name not in self.graded_data:
                        self.graded_data[piece_name] = []
                    self.graded_data[piece_name].append(df)
                    
                    # Identify and store base file (file without size in name)
                    if not re.search(r'-\d+\.xlsx$', file):
                        self.base_files[piece_name] = file_path

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

    def get_sizes_from_text(self, text, min_size=30, max_size=62):
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

    def extract_and_insert_mtm_points(self):
        try:
            for piece_name, labeled_dfs in self.labeled_data.items():
                logging.debug(f"Processing piece: {piece_name}")
                if piece_name in self.graded_data:
                    # Get the reference MTM point sequence from labeled data
                    mtm_sequence = {}
                    for labeled_df in labeled_dfs:
                        # Sort polylines by their vertex index to maintain consistent order
                        polylines = labeled_df[labeled_df['MTM Points'].notna()].sort_values('Vertex_Index')
                        
                        # Create mapping of vertex sequence to MTM point number
                        for _, row in polylines.iterrows():
                            vertex_idx = row['Vertex_Index']
                            point_label = row['Point Label']
                            mtm_point = row['MTM Points']
                            mtm_sequence[point_label] = mtm_point
                    
                    # Apply MTM points to graded files
                    for graded_df in self.graded_data[piece_name]:
                        # Remove debug print of columns
                        # Skip if DataFrame doesn't have required columns
                        if 'Point Label' not in graded_df.columns:
                            logging.warning(f"Skipping DataFrame for {piece_name} - missing 'Point Label' column")
                            continue
                        
                        # Extract sizes
                        sizes = set()
                        if 'Filename' in graded_df.columns:
                            sizes.update(int(size) for size in self.get_sizes_from_text(graded_df['Filename'].iloc[0], self.min_size, self.max_size))
                        if 'Text' in graded_df.columns:
                            for text in graded_df['Text'].dropna().unique():
                                sizes.update(int(size) for size in self.get_sizes_from_text(text, self.min_size, self.max_size))
                        
                        # Apply MTM points without sorting
                        for point_label, mtm_point in mtm_sequence.items():
                            mask = graded_df['Point Label'] == point_label
                            graded_df.loc[mask, 'MTM Points'] = mtm_point
                            graded_df.loc[mask, 'size'] = ','.join(str(size) for size in sorted(sizes))
                        
                        logging.info(f"Applied MTM points to graded file")
        except Exception as e:
            logging.error(f"Error in extract_and_insert_mtm_points: {str(e)}")
            logging.error("Full traceback:", exc_info=True)
            raise

    def save_merged_data(self):
        """Save processed data to output files"""
        for piece_name, dfs in self.graded_data.items():
            # Create output folders
            os.makedirs(self.output_folder, exist_ok=True)
            all_sizes_folder = os.path.join(self.output_folder, "all_sizes_merged")
            os.makedirs(all_sizes_folder, exist_ok=True)
            piece_folder = os.path.join(self.output_folder, piece_name)
            os.makedirs(piece_folder, exist_ok=True)
            
            # Initialize data collectors
            size_data = {}
            all_size_dfs = []
            
            # Process each DataFrame
            for df in dfs:
                if 'Filename' in df.columns:
                    original_filename = df['Filename'].iloc[0]
                    base_filename = re.sub(r'\.dxf$', '', original_filename)
                    base_filename = re.sub(r'-\d+$', '', base_filename)
                    
                    # Extract sizes from Text column
                    text_sizes = set()
                    if 'Text' in df.columns:
                        for text in df['Text'].dropna().unique():
                            found_sizes = self.get_sizes_from_text(text)
                            text_sizes.update(found_sizes)
                    
                    if text_sizes:
                        # Create separate DataFrames for each size
                        for size in text_sizes:
                            size_df = df.copy()
                            size_df['Filename'] = f"{base_filename}-{size}.dxf"
                            size_df['size'] = size
                            all_size_dfs.append(size_df)
                            
                            if size not in size_data:
                                size_data[size] = []
                            size_data[size].append(size_df)
                    else:
                        # Try to extract size from filename
                        size_match = re.search(r'-(\d+)\.dxf$', original_filename)
                        if size_match:
                            size = size_match.group(1)
                            df['size'] = size
                            all_size_dfs.append(df)
                            
                            if size not in size_data:
                                size_data[size] = []
                            size_data[size].append(df)
                        else:
                            all_size_dfs.append(df)
                else:
                    all_size_dfs.append(df)
            
            if all_size_dfs:
                # Save individual size files
                with tqdm(total=len(size_data), desc=f"Saving {piece_name}", unit="size") as pbar:
                    for size, size_dfs in size_data.items():
                        if size_dfs:
                            size_df = pd.concat(size_dfs, ignore_index=True)
                            size_output_path = os.path.join(piece_folder, f'{piece_name}-{size}.xlsx')
                            size_df.to_excel(size_output_path, index=False)
                            pbar.update(1)
                    
                    # Save the combined file for all sizes
                    final_df = pd.concat(all_size_dfs, ignore_index=True)
                    output_path = os.path.join(all_sizes_folder, f'{piece_name}_graded_combined_entities_labeled.xlsx')
                    final_df.to_excel(output_path, index=False)
                    logging.info(f"Saved combined file for {piece_name}")

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

    def process(self):
        """Main processing method without visualization"""
        try:
            self.cleanup_output_folders()
            self.load_graded_data()
            self.load_labeled_data()
            self.extract_and_insert_mtm_points()
            self.save_merged_data()
            # Return the processed data
            return {
                'labeled_data': self.labeled_data,
                'graded_data': self.graded_data
            }
        except Exception as e:
            logging.error(f"Error in process: {str(e)}")
            self.cleanup_output_folders()
            raise

# Usage example:
if __name__ == "__main__":
    try:
        item = "shirt"
        
        # Data processing
        graded_folder = f"data/input/graded_mtm_combined_entities/{item}"
        labeled_folder = f"data/input/mtm_combined_entities_labeled/{item}"
        merger = MergeGradedEntities(graded_folder, labeled_folder, item)
        processed_data = merger.process()
        
            
    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
