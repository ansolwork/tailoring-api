import pandas as pd
import os
import numpy as np
import logging
import re
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast  # To safely evaluate string representations of lists
import matplotlib

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
        # First assign all instance variables
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
        """Clean up output folders before processing new files"""
        output_base = f"data/input/graded_mtm_combined_entities_labeled/{self.item}"
        
        # Remove the entire output directory if it exists
        if os.path.exists(output_base):
            logging.info(f"Cleaning up existing output folder: {output_base}")
            shutil.rmtree(output_base)
        
        # Create fresh directories
        os.makedirs(output_base, exist_ok=True)
        
        # Log the cleanup
        logging.info(f"Created clean output directory: {output_base}")

    def load_graded_data(self):
        for root, dirs, files in os.walk(self.graded_folder):
            for file in files:
                if file.endswith('.xlsx'):
                    file_path = os.path.join(root, file)
                    piece_name = os.path.basename(os.path.dirname(file_path))
                    df = pd.read_excel(file_path)
                    
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

    def extract_and_insert_mtm_points(self):
        for piece_name, labeled_dfs in self.labeled_data.items():
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
                
                logging.info(f"Reference MTM sequence: {mtm_sequence}")
                
                # Apply MTM points to graded files maintaining the same sequence
                for graded_df in self.graded_data[piece_name]:
                    # Extract sizes
                    sizes = set()
                    if 'Filename' in graded_df.columns:
                        sizes.update(int(size) for size in get_sizes_from_text(graded_df['Filename'].iloc[0], self.min_size, self.max_size))
                    if 'Text' in graded_df.columns:
                        for text in graded_df['Text'].dropna().unique():
                            sizes.update(int(size) for size in get_sizes_from_text(text, self.min_size, self.max_size))
                    
                    # Sort the graded file points in the same order as the reference
                    sorted_points = graded_df.sort_values(['Point Label', 'Vertex_Index'])
                    
                    # Apply MTM points maintaining the sequence
                    for point_label, mtm_point in mtm_sequence.items():
                        mask = graded_df['Point Label'] == point_label
                        graded_df.loc[mask, 'MTM Points'] = mtm_point
                        graded_df.loc[mask, 'size'] = ','.join(str(size) for size in sorted(sizes))
                    
                    logging.info(f"Applied MTM points to graded file")

    def save_merged_data(self):
        for piece_name, dfs in self.graded_data.items():
            # Create main output folder and all_sizes_merged subfolder
            os.makedirs(self.output_folder, exist_ok=True)
            all_sizes_folder = os.path.join(self.output_folder, "all_sizes_merged")
            os.makedirs(all_sizes_folder, exist_ok=True)
            
            # Create piece-specific folder for individual size files
            piece_folder = os.path.join(self.output_folder, piece_name)
            os.makedirs(piece_folder, exist_ok=True)
            
            # Dictionary to collect data by size
            size_data = {}
            # Initialize the list to store all size DataFrames
            all_size_dfs = []
            
            for df in dfs:
                if 'Filename' in df.columns:
                    original_filename = df['Filename'].iloc[0]
                    base_filename = re.sub(r'\.dxf$', '', original_filename)
                    base_filename = re.sub(r'-\d+$', '', base_filename)
                    
                    text_sizes = set()
                    if 'Text' in df.columns:
                        for text in df['Text'].dropna().unique():
                            found_sizes = get_sizes_from_text(text)
                            text_sizes.update(found_sizes)
                    
                    if text_sizes:
                        for size in text_sizes:
                            size_df = df.copy()
                            size_df['Filename'] = f"{base_filename}-{size}.dxf"
                            size_df['size'] = size
                            all_size_dfs.append(size_df)
                            
                            # Add to size-specific collection
                            if size not in size_data:
                                size_data[size] = []
                            size_data[size].append(size_df)
                    else:
                        size_match = re.search(r'-(\d+)\.dxf$', original_filename)
                        if size_match:
                            size = size_match.group(1)
                            df['size'] = size
                            all_size_dfs.append(df)
                            
                            # Add to size-specific collection
                            if size not in size_data:
                                size_data[size] = []
                            size_data[size].append(df)
                        else:
                            all_size_dfs.append(df)
                else:
                    all_size_dfs.append(df)
            
            if all_size_dfs:
                # Save individual size files with a progress bar
                with tqdm(total=len(size_data), desc=f"Saving {piece_name}", unit="size") as pbar:
                    for size, size_dfs in size_data.items():
                        if size_dfs:
                            size_df = pd.concat(size_dfs, ignore_index=True)
                            size_output_path = os.path.join(piece_folder, f'{piece_name}-{size}.xlsx')
                            size_df.to_excel(size_output_path, index=False)
                            pbar.update(1)
                    
                    # Save the combined file
                    final_df = pd.concat(all_size_dfs, ignore_index=True)
                    output_path = os.path.join(all_sizes_folder, f'{piece_name}_graded_combined_entities_labeled.xlsx')
                    final_df.to_excel(output_path, index=False)
        
        for piece, dfs in self.labeled_data.items():
            total_mtm_points = sum(len(df) for df in dfs)
        
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

    def visualize_labeled_graded_base_data(self):
        """Create visualizations for the graded data with MTM points"""
        viz_dir = os.path.join(self.output_folder, "images_base_size")
        os.makedirs(viz_dir, exist_ok=True)
        
        if not self.graded_data:
            logging.warning("No graded data found for visualization")
            return
        
        for piece_name, graded_dfs in self.graded_data.items():
            try:
                base_df = graded_dfs[0]
                
                # Create figure with adjusted size and margins
                plt.figure(figsize=(8, 8))
                
                # Get all x and y coordinates for setting plot limits
                x_coords_all = []
                y_coords_all = []
                
                # Plot the polygon using Vertices column
                for _, row in base_df.iterrows():
                    if pd.notna(row['Vertices']):
                        vertices = ast.literal_eval(row['Vertices'])
                        x_coords, y_coords = zip(*vertices)
                        plt.plot(x_coords, y_coords, 'b-', alpha=0.5)
                        x_coords_all.extend(x_coords)
                        y_coords_all.extend(y_coords)
                
                # Highlight MTM points
                mtm_points = base_df[pd.notna(base_df['MTM Points'])]
                
                if not mtm_points.empty:
                    plt.scatter(mtm_points['PL_POINT_X'], mtm_points['PL_POINT_Y'], 
                              c='red', s=50, label='MTM Points')
                    
                    # Add MTM point labels with smaller font size
                    for _, row in mtm_points.iterrows():
                        plt.annotate(f"{int(float(row['MTM Points']))}", 
                                   (row['PL_POINT_X'], row['PL_POINT_Y']),
                                   xytext=(3, 3),
                                   textcoords='offset points',
                                   fontsize=8)
                
                    plt.legend()
                
                plt.title(f"{piece_name} - Pattern with MTM Points")
                plt.axis('equal')
                plt.grid(True)
                
                # Set axis limits with reduced padding
                if x_coords_all and y_coords_all:
                    x_min, x_max = min(x_coords_all), max(x_coords_all)
                    y_min, y_max = min(y_coords_all), max(y_coords_all)
                    padding = 0.05
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    plt.xlim(x_min - x_range * padding, x_max + x_range * padding)
                    plt.ylim(y_min - y_range * padding, y_max + y_range * padding)
                
                plt.tight_layout(pad=0.5)
                
                output_path = os.path.join(viz_dir, f"{piece_name}_pattern.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                                
            except Exception as e:
                logging.error(f"Error visualizing {piece_name}: {str(e)}")
                logging.error(f"DataFrame info: {base_df.info()}")

    def visualize_reference_labeled_base_data(self):
        """Create visualizations for the reference labeled MTM files"""
        viz_dir = os.path.join(self.labeled_folder, "images_labeled")
        os.makedirs(viz_dir, exist_ok=True)
        
        if not self.labeled_data:
            logging.warning("No labeled data found for visualization")
            return
        
        for piece_name, labeled_dfs in self.labeled_data.items():
            try:
                base_df = labeled_dfs[0]  # Take first DataFrame from the list
                
                # Create figure with adjusted size and margins
                plt.figure(figsize=(8, 8))
                
                # Get all x and y coordinates for setting plot limits
                x_coords_all = []
                y_coords_all = []
                
                # Plot the polygon using Vertices column
                for _, row in base_df.iterrows():
                    if pd.notna(row['Vertices']):
                        vertices = ast.literal_eval(row['Vertices'])
                        x_coords, y_coords = zip(*vertices)
                        plt.plot(x_coords, y_coords, 'b-', alpha=0.5)
                        x_coords_all.extend(x_coords)
                        y_coords_all.extend(y_coords)
                
                # Highlight MTM points
                mtm_points = base_df[pd.notna(base_df['MTM Points'])]
                
                if not mtm_points.empty:
                    plt.scatter(mtm_points['PL_POINT_X'], mtm_points['PL_POINT_Y'], 
                              c='red', s=50, label='MTM Points')
                    
                    # Add MTM point labels with smaller font size
                    for _, row in mtm_points.iterrows():
                        plt.annotate(f"{int(float(row['MTM Points']))}", 
                                   (row['PL_POINT_X'], row['PL_POINT_Y']),
                                   xytext=(3, 3),
                                   textcoords='offset points',
                                   fontsize=8)
                
                    plt.legend()
                
                plt.title(f"{piece_name} - Labeled Pattern with MTM Points")
                plt.axis('equal')
                plt.grid(True)
                
                # Set axis limits with reduced padding
                if x_coords_all and y_coords_all:
                    x_min, x_max = min(x_coords_all), max(x_coords_all)
                    y_min, y_max = min(y_coords_all), max(y_coords_all)
                    padding = 0.05
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    plt.xlim(x_min - x_range * padding, x_max + x_range * padding)
                    plt.ylim(y_min - y_range * padding, y_max + y_range * padding)
                
                plt.tight_layout(pad=0.5)
                
                output_path = os.path.join(viz_dir, f"{piece_name}_labeled_pattern.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                                
            except Exception as e:
                logging.error(f"Error visualizing labeled file for {piece_name}: {str(e)}")
                logging.error(f"DataFrame info: {base_df.info()}")

    def process(self):
        logging.debug("Starting process method")
        self.load_graded_data()
        logging.debug("Finished loading graded data")
        self.load_labeled_data()
        logging.debug("Finished loading labeled data")
        self.extract_and_insert_mtm_points()
        logging.debug("Finished extracting and inserting MTM points")
        self.save_merged_data()
        logging.debug("Finished saving merged data")
        self.visualize_labeled_graded_base_data()
        self.visualize_reference_labeled_base_data()
        logging.debug("Finished visualizing base data")

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
