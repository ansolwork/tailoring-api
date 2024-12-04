import logging
import os
import matplotlib.pyplot as plt
import ast
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
import re

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

"""
Pattern Visualization Module

This module handles the visualization of garment patterns, including base patterns, 
labeled patterns, and graded patterns across different sizes.

Directory Structure:
------------------
data/input/
    ├── mtm_combined_entities_labeled/{item}/
    │   ├── images/                    # Base pattern visualizations
    │   │   └── {piece_name}/         
    │   │       └── {piece_name}-size.png
    │   ├── images_labeled/            # Reference pattern visualizations
    │   │   └── {piece_name}_labeled-size.png
    │   └── coordinates/               # Coordinate data (if needed)
    └── graded_mtm_combined_entities/{item}/
        └── images_graded/             # Graded pattern visualizations
            └── {piece_name}/
                └── {piece_name}-{size}.png

Classes:
--------
PatternVisualizer: Main class for generating pattern visualizations
"""

class PatternVisualizer:
    """
    A class to generate and save pattern visualizations.
    
    This class handles three types of visualizations:
    1. Base patterns with MTM points
    2. Reference labeled patterns
    3. Graded patterns for each size
    
    Attributes:
        output_base_dir (str): Base directory for output files
        item (str): Item type (e.g., 'shirt')
        graded_folder (str): Path to graded pattern data
        labeled_folder (str): Path to labeled pattern data
        labeled_data (dict): Dictionary of labeled pattern DataFrames
        graded_data (dict): Dictionary of graded pattern DataFrames
    """

    def __init__(self, output_base_dir, item, labeled_data=None, graded_data=None):
        self.output_base_dir = output_base_dir
        self.item = item
        self.graded_folder = f"data/input/graded_mtm_combined_entities/{item}"
        self.labeled_folder = f"data/input/mtm_combined_entities_labeled/{item}"
        self.labeled_data = labeled_data
        self.graded_data = graded_data
        
    def visualize_labeled_graded_base_data(self):
        """
        Create visualizations for the labeled MTM files.
        
        Generates base pattern visualizations with:
        - Blue lines for pattern outlines
        - Red dots for MTM points
        - Point labels with MTM numbers
        - Grid and axis formatting
        
        Saves to: {labeled_folder}/images/{piece_name}/{piece_name}_pattern.png
        """
        print("\n📊 Generating base data visualizations:")
        print("======================================")
        
        # Group files by piece name
        piece_groups = {}
        for piece_name, labeled_dfs in self.labeled_data.items():
            if piece_name not in piece_groups:
                piece_groups[piece_name] = labeled_dfs
        
        # Create base directories
        viz_dir = os.path.join(self.labeled_folder, "images")
        coord_dir = os.path.join(self.labeled_folder, "coordinates")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(coord_dir, exist_ok=True)
        
        # Process each piece with its own progress bar
        for piece_name, labeled_dfs in tqdm(piece_groups.items(), 
                                          desc="Processing pieces", 
                                          unit="piece"):
            try:
                # Create piece-specific directory
                piece_viz_dir = os.path.join(viz_dir, piece_name)
                os.makedirs(piece_viz_dir, exist_ok=True)
                
                base_df = labeled_dfs[0]
                
                # Create figure with optimized size and DPI
                plt.figure(figsize=(15, 15), dpi=100)
                
                # Plot pattern with optimized settings
                for _, row in base_df.iterrows():
                    if pd.notna(row['Vertices']):
                        vertices = ast.literal_eval(row['Vertices'])
                        x_coords, y_coords = zip(*vertices)
                        plt.plot(x_coords, y_coords, 'b-', linewidth=2.0)
                
                # Plot MTM points
                mtm_points = base_df[pd.notna(base_df['MTM Points'])]
                if not mtm_points.empty:
                    plt.scatter(mtm_points['PL_POINT_X'], mtm_points['PL_POINT_Y'], 
                              c='red', s=800, zorder=5)
                    
                    for _, row in mtm_points.iterrows():
                        plt.annotate(
                            f"{int(float(row['MTM Points']))}", 
                            (row['PL_POINT_X'], row['PL_POINT_Y']),
                            xytext=(20, 20),
                            textcoords='offset points',
                            fontsize=20,
                            weight='bold'
                        )
                
                # Optimize grid and axis settings
                plt.grid(True, which='major', linestyle='-', linewidth=1.0, alpha=0.3)
                plt.minorticks_on()
                
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                
                # Simplified title
                plt.title(f"{piece_name}\nBase Pattern with MTM Points", 
                         fontsize=16, pad=20)
                
                # Save with optimized settings
                output_path = os.path.join(piece_viz_dir, f"{piece_name}_pattern.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=100)
                plt.close()
                
                print(f"✅ Saved: {output_path}")
                
            except Exception as e:
                logging.error(f"Error visualizing {piece_name}: {str(e)}")

    def visualize_reference_labeled_base_data(self):
        """
        Create visualizations for the reference labeled MTM files.
        
        Generates reference pattern visualizations with:
        - Semi-transparent blue lines for pattern outlines
        - Red dots for MTM points with labels
        - Smaller, more compact visualization format
        
        Saves to: {labeled_folder}/images_labeled/{piece_name}_labeled_pattern.png
        """
        print("\n📊 Generating reference visualizations:")
        print("======================================")
        
        viz_dir = os.path.join(self.labeled_folder, "images_labeled")
        os.makedirs(viz_dir, exist_ok=True)
        
        if not self.labeled_data:
            logging.warning("No labeled data found for visualization")
            return
        
        # Create progress bar for pieces
        for piece_name, labeled_dfs in tqdm(self.labeled_data.items(),
                                          desc="Saving reference visualizations",
                                          unit="piece"):
            try:
                base_df = labeled_dfs[0]  # Take first DataFrame from the list
                
                # Create figure with adjusted size and margins
                plt.figure(figsize=(15, 15), dpi=100)  # Reduced size and DPI
                
                # Get all x and y coordinates for setting plot limits
                x_coords_all = []
                y_coords_all = []
                
                # Plot the polygon using Vertices column
                for _, row in base_df.iterrows():
                    if pd.notna(row['Vertices']):
                        vertices = ast.literal_eval(row['Vertices'])
                        x_coords, y_coords = zip(*vertices)
                        plt.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=2.0)  # Reduced line width
                        x_coords_all.extend(x_coords)
                        y_coords_all.extend(y_coords)
                
                # Highlight MTM points
                mtm_points = base_df[pd.notna(base_df['MTM Points'])]
                
                if not mtm_points.empty:
                    plt.scatter(mtm_points['PL_POINT_X'], mtm_points['PL_POINT_Y'], 
                              c='red', s=800, label='MTM Points')  # Reduced point size
                    
                    # Add MTM point labels with smaller font size
                    for _, row in mtm_points.iterrows():
                        plt.annotate(f"{int(float(row['MTM Points']))}", 
                                   (row['PL_POINT_X'], row['PL_POINT_Y']),
                                   xytext=(3, 3),
                                   textcoords='offset points',
                                   fontsize=12)
                
                    plt.legend()
                
                plt.title(f"{piece_name} - Labeled Pattern with MTM Points", fontsize=16)
                plt.axis('equal')
                plt.grid(True, alpha=0.3)
                
                # Set axis limits with reduced padding
                if x_coords_all and y_coords_all:
                    x_min, x_max = min(x_coords_all), max(x_coords_all)
                    y_min, y_max = min(y_coords_all), max(y_coords_all)
                    padding = 0.005
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    plt.xlim(x_min - x_range * padding, x_max + x_range * padding)
                    plt.ylim(y_min - y_range * padding, y_max + y_range * padding)
                
                plt.tight_layout(pad=0.5)
                
                output_path = os.path.join(viz_dir, f"{piece_name}_labeled_pattern.png")
                plt.savefig(output_path, dpi=100, bbox_inches='tight')  # Reduced DPI
                plt.close()  # Immediately close the figure
                
                print(f"✅ Saved: {output_path}")
                
            except Exception as e:
                logging.error(f"Error visualizing labeled file for {piece_name}: {str(e)}")

    def visualize_graded_base_pieces(self):
        """
        Create visualizations for all graded pieces using the saved Excel files.
        
        Generates size-specific pattern visualizations with:
        - Thick blue lines for pattern outlines
        - Large red dots for vertex points
        - Point labels in white boxes
        - Scaled coordinates based on pattern size
        - Enhanced grid and axis formatting
        
        File Structure:
        - Creates a directory for each piece
        - Saves size-specific visualizations
        
        Saves to: {graded_folder}/images_graded/{piece_name}/{piece_name}-{size}.png
        
        Parameters in visualization:
        - Figure size: 200x200 inches
        - DPI: 150 for creation, 75 for saving
        - Font sizes: 32-120pt for various elements
        - Scaling factor: Automatically calculated based on pattern dimensions
        """
        print("\n📊 Generating graded piece visualizations:")
        print("=========================================")
        
        # Create and clean visualization directory
        graded_viz_dir = os.path.join(self.graded_folder, "images_graded")
        if os.path.exists(graded_viz_dir):
            shutil.rmtree(graded_viz_dir)
        os.makedirs(graded_viz_dir, exist_ok=True)
        
        # Get the path to the saved graded files
        graded_output_dir = os.path.join("data/input/graded_mtm_combined_entities_labeled", self.item)
        
        # Create progress bar for pieces
        pieces = [p for p in os.listdir(graded_output_dir) if os.path.isdir(os.path.join(graded_output_dir, p))]
        for piece_name in tqdm(pieces, desc="Processing pieces", unit="piece"):
            piece_dir = os.path.join(graded_output_dir, piece_name)
            piece_viz_dir = os.path.join(graded_viz_dir, piece_name)
            os.makedirs(piece_viz_dir, exist_ok=True)
            
            # Create progress bar for sizes within each piece
            size_files = [f for f in os.listdir(piece_dir) if f.endswith('.xlsx')]
            for size_file in tqdm(size_files, desc=f"Saving {piece_name}", unit="size", leave=False):
                try:
                    file_path = os.path.join(piece_dir, size_file)
                    df = pd.read_excel(file_path)
                    
                    size_match = re.search(r'-(\d+)\.xlsx$', size_file)
                    if not size_match:
                        continue
                    size = size_match.group(1)
                    
                    # Initialize scaling factor at the class level or method start
                    self.scaling_factor = None
                    
                    # Create larger figure (same as PlotGradedMTM)
                    plt.figure(figsize=(200, 200))
                    
                    # Calculate scaling factor based on data extents
                    x_min = min(df['Point_X'].min(), df['Line_Start_X'].min(), df['Line_End_X'].min())
                    x_max = max(df['Point_X'].max(), df['Line_Start_X'].max(), df['Line_End_X'].max())
                    y_min = min(df['Point_Y'].min(), df['Line_Start_Y'].min(), df['Line_End_Y'].min())
                    y_max = max(df['Point_Y'].max(), df['Line_Start_Y'].max(), df['Line_End_Y'].max())
                    
                    # Calculate scaling factor with safety checks
                    data_width = x_max - x_min
                    data_height = y_max - y_min
                    
                    # Add safety checks for scaling factor
                    if data_width == 0 or data_height == 0:
                        self.scaling_factor = 1.0  # Use default scaling if dimensions are zero
                    else:
                        self.scaling_factor = min(80 / data_width, 80 / data_height) * 0.9
                    
                    vertex_points = df[pd.notna(df['PL_POINT_X'])]
                    
                    # Initialize lists to collect ALL coordinates
                    x_coords_all = []
                    y_coords_all = []
                    
                    # First plot the polygon/vertices
                    for _, row in df.iterrows():
                        if pd.notna(row['Vertices']):  # Note the capital V in 'Vertices'
                            vertices = ast.literal_eval(row['Vertices'])
                            x_coords, y_coords = zip(*vertices)
                            # Scale the coordinates
                            x_coords = [x * self.scaling_factor for x in x_coords]
                            y_coords = [y * self.scaling_factor for y in y_coords]
                            plt.plot(x_coords, y_coords, 'b-', linewidth=8.0)  # Thick blue line
                            x_coords_all.extend(x_coords)
                            y_coords_all.extend(y_coords)
                    
                    # Then plot the points
                    if not vertex_points.empty:
                        scaled_x = vertex_points['PL_POINT_X'] * self.scaling_factor
                        scaled_y = vertex_points['PL_POINT_Y'] * self.scaling_factor
                        
                        plt.scatter(scaled_x, scaled_y, 
                                   c='red', s=2000, zorder=5, alpha=0.7)
                        
                        # Add labels
                        for _, row in vertex_points.iterrows():
                            if pd.notna(row.get('Point Label')):
                                plt.annotate(
                                    str(row['Point Label']), 
                                    (row['PL_POINT_X'] * self.scaling_factor, row['PL_POINT_Y'] * self.scaling_factor),
                                    xytext=(40, 40),
                                    textcoords='offset points',
                                    fontsize=32,
                                    color='black',
                                    weight='bold',
                                    bbox=dict(facecolor='white', 
                                             edgecolor='none',
                                             alpha=0.7,
                                             pad=1.5),
                                    ha='center',
                                    va='center'
                                )
                    
                    # Calculate the data extent from all coordinates
                    if x_coords_all and y_coords_all:
                        x_min, x_max = min(x_coords_all), max(x_coords_all)
                        y_min, y_max = min(y_coords_all), max(y_coords_all)
                        
                        # Add small padding (3%)
                        x_padding = (x_max - x_min) * 0.03
                        y_padding = (y_max - y_min) * 0.03
                        
                        # Set limits with padding
                        ax = plt.gca()
                        ax.set_xlim(x_min - x_padding, x_max + x_padding)
                        ax.set_ylim(y_min - y_padding, y_max + y_padding)
                        ax.set_aspect('equal', adjustable='box')
                    
                    # Rest of the styling remains the same
                    plt.grid(True, which='major', linestyle='-', linewidth=1.5, color='gray', alpha=0.5)
                    plt.grid(True, which='minor', linestyle=':', linewidth=0.8, color='gray', alpha=0.3)
                    plt.minorticks_on()
                    
                    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
                    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
                    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
                    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
                    
                    # Larger title
                    plt.title(f"{piece_name} - Size {size}", fontsize=80, pad=20)
                    
                    # Configure labels and title with larger fonts
                    plt.xlabel('X', fontsize=100, labelpad=20, weight='bold')
                    plt.ylabel('Y', fontsize=100, labelpad=20, weight='bold')
                    plt.tick_params(axis='both', which='major', labelsize=60, length=20, width=3, pad=15)
                    
                    # Save and close the figure
                    output_path = os.path.join(piece_viz_dir, f"{piece_name}-{size}.png")
                    plt.savefig(output_path, bbox_inches='tight', dpi=72, pad_inches=0.0)
                    plt.close()
                
                except Exception as e:
                    logging.error(f"Error visualizing {size_file}: {str(e)}")

    def visualize_all(self):
        """
        Run all visualization methods in sequence.
        
        Order of execution:
        1. Base pattern visualizations
        2. Reference labeled pattern visualizations
        3. Graded pattern visualizations
        
        Raises:
            Exception: If any visualization step fails
        """
        try:
            logging.info("Starting visualization process...")
            self.visualize_labeled_graded_base_data()
            self.visualize_reference_labeled_base_data()
            self.visualize_graded_base_pieces()
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            raise

if __name__ == "__main__":
    """
    Command-line interface for pattern visualization.
    
    Usage:
        python -m app.graded_pattern_visualizer --item shirt
    
    Arguments:
        --item: Type of garment to visualize (default: 'shirt')
    
    Process:
    1. Loads processed data from graded_mtm_combined_entities_labeled
    2. Loads labeled data from mtm_combined_entities_labeled
    3. Creates visualizations for all patterns and sizes
    
    Required Data Structure:
    - Labeled data must contain 'MTM Points' column
    - Graded data must be in *_graded_combined_entities_labeled.xlsx format
    - Pattern pieces must have consistent naming across files
    """
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Visualize pattern data')
    parser.add_argument('--item', default='shirt', help='Item to visualize')
    args = parser.parse_args()
    
    try:
        # Load processed data from saved files
        output_base_dir = "data/input/graded_mtm_combined_entities_labeled"
        item_dir = os.path.join(output_base_dir, args.item)
        
        # Initialize dictionaries to store data
        labeled_data = {}
        graded_data = {}
        
        # Load graded data
        all_sizes_dir = os.path.join(item_dir, "all_sizes_merged")
        if os.path.exists(all_sizes_dir):
            for file in os.listdir(all_sizes_dir):
                if file.endswith('_graded_combined_entities_labeled.xlsx'):
                    piece_name = file.replace('_graded_combined_entities_labeled.xlsx', '')
                    file_path = os.path.join(all_sizes_dir, file)
                    df = pd.read_excel(file_path)
                    if piece_name not in graded_data:
                        graded_data[piece_name] = []
                    graded_data[piece_name].append(df)
        
        # Load labeled data from original source
        labeled_folder = f"data/input/mtm_combined_entities_labeled/{args.item}"
        if os.path.exists(labeled_folder):
            for file in os.listdir(labeled_folder):
                if file.endswith('.xlsx'):
                    file_path = os.path.join(labeled_folder, file)
                    df = pd.read_excel(file_path)
                    labeled_df = df[df['MTM Points'].notna()]
                    
                    # Find matching piece name
                    for piece_name in graded_data.keys():
                        if piece_name in file:
                            if piece_name not in labeled_data:
                                labeled_data[piece_name] = []
                            labeled_data[piece_name].append(labeled_df)
        
        visualizer = PatternVisualizer(
            output_base_dir,
            args.item,
            labeled_data=labeled_data,
            graded_data=graded_data
        )
        
        visualizer.visualize_all()
        
    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
        raise