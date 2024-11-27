"""
Graded Plot Generator

This module generates visual plots of garment patterns with MTM (Made-to-Measure) points.
It visualizes pattern pieces with their grading rules and measurement points, creating
clear visual representations for each size variant.

The module can process either individual files or entire directories of pattern pieces,
generating high-resolution plots that show:
- Pattern outlines
- MTM measurement points
- Point labels
- Size-specific variations

Usage:
    from app.graded_plot_generator import PlotGradedMTM
    
    # For a single file
    plotter = PlotGradedMTM(
        item="shirt",
        piece_name="FFS-V2-SH-01-CCB-FO",
        single_file_mode=True,
        custom_file_path="path/to/file.xlsx"
    )
    plotter.plot_mtm_points()
    
    # For processing all files in a directory
    plotter = PlotGradedMTM(item="shirt", piece_name="FFS-V2-SH-01-CCB-FO")
    plotter.plot_mtm_points()
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import ast
import re
import numpy as np

class PlotGradedMTM:
    """
    A class to generate visual plots of graded pattern pieces with MTM points.
    
    This class handles the creation of high-resolution plots showing pattern pieces
    with their measurement points, supporting both single-file and directory-based processing.
    
    Attributes:
        item (str): The type of garment (e.g., "shirt", "pants")
        piece_name (str): The specific pattern piece identifier
        single_file_mode (bool): Whether to process a single file or entire directory
        custom_file_path (str): Path to specific file when in single_file_mode
        base_dir (str): Base directory for input files
        output_dir (str): Directory where generated plots will be saved
    """

    def __init__(self, item="shirt", piece_name="FFS-V2-SH-01-CCB-FO", 
                 single_file_mode=False, custom_file_path=None):
        """
        Initialize the PlotGradedMTM instance.
        
        Args:
            item (str): The garment type
            piece_name (str): Pattern piece identifier
            single_file_mode (bool): Process single file if True
            custom_file_path (str, optional): Path to specific file for single_file_mode
        """
        self.item = item
        self.piece_name = piece_name
        self.single_file_mode = single_file_mode
        self.custom_file_path = custom_file_path
        
        # Set up directory structure
        self.base_dir = os.path.join(
            "data/input/graded_mtm_combined_entities_labeled",
            item,
            "generated_with_grading_rule",
            piece_name
        )
        self.output_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_mtm_points(self):
        """
        Create visualizations for all sizes in the graded files.
        
        This method handles the high-level flow of plot generation, either
        processing a single file or all files in the directory based on
        the single_file_mode setting.
        
        Prints progress information and handles file selection logic.
        """
        print(f"\n📊 Generating plots for {self.piece_name}:")
        print("========================================")
        
        if self.single_file_mode:
            if not self.custom_file_path or not os.path.exists(self.custom_file_path):
                logging.error(f"Invalid custom file path: {self.custom_file_path}")
                return
                
            print(f"📂 Processing single file: {self.custom_file_path}")
            self._process_file(self.custom_file_path)
        else:
            print(f"📂 Input directory: {self.base_dir}")
            print("----------------------------------------")
            
            for file in os.listdir(self.base_dir):
                if not file.endswith('.xlsx'):
                    continue
                    
                file_path = os.path.join(self.base_dir, file)
                self._process_file(file_path)

    def _process_file(self, file_path):
        """
        Process a single file and generate its plot.
        
        This method handles the detailed plot generation for a single pattern file,
        including:
        - Loading the pattern data
        - Creating the plot with appropriate sizing and styling
        - Drawing pattern outlines
        - Adding MTM points and labels
        - Saving the generated plot
        
        Args:
            file_path (str): Path to the Excel file containing pattern data
            
        The generated plots include:
        - Blue lines for pattern outlines
        - Blue dots for pattern points
        - Red dots for MTM points
        - Numbered labels for MTM points
        - Grid lines for measurement reference
        """
        try:
            # Extract size from filename
            file_name = os.path.basename(file_path)
            size_match = re.search(r'-(\d+)\.dxf', file_name)
            if not size_match:
                return
            
            size = size_match.group(1)
            print(f"\n📄 Processing file: {file_name}")
            print(f"🔢 Size: {size}")
            
            # Create plot
            plt.figure(figsize=(200, 200))
            
            # Process and plot pattern data
            df = pd.read_excel(file_path)
            x_coords_all, y_coords_all = [], []
            
            # Plot pattern outlines
            for _, row in df.iterrows():
                if pd.notna(row['Vertices']):
                    vertices = ast.literal_eval(row['Vertices'])
                    x_coords, y_coords = zip(*vertices)
                    plt.plot(x_coords, y_coords, 'b-', linewidth=3.0)
                    x_coords_all.extend(x_coords)
                    y_coords_all.extend(y_coords)
            
            # Plot pattern points
            if 'PL_POINT_X' in df.columns and 'PL_POINT_Y' in df.columns:
                valid_points = df[pd.notna(df['PL_POINT_X']) & pd.notna(df['PL_POINT_Y'])]
                x_coords_all.extend(valid_points['PL_POINT_X'])
                y_coords_all.extend(valid_points['PL_POINT_Y'])
                plt.scatter(valid_points['PL_POINT_X'], valid_points['PL_POINT_Y'],
                          c='blue', s=800, zorder=4)
            
            # Plot MTM points and labels
            mtm_points = df[pd.notna(df['MTM Points'])]
            if not mtm_points.empty:
                x_coords_all.extend(mtm_points['PL_POINT_X'])
                y_coords_all.extend(mtm_points['PL_POINT_Y'])
                plt.scatter(mtm_points['PL_POINT_X'], mtm_points['PL_POINT_Y'], 
                          c='red', s=1500, zorder=5)
                
                for _, row in mtm_points.iterrows():
                    plt.annotate(
                        f"{int(float(row['MTM Points']))}", 
                        (row['PL_POINT_X'], row['PL_POINT_Y']),
                        xytext=(40, 40),
                        textcoords='offset points',
                        fontsize=40,
                        color='black',
                        weight='bold'
                    )
            
            # Configure plot styling and save
            self._configure_plot_style(x_coords_all, y_coords_all, size)
            
            # Save plot
            output_path = os.path.join(self.output_dir, f"{self.piece_name}_size_{size}.png")
            plt.savefig(output_path, dpi=72, bbox_inches='tight', pad_inches=0.0)
            plt.close()
            
            print(f"✅ Generated plot: {output_path}")
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

    def _configure_plot_style(self, x_coords_all, y_coords_all, size):
        """
        Configure the plot style and appearance.
        
        Args:
            x_coords_all (list): All X coordinates for setting plot bounds
            y_coords_all (list): All Y coordinates for setting plot bounds
            size (str): Size identifier for the plot title
        """
        if x_coords_all and y_coords_all:
            # Set axis limits
            x_min, x_max = min(x_coords_all), max(x_coords_all)
            y_min, y_max = min(y_coords_all), max(y_coords_all)
            x_range = x_max - x_min
            y_range = y_max - y_min
            padding_x = x_range * 0.03
            padding_y = y_range * 0.03
            
            ax = plt.gca()
            ax.set_xlim(x_min - padding_x, x_max + padding_x)
            ax.set_ylim(y_min - padding_y, y_max + padding_y)
            ax.set_aspect('equal', adjustable='box')
            
            # Configure grid
            plt.grid(True, which='major', linestyle='-', linewidth=1.5, color='gray', alpha=0.5)
            plt.grid(True, which='minor', linestyle=':', linewidth=0.8, color='gray', alpha=0.3)
            plt.minorticks_on()
            
            # Set grid intervals
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
            ax.xaxis.set_major_locator(plt.MultipleLocator(2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(2))
            
            # Configure labels and title
            plt.title(f"{self.piece_name} - Size {size}", fontsize=100, pad=40, weight='bold')
            plt.xlabel('X', fontsize=100, labelpad=20, weight='bold')
            plt.ylabel('Y', fontsize=100, labelpad=20, weight='bold')
            plt.tick_params(axis='both', which='major', labelsize=60, length=20, width=3, pad=15)

def main():
    """
    Main execution function for demonstration purposes.
    
    Creates a PlotGradedMTM instance and generates plots for a sample file.
    """
    plotter = PlotGradedMTM(
        item="shirt",
        piece_name="FFS-V2-SH-01-CCB-FO",
        single_file_mode=False,
        custom_file_path="data/input/sample/LGFG-SH-01-CCB-FOA-39.dxf_combined_entities.xlsx"
    )
    
    plotter.plot_mtm_points()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
