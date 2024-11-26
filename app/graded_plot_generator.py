import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import ast
import re
import numpy as np

class PlotGradedMTM:
    def __init__(self, item="shirt", piece_name="FFS-V2-SH-01-CCB-FO", single_file_mode=False, custom_file_path=None):
        self.item = item
        self.piece_name = piece_name
        self.single_file_mode = single_file_mode
        self.custom_file_path = custom_file_path
        
        # Original directory setup
        self.base_dir = os.path.join(
            "data/input/graded_mtm_combined_entities_labeled",
            item,
            "generated_with_grading_rule",
            piece_name
        )
        self.output_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_mtm_points(self):
        """Create visualizations for all sizes in the graded files"""
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
        """Process a single file and generate its plot"""
        try:
            file_name = os.path.basename(file_path)
            size_match = re.search(r'-(\d+)\.dxf', file_name)
            if not size_match:
                return
            
            size = size_match.group(1)
            print(f"\n📄 Processing file: {file_name}")
            print(f"🔢 Size: {size}")
            
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # Create a much larger figure
            plt.figure(figsize=(200, 200))
            
            # Initialize lists to collect ALL coordinates
            x_coords_all = []
            y_coords_all = []
            
            # Plot the polygon with thicker, more visible lines
            for _, row in df.iterrows():
                if pd.notna(row['Vertices']):
                    vertices = ast.literal_eval(row['Vertices'])
                    x_coords, y_coords = zip(*vertices)
                    plt.plot(x_coords, y_coords, 'b-', linewidth=3.0)
                    x_coords_all.extend(x_coords)
                    y_coords_all.extend(y_coords)
            
            # Plot PL_POINT coordinates with larger, more visible points
            if 'PL_POINT_X' in df.columns and 'PL_POINT_Y' in df.columns:
                valid_points = df[pd.notna(df['PL_POINT_X']) & pd.notna(df['PL_POINT_Y'])]
                x_coords_all.extend(valid_points['PL_POINT_X'])
                y_coords_all.extend(valid_points['PL_POINT_Y'])
                plt.scatter(valid_points['PL_POINT_X'], valid_points['PL_POINT_Y'],
                          c='blue', s=800, zorder=4)
            
            # Plot MTM points with even larger markers
            mtm_points = df[pd.notna(df['MTM Points'])]
            if not mtm_points.empty:
                x_coords_all.extend(mtm_points['PL_POINT_X'])
                y_coords_all.extend(mtm_points['PL_POINT_Y'])
                plt.scatter(mtm_points['PL_POINT_X'], mtm_points['PL_POINT_Y'], 
                          c='red', s=1500, zorder=5)
                
                # Larger, more visible labels
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
            
            # Set axis limits tightly around the data
            if x_coords_all and y_coords_all:
                x_min, x_max = min(x_coords_all), max(x_coords_all)
                y_min, y_max = min(y_coords_all), max(y_coords_all)
                
                # Calculate the range
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Add slightly more padding (1.5% of the range)
                padding_x = x_range * 0.03
                padding_y = y_range * 0.03
                
                # Get current axis
                ax = plt.gca()
                
                # Set limits with minimal padding
                ax.set_xlim(x_min - padding_x, x_max + padding_x)
                ax.set_ylim(y_min - padding_y, y_max + padding_y)
                
                # Ensure aspect ratio is equal AFTER setting limits
                ax.set_aspect('equal', adjustable='box')
            
            # Much finer grid
            plt.grid(True, which='major', linestyle='-', linewidth=1.5, color='gray', alpha=0.5)
            plt.grid(True, which='minor', linestyle=':', linewidth=0.8, color='gray', alpha=0.3)
            plt.minorticks_on()
            
            # Even finer grid intervals
            ax = plt.gca()
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
            ax.xaxis.set_major_locator(plt.MultipleLocator(2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(2))
            
            # Title and labels
            plt.title(f"{self.piece_name} - Size {size}", 
                     fontsize=100, 
                     pad=40,
                     weight='bold')
            
            plt.xlabel('X', 
                     fontsize=100,
                     labelpad=20,
                     weight='bold')
            
            plt.ylabel('Y', 
                     fontsize=100,
                     labelpad=20,
                     weight='bold')
            
            # Add larger tick labels
            plt.tick_params(axis='both', 
                          which='major',
                          labelsize=60,
                          length=20,
                          width=3,
                          pad=15)
            
            # Define output path
            output_path = os.path.join(self.output_dir, f"{self.piece_name}_size_{size}.png")
            
            # Save with minimal padding
            plt.savefig(output_path, 
                      dpi=72, 
                      bbox_inches='tight',
                      pad_inches=0.0)  # No padding
            plt.close()
            
            print(f"✅ Generated plot: {output_path}")
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

def main():
    # Example with your specific file
    plotter = PlotGradedMTM(
        item="shirt",
        piece_name="FFS-V2-SH-01-CCB-FO",
        single_file_mode=True,
        custom_file_path="data/input/sample/LGFG-SH-01-CCB-FOA-39.dxf_combined_entities.xlsx"
    )
    
    plotter.plot_mtm_points()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
