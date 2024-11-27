"""
Auto Grading System for Pattern Pieces

This module handles the automatic grading of pattern pieces using predefined rules
and base templates. It supports configurable sizes, items, and piece names.
"""

import pandas as pd
import numpy as np
import os
import logging
import re
import traceback
from typing import List, Dict, Optional, Tuple, Set

class GradingConfig:
    """Configuration class for grading parameters and file paths"""
    def __init__(self, 
                 item: str = "shirt",
                 piece_name: str = "FFS-V2-SH-01-CCB-FO",
                 base_size: int = 39,
                 sizes_to_generate: List[int] = None):
        """
        Initialize grading configuration.
        
        Args:
            item: The garment type (e.g., "shirt", "pants")
            piece_name: The specific pattern piece identifier
            base_size: The reference size used for grading (must be pre-labeled)
            sizes_to_generate: List of sizes to generate. If None, defaults to [38, 40]
        """
        self.item = item
        self.piece_name = piece_name
        self.base_size = base_size
        self.sizes_to_generate = sizes_to_generate or [38, 40]
        
        # Define directory structure
        self.base_dir = "data/input"
        self.graded_dir = f"{self.base_dir}/graded_mtm_combined_entities"
        self.labeled_dir = f"{self.base_dir}/graded_mtm_combined_entities_labeled"
        
    @property
    def rules_file(self) -> str:
        """Path to grading rules file"""
        return os.path.join(
            self.graded_dir,
            self.item,
            "LGFG-SH-01-CCB-FOA",
            "LGFG-SH-01-CCB-FOA-GRADE-RULE.xlsx"
        )
    
    @property
    def input_dir(self) -> str:
        """Directory containing pre-labeled files"""
        return os.path.join(
            self.graded_dir,
            self.item,
            "pre_labeled_graded_files"
        )
    
    @property
    def source_dir(self) -> str:
        """Directory containing source files"""
        return os.path.join(
            self.graded_dir,
            self.item,
            "LGFG-SH-01-CCB-FOA"
        )
    
    @property
    def target_dir(self) -> str:
        """Directory for output files"""
        return os.path.join(
            self.labeled_dir,
            self.item,
            "generated_with_grading_rule",
            self.piece_name
        )

    def validate_files(self) -> None:
        """Validate that all required input files exist"""
        # Check base template file
        base_file = os.path.join(
            self.input_dir,
            f"LGFG-SH-01-CCB-FOA-{self.base_size}.dxf_combined_entities.xlsx"
        )
        if not os.path.exists(base_file):
            raise FileNotFoundError(
                f"\nBase template file not found: {base_file}"
                f"\nPlease ensure the pre-labeled base size file exists in:"
                f"\n{self.input_dir}"
            )

        # Check grading rules file
        if not os.path.exists(self.rules_file):
            raise FileNotFoundError(
                f"\nGrading rules file not found: {self.rules_file}"
                f"\nPlease ensure the grading rules file exists."
            )

        # Check source files for sizes to generate
        for size in self.sizes_to_generate:
            source_file = os.path.join(
                self.source_dir,
                f"LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx"
            )
            if not os.path.exists(source_file):
                raise FileNotFoundError(
                    f"\nSource file not found for size {size}: {source_file}"
                    f"\nPlease ensure all source files exist in:"
                    f"\n{self.source_dir}"
                )

class GradingRules:
    """Handles the grading rules and pattern piece processing"""
    def __init__(self, config: GradingConfig):
        self.config = config
        self.rules = {}
        self.mtm_points_by_size = {}
        
    def get_measurements(self, piece_name: str, points: List[str], 
                        current_size: int, next_size: int) -> Dict:
        """
        Get measurements for specified points between sizes.
        
        Args:
            piece_name: Name of the pattern piece
            points: List of point numbers as strings
            current_size: Starting size
            next_size: Target size
            
        Returns:
            Dictionary containing delta movements for each point
        """
        measurements = {}
        break_range = f"{current_size} - {next_size}"
        
        for point in points:
            try:
                rule_data = self.rules[piece_name][point][break_range]['data']
                measurements[point] = {
                    'delta_x': float(rule_data[1]),
                    'delta_y': float(rule_data[2]),
                    'distance': float(rule_data[3])
                }
            except (KeyError, IndexError, ValueError, TypeError):
                logging.debug(f"No rule found for point {point} in range {break_range}")
                continue
                
        return measurements

    def load_rules(self):
        """Load grading rules from Excel file"""
        try:
            df = pd.read_excel(self.config.rules_file)
            
            # First row contains multiple piece/point definitions
            first_row = df.iloc[0]
            
            for col in range(len(first_row)):
                cell_value = str(first_row.iloc[col])
                if 'Piece:' in cell_value:
                    parts = cell_value.split('\n')
                    piece = re.search(r'Piece: (.+)', parts[0]).group(1).strip()
                    point = re.search(r'Point: (\d+)', parts[1]).group(1)
                    
                    break_row = None
                    for row_idx in range(len(df)):
                        if df.iloc[row_idx, col] == 'Break':
                            break_row = row_idx
                            break
                    
                    if break_row is not None:
                        # Find the Delta X and Delta Y columns
                        delta_x_col = None
                        delta_y_col = None
                        for i in range(col, col + 4):  # Look at next few columns
                            if str(df.iloc[break_row, i]) == 'Delta X':
                                delta_x_col = i
                            elif str(df.iloc[break_row, i]) == 'Delta Y':
                                delta_y_col = i
                        
                        if delta_x_col is not None and delta_y_col is not None:
                            for row_idx in range(break_row + 1, len(df)):
                                break_range = str(df.iloc[row_idx, col]).strip()
                                if not break_range or break_range == 'nan' or ' - ' not in break_range:
                                    continue
                                
                                # Store measurements in correct order
                                self.rules.setdefault(piece, {}).setdefault(point, {})[break_range] = {
                                    'data': [
                                        break_range,
                                        float(df.iloc[row_idx, delta_x_col]) if not pd.isna(df.iloc[row_idx, delta_x_col]) else 0.0,
                                        float(df.iloc[row_idx, delta_y_col]) if not pd.isna(df.iloc[row_idx, delta_y_col]) else 0.0,
                                        0.0  # Distance can be calculated later if needed
                                    ]
                                }
                    
        except Exception as e:
            logging.error(f"Error loading rules: {str(e)}")
            raise

    def calculate_scaling_factor(self, reference_point=8002, verify_with_prelabeled=True):
        """Calculate scaling factor between inches and coordinates"""
        sizes = sorted(self.mtm_points_by_size.keys())
        scaling_factors = []
        
        print(f"\n📏 Calculating Scaling Factors using Point {reference_point}:")
        print("=" * 50)
        
        for i in range(len(sizes) - 1):
            current_size = sizes[i]
            next_size = sizes[i + 1]
            
            # Get grading rule movements (in inches)
            rule_movements = self.get_measurements(
                piece_name="FFS-V2-SH-01-CCB-FO",
                points=[str(reference_point)],
                current_size=current_size,
                next_size=next_size
            )
            
            if not rule_movements or str(reference_point) not in rule_movements:
                continue
            
            rule = rule_movements[str(reference_point)]
            inch_dx = rule['delta_x']
            inch_dy = rule['delta_y']
            
            # Get actual coordinates from pre-labeled files
            current_df = pd.read_excel(os.path.join(self.config.input_dir, f"LGFG-SH-01-CCB-FOA-{current_size}.dxf_combined_entities.xlsx"))
            next_df = pd.read_excel(os.path.join(self.config.input_dir, f"LGFG-SH-01-CCB-FOA-{next_size}.dxf_combined_entities.xlsx"))
            
            current_row = current_df[current_df['MTM Points'] == float(reference_point)]
            next_row = next_df[next_df['MTM Points'] == float(reference_point)]
            
            if current_row.empty or next_row.empty:
                continue
            
            # Calculate actual coordinate differences
            coord_dx = next_row['PL_POINT_X'].iloc[0] - current_row['PL_POINT_X'].iloc[0]
            coord_dy = next_row['PL_POINT_Y'].iloc[0] - current_row['PL_POINT_Y'].iloc[0]
            
            # Calculate scaling factors
            scale_x = coord_dx / inch_dx if inch_dx != 0 else 0
            scale_y = coord_dy / inch_dy if inch_dy != 0 else 0
            
            scaling_factors.append({
                'size_range': (current_size, next_size),
                'scale_x': scale_x,
                'scale_y': scale_y,
                'inch_movement': (inch_dx, inch_dy),
                'coord_movement': (coord_dx, coord_dy)
            })
            
            print(f"\nSize Range {current_size} - {next_size}:")
            print(f"  Grading Rule Movement (inches): dx={inch_dx:.3f}, dy={inch_dy:.3f}")
            print(f"  Actual Movement (coords): dx={coord_dx:.3f}, dy={coord_dy:.3f}")
            print(f"  Scaling Factors: X={scale_x:.3f}, Y={scale_y:.3f}")
        
        return scaling_factors

    def analyze_files(self, verify_with_prelabeled=True, reference_point=8016):
        """Analyze MTM points across different sizes"""
        self.load_mtm_points_from_files()
        results = {}
        
        # ... (existing analyze_files implementation)
        
    def process_grading(self) -> None:
        """Main processing method for grading pattern pieces"""
        print("\n📝 Generating Labeled Files:")
        print("========================================")
        
        # Create output directory
        os.makedirs(self.config.target_dir, exist_ok=True)
        
        # Load base template
        base_template = pd.read_excel(os.path.join(
            self.config.input_dir,
            f"LGFG-SH-01-CCB-FOA-{self.config.base_size}.dxf_combined_entities.xlsx"
        ))
        
        # Get MTM points from base template
        mtm_points = self._get_mtm_points_from_template(base_template)
        
        print(f"Found {len(mtm_points)} MTM points in base template (size {self.config.base_size}):")
        for point in mtm_points:
            print(f"  Point {point['point']}: ({point['x']:.3f}, {point['y']:.3f})")
        
        # Process each size
        for size in self.config.sizes_to_generate:
            source_file = os.path.join(
                self.config.source_dir,
                f"LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx"
            )
            target_file = os.path.join(
                self.config.target_dir,
                f"LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx"
            )
            
            self._process_size(size, source_file, target_file, mtm_points)

    def _process_size(self, size: int, source_file: str, target_file: str, mtm_points: List[Dict]) -> None:
        """Process a single size"""
        print(f"\nProcessing size {size}:")
        df = pd.read_excel(source_file)
        used_indices = set()
        
        # Process each point
        for point_data in mtm_points:
            mtm_point = point_data['point']
            base_x = point_data['x']
            base_y = point_data['y']
            
            x, y = self._calculate_point_position(
                size, mtm_point, base_x, base_y
            )
            
            # Find and mark closest point
            closest_idx = find_closest_unused_point(df, x, y, used_indices)
            df.at[closest_idx, 'MTM Points'] = mtm_point
            used_indices.add(closest_idx)
            
            print(f"  Added point {mtm_point} at position ({x:.3f}, {y:.3f})")
        
        self._verify_and_save(df, mtm_points, target_file)

    def _calculate_point_position(self, size: int, mtm_point: float, 
                                base_x: float, base_y: float) -> Tuple[float, float]:
        """Calculate the position for a point based on size and grading rules"""
        if size == self.config.base_size:
            return base_x, base_y
            
        # Get grading rule
        rule_movements = self.get_measurements(
            piece_name=self.config.piece_name,
            points=[str(int(mtm_point))],
            current_size=min(size, self.config.base_size),
            next_size=max(size, self.config.base_size)
        )
        
        if rule_movements and str(int(mtm_point)) in rule_movements:
            rule = rule_movements[str(int(mtm_point))]
            if size < self.config.base_size:
                return base_x - rule['delta_x'], base_y - rule['delta_y']
            else:
                return base_x + rule['delta_x'], base_y + rule['delta_y']
        
        return base_x, base_y

    def _verify_and_save(self, df: pd.DataFrame, mtm_points: List[Dict], 
                        target_file: str) -> None:
        """Verify point transfer and save results"""
        transferred_points = sorted(df[df['MTM Points'].notna()]['MTM Points'].unique())
        print("\nVerification:")
        print(f"  Expected points: {[p['point'] for p in mtm_points]}")
        print(f"  Transferred points: {transferred_points}")
        
        if len(transferred_points) != len(mtm_points):
            print("⚠️ WARNING: Not all points were transferred!")
        
        df.to_excel(target_file, index=False)
        print(f"✅ Generated: {target_file}")

    def _get_mtm_points_from_template(self, template: pd.DataFrame) -> List[Dict]:
        """
        Extract MTM points from base template.
        
        Args:
            template: DataFrame containing the base template data
            
        Returns:
            List of dictionaries containing point data, sorted by point number
        """
        mtm_points = []
        for idx, row in template.iterrows():
            if pd.notna(row['MTM Points']):
                mtm_points.append({
                    'point': float(row['MTM Points']),
                    'x': row['PL_POINT_X'],
                    'y': row['PL_POINT_Y'],
                    'idx': idx
                })
        
        # Sort points numerically
        return sorted(mtm_points, key=lambda x: x['point'])

def find_closest_unused_point(df: pd.DataFrame, target_x: float, 
                            target_y: float, used_indices: Set[int]) -> int:
    """Find closest unused point in the DataFrame"""
    distances = np.sqrt(
        (df['PL_POINT_X'] - target_x)**2 + 
        (df['PL_POINT_Y'] - target_y)**2
    )
    
    distances[list(used_indices)] = np.inf
    return distances.idxmin()

def main(item: str = "shirt",
         piece_name: str = "FFS-V2-SH-01-CCB-FO",
         base_size: int = 39,
         sizes_to_generate: List[int] = None) -> None:
    """
    Main execution function for the grading system.
    """
    try:
        # Initialize configuration
        config = GradingConfig(
            item=item,
            piece_name=piece_name,
            base_size=base_size,
            sizes_to_generate=sizes_to_generate
        )
        
        # Validate input files exist
        print("\n📂 Validating input files...")
        config.validate_files()
        
        # Initialize and process grading
        grading = GradingRules(config)
        grading.load_rules()
        grading.process_grading()
        
    except FileNotFoundError as e:
        logging.error(f"Missing required files: {str(e)}")
        print("\nRequired directory structure:")
        print("data/input/graded_mtm_combined_entities/")
        print("├── shirt/")
        print("│   ├── LGFG-SH-01-CCB-FOA/")
        print("│   │   ├── LGFG-SH-01-CCB-FOA-38.dxf_combined_entities.xlsx")
        print("│   │   ├── LGFG-SH-01-CCB-FOA-40.dxf_combined_entities.xlsx")
        print("│   │   └── LGFG-SH-01-CCB-FOA-GRADE-RULE.xlsx")
        print("│   └── pre_labeled_graded_files/")
        print("│       └── LGFG-SH-01-CCB-FOA-39.dxf_combined_entities.xlsx")
        raise
    except Exception as e:
        logging.error(f"Error in grading process: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base_size = 39
    sizes_to_generate = [38, 40]
    # Pass as named argument
    main(base_size=base_size, sizes_to_generate=sizes_to_generate)