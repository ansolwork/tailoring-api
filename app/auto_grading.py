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
from tabulate import tabulate  # Add to requirements.txt if not present

class GradingConfig:
    """Configuration class for grading parameters and file paths"""
    def __init__(self, 
                 item: str = "shirt",
                 piece_name: str = "LGFG-SH-01-CCB-FOA",
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
    def grade_rule_piece(self) -> str:
        """Get the piece name to use for grading rules"""
        return self.piece_name
        
    @property
    def rules_file(self) -> str:
        """Path to grading rules file"""
        return os.path.join(
            self.graded_dir,
            self.item,
            self.piece_name,
            f"{self.piece_name}-GRADE-RULE.xlsx"
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
            self.piece_name
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
            f"{self.piece_name}-{self.base_size}.dxf_combined_entities.xlsx"
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
                f"{self.piece_name}-{size}.dxf_combined_entities.xlsx"
            )
            if not os.path.exists(source_file):
                raise FileNotFoundError(
                    f"\nSource file not found for size {size}: {source_file}"
                    f"\nPlease ensure all source files exist in:"
                    f"\n{self.source_dir}"
                )

class GradingRules:
    """Handles the grading rules and pattern piece processing"""
    def __init__(self, item="shirt", piece_name="LGFG-SH-01-CCB-FOA"):
        self.item = item
        self.piece_name = piece_name
        self.rules_file = os.path.join(
            "data/input/graded_mtm_combined_entities",
            item,
            "LGFG-SH-01-CCB-FOA",
            "LGFG-SH-01-CCB-FOA-GRADE-RULE.xlsx"
        )
        self.input_dir = os.path.join(
            "data/input/graded_mtm_combined_entities",
            item,
            "pre_labeled_graded_files"
        )
        self.rules = {}
        self.mtm_points_by_size = {}
        
    def get_measurements(self, piece_name: str, points: List[str], 
                        current_size: int, next_size: int) -> Dict:
        """Get measurements for specified points between sizes."""
        measurements = {}
        break_range = f"{current_size} - {next_size}"

        for point in points:
            try:
                # Convert point to string to match the format in rules
                point_str = str(int(float(point)))
                
                if piece_name not in self.rules:
                    continue
                    
                if point_str not in self.rules[piece_name]:
                    continue
                    
                if break_range not in self.rules[piece_name][point_str]:
                    continue
                    
                rule_data = self.rules[piece_name][point_str][break_range]
                measurements[point_str] = {
                    'delta_x': float(rule_data['dx']),
                    'delta_y': float(rule_data['dy'])
                }
                
            except (KeyError, IndexError, ValueError, TypeError) as e:
                print(f"Debug: Error processing rule for point {point}: {str(e)}")
                continue
            
        return measurements

    def load_rules(self):
        """Load grading rules from Excel file"""
        try:
            print("\n📋 Loading Grading Rules:")
            print("=" * 50)
            print(f"Reading file: {self.rules_file}")
            
            # Check if file exists
            if not os.path.exists(self.rules_file):
                raise FileNotFoundError(f"Grade rule file not found: {self.rules_file}")
                
            # Load Excel file
            df = pd.read_excel(self.rules_file)
            
            # First row contains multiple piece/point definitions
            first_row = df.iloc[0]
            print("\nAnalyzing first row for piece/point definitions:")
            
            for col in range(len(first_row)):
                cell_value = str(first_row.iloc[col])
                
                if 'Piece:' in cell_value:
                    parts = cell_value.split('\n')
                    piece = re.search(r'Piece: (.+)', parts[0]).group(1).strip()
                    point = re.search(r'Point: (\d+)', parts[1]).group(1)

                    # Find the Break row
                    break_row = None
                    for row_idx in range(len(df)):
                        if df.iloc[row_idx, col] == 'Break':
                            break_row = row_idx
                            break
                    
                    if break_row is not None:                        
                        # Find Delta X and Delta Y columns
                        delta_cols = {}
                        for i in range(col, min(col + 4, len(df.columns))):
                            col_value = str(df.iloc[break_row, i]).strip()
                            if col_value == 'Delta X':
                                delta_cols['x'] = i
                            elif col_value == 'Delta Y':
                                delta_cols['y'] = i
                        
                        # Process each row after Break
                        for row_idx in range(break_row + 1, len(df)):
                            break_range = str(df.iloc[row_idx, col]).strip()
                            if not break_range or break_range == 'nan' or ' - ' not in break_range:
                                continue
                            
                            delta_x = float(df.iloc[row_idx, delta_cols['x']]) if not pd.isna(df.iloc[row_idx, delta_cols['x']]) else 0.0
                            delta_y = float(df.iloc[row_idx, delta_cols['y']]) if not pd.isna(df.iloc[row_idx, delta_cols['y']]) else 0.0
                                                        
                            self.rules.setdefault(piece, {}).setdefault(point, {})[break_range] = {
                                'dx': delta_x,
                                'dy': delta_y
                            }
                        
        except Exception as e:
            logging.error(f"Error loading rules: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def calculate_scaling_factor(self, reference_point=8002, verify_with_prelabeled=True):
        """Calculate scaling factor between inches and coordinates for multiple points"""
        sizes = sorted(self.mtm_points_by_size.keys())
        scaling_factors = []
        
        print("\n📏 Calculating Scaling Factors:")
        print("=" * 80)
        
        for i in range(len(sizes) - 1):
            current_size = sizes[i]
            next_size = sizes[i + 1]
            
            print(f"\n🔍 Size Range {current_size} - {next_size}:")
            print("-" * 80)
            print(f"{'Point':^8} | {'Grading Rule (inches)':^25} | {'Actual Movement':^25} | {'Scaling Factors':^25}")
            print("-" * 80)
            
            try:
                # Get actual coordinates from pre-labeled files
                current_df = pd.read_excel(os.path.join(self.config.input_dir, 
                    f"{self.config.piece_name}-{current_size}.dxf_combined_entities.xlsx"))
                next_df = pd.read_excel(os.path.join(self.config.input_dir, 
                    f"{self.config.piece_name}-{next_size}.dxf_combined_entities.xlsx"))
                
                for point in [reference_point]:
                    # Get grading rule movements (in inches)
                    rule_movements = self.get_measurements(
                        piece_name="LGFG-SH-01-CCB-FOA",
                        points=[str(point)],
                        current_size=current_size,
                        next_size=next_size
                    )
                    
                    if not rule_movements or str(point) not in rule_movements:
                        print(f"{point:^8} | {'No grading rule found':^77}")
                        continue
                    
                    rule = rule_movements[str(point)]
                    inch_dx = rule['delta_x']
                    inch_dy = rule['delta_y']
                    
                    # Get point coordinates
                    current_row = current_df[current_df['MTM Points'] == float(point)]
                    next_row = next_df[next_df['MTM Points'] == float(point)]
                    
                    if current_row.empty or next_row.empty:
                        print(f"{point:^8} | {'Point not found in files':^77}")
                        continue
                    
                    # Calculate actual coordinate differences
                    coord_dx = next_row['PL_POINT_X'].iloc[0] - current_row['PL_POINT_X'].iloc[0]
                    coord_dy = next_row['PL_POINT_Y'].iloc[0] - current_row['PL_POINT_Y'].iloc[0]
                    
                    # Calculate scaling factors
                    scale_x = coord_dx / inch_dx if inch_dx != 0 else 0
                    scale_y = coord_dy / inch_dy if inch_dy != 0 else 0
                    
                    # Print formatted row
                    print(f"{point:^8} | dx={inch_dx:6.3f}, dy={inch_dy:6.3f} | "
                          f"dx={coord_dx:6.3f}, dy={coord_dy:6.3f} | "
                          f"X={scale_x:6.3f}, Y={scale_y:6.3f}")
                    
                    scaling_factors.append({
                        'point': point,
                        'size_range': (current_size, next_size),
                        'scale_x': scale_x,
                        'scale_y': scale_y,
                        'inch_movement': (inch_dx, inch_dy),
                        'coord_movement': (coord_dx, coord_dy)
                    })
                    
            except FileNotFoundError:
                print(f"❌ Required files not found for size range {current_size}-{next_size}")
                continue
            except Exception as e:
                print(f"❌ Error processing size range {current_size}-{next_size}: {str(e)}")
                continue
        
        print("\n" + "=" * 80)
        return scaling_factors

    def analyze_files(self, verify_with_prelabeled=True, reference_point=8016):
        """Analyze MTM points across different sizes"""
        self.load_mtm_points_from_files()
        results = {}
        
        # ... (existing analyze_files implementation)
        
    def process_grading(self) -> None:
        """Main processing method for grading pattern pieces"""
        print("\n📝 Grading Process:")
        print("========================================")
        
        # Load base template first
        base_template = pd.read_excel(os.path.join(
            self.config.input_dir,
            f"{self.config.piece_name}-{self.config.base_size}.dxf_combined_entities.xlsx"
        ))
        
        # Get MTM points from base template
        mtm_points = self._get_mtm_points_from_template(base_template)
        
        # Sort sizes and validate they can be processed sequentially
        all_sizes = sorted(self.config.sizes_to_generate)
        base_size = self.config.base_size
        
        # Split sizes into those smaller and larger than base size
        smaller_sizes = sorted([s for s in all_sizes if s < base_size], reverse=True)
        larger_sizes = sorted([s for s in all_sizes if s > base_size])
        
        # Process sizes in the correct order
        processed_sizes = []
        
        # Process sizes smaller than base size (in descending order)
        current_size = base_size
        for target_size in smaller_sizes:
            print(f"\n🔄 Processing size {target_size} from size {current_size}:")
            print("----------------------------------------")
            
            source_file = os.path.join(
                self.config.source_dir,
                f"{self.config.piece_name}-{current_size}.dxf_combined_entities.xlsx"
            )
            target_file = os.path.join(
                self.config.target_dir,
                f"{self.config.piece_name}-{target_size}.dxf_combined_entities.xlsx"
            )
            
            self._process_size(target_size, source_file, target_file, mtm_points)
            processed_sizes.append(target_size)
            current_size = target_size
        
        # Process sizes larger than base size (in ascending order)
        current_size = base_size
        for target_size in larger_sizes:
            print(f"\n🔄 Processing size {target_size} from size {current_size}:")
            print("----------------------------------------")
            
            source_file = os.path.join(
                self.config.source_dir,
                f"{self.config.piece_name}-{current_size}.dxf_combined_entities.xlsx"
            )
            target_file = os.path.join(
                self.config.target_dir,
                f"{self.config.piece_name}-{target_size}.dxf_combined_entities.xlsx"
            )
            
            self._process_size(target_size, source_file, target_file, mtm_points)
            processed_sizes.append(target_size)
            current_size = target_size
        
        print(f"\n✅ Successfully processed sizes: {processed_sizes}")

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
            
            # Debug print the grading rules for this point
            print(f"\nDebug: Looking up grading rules for point {mtm_point}")
            
            # Get the grading rule movements for this point
            source_size = size - 1  # The size we're grading from
            movements = self.get_measurements(
                piece_name="LGFG-SH-01-CCB-FOA",
                points=[str(int(mtm_point))],
                current_size=source_size,
                next_size=size
            )
            
            # Debug print the movements found
            print(f"Debug: Movements found for {source_size} -> {size}:")
            if movements and str(int(mtm_point)) in movements:
                rule = movements[str(int(mtm_point))]
                print(f"  dx={rule['delta_x']:.3f}, dy={rule['delta_y']:.3f}")
            else:
                print("  No movements found!")
            
            # Calculate new position
            x, y = self._calculate_point_position(size, mtm_point, base_x, base_y)
            
            # Get dx and dy from movements
            dx = dy = 0.0
            if movements and str(int(mtm_point)) in movements:
                dx = movements[str(int(mtm_point))]['delta_x']
                dy = movements[str(int(mtm_point))]['delta_y']
            
            # Find and mark closest point
            closest_idx = find_closest_unused_point(df, x, y, used_indices)
            df.at[closest_idx, 'MTM Points'] = mtm_point
            used_indices.add(closest_idx)
                    
        self._verify_and_save(df, mtm_points, target_file)

    def _calculate_point_position(self, size: int, mtm_point: float, 
                                base_x: float, base_y: float) -> Tuple[float, float]:
        """
        Calculate the position for a point based on size and grading rules
        
        Args:
            size: The size we want to calculate position for
            mtm_point: The MTM point number
            base_x: X coordinate in base size
            base_y: Y coordinate in base size
            
        Returns:
            Tuple of (x, y) coordinates for the target size
        """
        if size == self.config.base_size:
            return base_x, base_y
            
        # Determine if we're going up or down in sizes
        going_up = size > self.config.base_size
        
        # Start from base size
        current_x = base_x
        current_y = base_y
        current_size = self.config.base_size
        
        # Calculate position incrementally
        while current_size != size:
            next_size = current_size + (1 if going_up else -1)
            
            # Get movements for this size step
            size_range = f"{min(current_size, next_size)} - {max(current_size, next_size)}"
            movements = self.get_measurements(
                piece_name="LGFG-SH-01-CCB-FOA",
                points=[str(int(mtm_point))],
                current_size=min(current_size, next_size),
                next_size=max(current_size, next_size)
            )
            
            print(f"Debug: Calculating {current_size} -> {next_size} for point {mtm_point}")
            
            if movements and str(int(mtm_point)) in movements:
                dx = movements[str(int(mtm_point))]['delta_x']
                dy = movements[str(int(mtm_point))]['delta_y']
                
                # Apply movement in correct direction
                if going_up:
                    current_x += dx
                    current_y += dy
                else:
                    current_x -= dx
                    current_y -= dy
                    
                print(f"  Movement: dx={dx:.3f}, dy={dy:.3f}")
                print(f"  New position: ({current_x:.3f}, {current_y:.3f})")
            else:
                print(f"  Warning: No movement found for {size_range}")
                
            current_size = next_size
        
        return current_x, current_y

    def _verify_and_save(self, df: pd.DataFrame, mtm_points: List[Dict], 
                        target_file: str) -> None:
        """Verify point transfer and save results"""
        transferred_points = sorted(df[df['MTM Points'].notna()]['MTM Points'].unique())
        
        if len(transferred_points) != len(mtm_points):
            print("⚠️ WARNING: Not all points were transferred!")
        
        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
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

    def get_available_piece_names(self) -> List[str]:
        """Get list of available piece names from grading rules"""
        # Return only the configured piece name instead of reading from Excel
        return [self.config.piece_name]

def find_closest_unused_point(df: pd.DataFrame, target_x: float, 
                            target_y: float, used_indices: Set[int]) -> int:
    """Find closest unused point in the DataFrame"""
    distances = np.sqrt(
        (df['PL_POINT_X'] - target_x)**2 + 
        (df['PL_POINT_Y'] - target_y)**2
    )
    
    distances[list(used_indices)] = np.inf
    return distances.idxmin()

def print_points_summary(size, points):
    print(f"\n📏 Size {size}:")
    print("----------------------------------------")
    
    # Print total points added
    print(f"✓ Added {len(points)} points")
    
    # Print first few and last few points as sample
    sample_size = 3  # Number of points to show at start/end
    if len(points) > sample_size * 2:
        # Print first few points
        for i in range(sample_size):
            point = points[i]
            print(f"  • Point {point.id:.0f}: ({point.x:.3f}, {point.y:.3f})")
        
        print(f"  ... {len(points) - sample_size * 2} more points ...")
        
        # Print last few points
        for i in range(-sample_size, 0):
            point = points[i]
            print(f"  • Point {point.id:.0f}: ({point.x:.3f}, {point.y:.3f})")
    else:
        # If few points, print all
        for point in points:
            print(f"  • Point {point.id:.0f}: ({point.x:.3f}, {point.y:.3f})")

def print_grading_changes(from_size, to_size, points_data):
    print(f"\n🔄 Size {from_size} → {to_size}:")
    print("----------------------------------------")
    print(f"✓ Processing {len(points_data)} points")
    
    # Group points by their movement patterns
    movement_groups = {}
    for point_id, data in points_data.items():
        key = (data['dx'], data['dy'])
        if key not in movement_groups:
            movement_groups[key] = []
        movement_groups[key].append(point_id)
    
    # Print grouped movements
    for (dx, dy), point_ids in movement_groups.items():
        if dx == 0 and dy == 0:
            continue  # Skip printing static points
            
        # Format point ranges (e.g., "8000-8009, 8050")
        ranges = []
        start = end = point_ids[0]
        for pid in point_ids[1:]:
            if pid == end + 1:
                end = pid
            else:
                ranges.append(f"{start}" if start == end else f"{start}-{end}")
                start = end = pid
        ranges.append(f"{start}" if start == end else f"{start}-{end}")
        
        print(f"  • Points {', '.join(ranges)}:")
        print(f"    Δx: {dx:+.3f}, Δy: {dy:+.3f}")

    print(f"\n✅ Generated: {output_file}")

def extract_mtm_points(df):
    """Extract MTM points and their coordinates from the dataframe"""
    mtm_points = {}
    
    # Filter rows that have MTM Points
    mtm_rows = df[df['MTM Points'].notna()]
    
    for _, row in mtm_rows.iterrows():
        point_num = str(int(row['MTM Points']))  # Convert to string for consistent keys
        mtm_points[point_num] = {
            'x': row['PL_POINT_X'],
            'y': row['PL_POINT_Y']
        }
    
    if not mtm_points:
        raise ValueError("No MTM points found in the input file")
        
    print(f"Extracted {len(mtm_points)} MTM points: {sorted(mtm_points.keys())}")
    return mtm_points

def get_grading_rules(rules_df, point_num, from_size, to_size):
    """Extract dx and dy for a specific point and size range"""
    total_dx = 0.0
    total_dy = 0.0
    
    # Process each single-size increment
    current_size = from_size
    while current_size < to_size:
        # Format as single-size increment (e.g. "39 - 40")
        range_str = f"{current_size} - {current_size + 1}"

        # Find rule for this specific range
        rule = rules_df[
            (rules_df['Point'] == float(point_num)) & 
            (rules_df['Break'] == range_str)
        ]
        
        if not rule.empty:
            dx = float(rule['Delta X'].iloc[0])
            dy = float(rule['Delta Y'].iloc[0])
            total_dx += dx
            total_dy += dy
            print(f"Debug: Found movement for {range_str}: dx={dx}, dy={dy}")
        else:
            print(f"Debug: No rule found for range {range_str}")
            
        current_size += 1
    
    print(f"Debug: Total accumulated movement: dx={total_dx}, dy={total_dy}")
    return total_dx, total_dy

def process_size(base_points, rules_df, from_size, to_size):
    """Process points for a specific size"""
    new_points = {}
    
    for point_num in base_points:
        # Get original coordinates
        x = base_points[point_num]['x']
        y = base_points[point_num]['y']
        
        # Get cumulative movement for this point
        dx, dy = get_grading_rules(rules_df, point_num, from_size, to_size)
        
        # Apply the movement
        new_x = x + dx
        new_y = y + dy
        
        new_points[point_num] = {
            'x': new_x,
            'y': new_y,
            'dx': dx,
            'dy': dy
        }
            
    return new_points

def print_grading_summary(piece_name, base_size, target_sizes, points_data):
    """Print a formatted summary of grading rules application"""
    print(f"\n📊 Grading Summary for {piece_name}")
    print(f"Base Size: {base_size}")
    print(f"Target Sizes: {target_sizes}\n")
    
    for target_size in target_sizes:
        headers = ["Point", "Original (x,y)", f"Size {target_size} (x,y)", "Movement (dx,dy)"]
        rows = []
        
        # Assuming points_data structure contains results for each size
        size_data = points_data.get(target_size, {})
        for point_num, data in sorted(size_data.items()):
            rows.append([
                f"{point_num:>4}",
                f"({data['original_x']:.3f}, {data['original_y']:.3f})",
                f"({data['new_x']:.3f}, {data['new_y']:.3f})",
                f"({data['dx']:.3f}, {data['dy']:.3f})"
            ])
        
        print(f"\n🔍 Size {target_size} Adjustments:")
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Optional: Print summary statistics
        total_points = len(rows)
        points_with_movement = sum(1 for r in rows if float(r[3].split(',')[0][1:]) != 0 or float(r[3].split(',')[1][:-1]) != 0)
        
        print(f"\nSummary:")
        print(f"- Total Points: {total_points}")
        print(f"- Points Modified: {points_with_movement}")
        print(f"- Movement Rate: {(points_with_movement/total_points)*100:.1f}%")
        print("\n" + "="*80)

def main(item="shirt", piece_name="LGFG-SH-01-CCB-FOA", base_size=39, sizes_to_generate=None):
    # Initialize and load rules
    grading = GradingRules(item=item, piece_name=piece_name)
    grading.load_rules()
    
    print("\n📝 Generating Labeled Files:")
    print("========================================")
    
    # Define directories
    source_dir = os.path.join(
        "data/input/graded_mtm_combined_entities",
        item,
        piece_name
    )
    target_dir = os.path.join(
        "data/input/graded_mtm_combined_entities_labeled",
        item,
        "generated_with_grading_rule",
        piece_name
    )
    os.makedirs(target_dir, exist_ok=True)
    
    # Load base template (pre-labeled base size)
    base_file = f"data/input/graded_mtm_combined_entities/shirt/pre_labeled_graded_files/{piece_name}-{base_size}.dxf_combined_entities.xlsx"
    base_template = pd.read_excel(base_file)
    
    # Get ALL MTM points from base template, sorted numerically
    mtm_points = []
    for idx, row in base_template.iterrows():
        if pd.notna(row['MTM Points']):
            mtm_points.append({
                'point': float(row['MTM Points']),
                'x': row['PL_POINT_X'],
                'y': row['PL_POINT_Y'],
                'idx': idx
            })
    
    # Sort points numerically
    mtm_points.sort(key=lambda x: x['point'])
    
    # Process each size
    sizes = sizes_to_generate or [base_size - 1, base_size, base_size + 1]
    processed_sizes = []
    
    points_data = {}  # Store results for summary
    
    for target_size in sizes:
        if target_size == base_size:
            continue  # Skip base size as it's already labeled
            
        source_file = os.path.join(source_dir, f"{piece_name}-{target_size}.dxf_combined_entities.xlsx")
        target_file = os.path.join(target_dir, f"{piece_name}-{target_size}.dxf_combined_entities.xlsx")
        
        print(f"\n🔄 Processing size {target_size} from size {base_size}:")
        print("-" * 40)
        
        print(f"\nProcessing size {target_size}:")
        df = pd.read_excel(source_file)
        
        # Initialize points data for this size
        points_data[target_size] = {}
        
        # Keep track of used points
        used_indices = set()
        
        # Process each point from base template
        for point_data in mtm_points:
            mtm_point = point_data['point']
            base_x = point_data['x']
            base_y = point_data['y']
            
            # Get grading rule
            rule_movements = grading.get_measurements(
                piece_name="LGFG-SH-01-CCB-FOA",
                points=[str(int(mtm_point))],
                current_size=min(target_size, base_size),
                next_size=max(target_size, base_size)
            )
            
            if rule_movements and str(int(mtm_point)) in rule_movements:
                # Calculate new position using grading rule
                rule = rule_movements[str(int(mtm_point))]
                if target_size < base_size:
                    x = base_x - rule['delta_x']
                    y = base_y - rule['delta_y']
                else:
                    x = base_x + rule['delta_x']
                    y = base_y + rule['delta_y']
            else:
                # No grading rule - use base position
                x, y = base_x, base_y
            
            # Store point data for summary
            points_data[target_size][mtm_point] = {
                'original_x': base_x,
                'original_y': base_y,
                'new_x': x,
                'new_y': y,
                'dx': x - base_x,
                'dy': y - base_y
            }
            
            # Find closest unused point and update df
            closest_idx = find_closest_unused_point(df, x, y, used_indices)
            df.at[closest_idx, 'MTM Points'] = mtm_point
            used_indices.add(closest_idx)
                    
        # Save file
        df.to_excel(target_file, index=False)
        print(f"✅ Generated: {target_file}")
        processed_sizes.append(target_size)
    
    print_grading_summary(piece_name, base_size, processed_sizes, points_data)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Set inputs directly
    item = "shirt"
    piece_name = "LGFG-SH-01-CCB-FOA"
    base_size = 39
    min_size = 38
    max_size = 40
    sizes_to_generate = list(range(min_size, max_size))  
    
    # Call main with all inputs
    main(
        item=item,
        piece_name=piece_name,
        base_size=base_size,
        sizes_to_generate=sizes_to_generate
    )