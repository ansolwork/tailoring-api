import pandas as pd
import logging
import os
import re
import numpy as np
import traceback

class GradingRules:
    def __init__(self, item="shirt"):
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
        
    def extract_size_from_filename(self, filename):
        """Extract size from filename like LGFG-SH-01-CCB-FOA-38.dxf_combined_entities.xlsx"""
        match = re.search(r'FOA-(\d+)\.dxf', filename)
        if match:
            return int(match.group(1))
        return None
        
    def extract_size_from_text(self, df):
        """Extract size from text column containing format '30 - 62 (38)'"""
        size_rows = df['Text'].str.contains(r'\(\d+\)', na=False)
        if size_rows.any():
            text = df[size_rows]['Text'].iloc[0]
            match = re.search(r'\((\d+)\)', text)
            if match:
                return int(match.group(1))
        return None

    def load_mtm_points_from_files(self):
        """Load MTM points from pre-labeled files first"""
        self.mtm_points_by_size = {}
        
        # Change directory to pre-labeled files
        pre_labeled_dir = "data/input/graded_mtm_combined_entities/shirt/pre_labeled_graded_files"
        logging.info(f"Looking for files in: {pre_labeled_dir}")
        
        # Get all available sizes from pre-labeled files
        for file in os.listdir(pre_labeled_dir):
            if file.endswith('.xlsx'):
                # Extract size from filename (e.g., LGFG-SH-01-CCB-FOA-39.dxf_combined_entities.xlsx)
                size_match = re.search(r'-(\d+)\.dxf', file)
                if size_match:
                    size = int(size_match.group(1))
                    file_path = os.path.join(pre_labeled_dir, file)
                    
                    logging.info(f"\nProcessing file: {file_path}")
                    df = pd.read_excel(file_path)
                    
                    # Get MTM points for this size
                    mtm_points = df[df['MTM Points'].notna()]['MTM Points'].unique()
                    self.mtm_points_by_size[size] = sorted(mtm_points)
                    logging.info(f"Found {len(mtm_points)} MTM points for size {size}")

    def load_rules(self):
        """Load grading rules from Excel file"""
        try:
            df = pd.read_excel(self.rules_file)
            
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

    def get_measurements(self, piece_name, points, current_size, next_size):
        """Get measurements for specified points between sizes"""
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
                logging.warning(f"No measurements found for point {point}")
                
        return measurements

    def get_point_coordinates(self, point, size):
        """Get coordinates for a point from pre-labeled file"""
        file_path = f"data/input/graded_mtm_combined_entities/shirt/pre_labeled_graded_files/LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx"
        
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_excel(file_path)
        point_row = df[df['MTM Points'] == float(point)]
        
        if point_row.empty:
            return None
        
        return (point_row['PL_POINT_X'].iloc[0], point_row['PL_POINT_Y'].iloc[0])

    def calculate_scaling_factor(self, reference_point=8002, verify_with_prelabeled=True):
        """Calculate scaling factor between inches and coordinates using a reference point"""
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
            current_df = pd.read_excel(os.path.join(self.input_dir, f"LGFG-SH-01-CCB-FOA-{current_size}.dxf_combined_entities.xlsx"))
            next_df = pd.read_excel(os.path.join(self.input_dir, f"LGFG-SH-01-CCB-FOA-{next_size}.dxf_combined_entities.xlsx"))
            
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
        """Analyze MTM points across different sizes with optional verification"""
        self.load_mtm_points_from_files()
        results = {}
        sizes = sorted(self.mtm_points_by_size.keys())
        
        for i in range(len(sizes) - 1):
            current_size = sizes[i]
            next_size = sizes[i + 1]
            
            # Get grading rules for this size range
            measurements = {}
            for point in self.mtm_points_by_size[current_size]:
                current_coords = self.get_point_coordinates(point, current_size)
                if not current_coords:
                    continue
                    
                # Get grading rule movements
                rule_movements = self.get_measurements(
                    piece_name="FFS-V2-SH-01-CCB-FO",
                    points=[str(int(point))],
                    current_size=current_size,
                    next_size=next_size
                )
                
                # Calculate predicted position using grading rules
                if rule_movements and str(int(point)) in rule_movements:
                    rule = rule_movements[str(int(point))]
                    predicted_x = current_coords[0] + rule['delta_x']
                    predicted_y = current_coords[1] + rule['delta_y']
                    
                    # Verify against pre-labeled file if it exists
                    if verify_with_prelabeled:
                        actual_coords = self.get_point_coordinates(point, next_size)
                        if actual_coords:
                            print(f"\n📍 Point {point} (Size {current_size} → {next_size}):")
                            print(f"Current Position: ({current_coords[0]:.3f}, {current_coords[1]:.3f})")
                            print(f"Grading Movement: dx={rule['delta_x']:.3f}\", dy={rule['delta_y']:.3f}\"")
                            print(f"Calculated Next Position: ({predicted_x:.3f}, {predicted_y:.3f})")
                            print(f"Pre-labeled Position: ({actual_coords[0]:.3f}, {actual_coords[1]:.3f})")
                            
                            # Calculate differences
                            x_diff = abs(predicted_x - actual_coords[0])
                            y_diff = abs(predicted_y - actual_coords[1])
                            if x_diff > 0.001 or y_diff > 0.001:  # Show differences larger than 0.001 inches
                                print(f"⚠️ Differences:")
                                print(f"  X diff: {x_diff:.3f}\"")
                                print(f"  Y diff: {y_diff:.3f}\"")
                            else:
                                print("✅ Positions match!")
                            print("-" * 40)
                
                # Store results
                measurements[point] = {
                    'current_coords': current_coords,
                    'rule_movements': rule_movements.get(str(int(point))) if rule_movements else None,
                    'predicted_coords': (predicted_x, predicted_y) if 'predicted_x' in locals() else None,
                    'actual_coords': actual_coords if 'actual_coords' in locals() else None
                }
            
            if measurements:
                results[(current_size, next_size)] = measurements
        
        return results

    def generate_labeled_files(self):
        """Generate new labeled files using grading rules"""
        source_dir = "data/input/graded_mtm_combined_entities/shirt/LGFG-SH-01-CCB-FOA"
        target_dir = "data/input/graded_mtm_combined_entities_labeled/shirt/generated_with_grading_rule"
        os.makedirs(target_dir, exist_ok=True)
        
        # Load base size (39) to get all MTM points that should exist
        base_size = 39
        base_file = os.path.join(source_dir, f"LGFG-SH-01-CCB-FOA-{base_size}.dxf_combined_entities.xlsx")
        base_df = pd.read_excel(base_file)
        base_points = base_df[base_df['MTM Points'].notna()]['MTM Points'].unique()
        
        print(f"\nFound {len(base_points)} MTM points in base size {base_size}")
        sizes = [38, 39, 40]  # Process all sizes
        
        for size in sizes:
            source_file = os.path.join(source_dir, f"LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx")
            target_file = os.path.join(target_dir, f"LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx")
            
            if not os.path.exists(source_file):
                print(f"⚠️ Source file not found: {source_file}")
                continue
            
            df = pd.read_excel(source_file)
            print(f"\nProcessing size {size}:")
            
            if size == base_size:
                # For base size, use positions directly from base file
                print(f"Using base size positions")
                df = base_df.copy()
                
            else:
                # For other sizes, calculate positions using grading rules
                print(f"Calculating positions for {len(base_points)} points")
                
                for point in base_points:
                    # Get base position from size 39
                    base_coords = base_df[base_df['MTM Points'] == point][['PL_POINT_X', 'PL_POINT_Y']].iloc[0]
                    
                    # Get grading rule
                    rule_movements = self.get_measurements(
                        piece_name="FFS-V2-SH-01-CCB-FO",
                        points=[str(int(point))],
                        current_size=min(size, base_size),  # From
                        next_size=max(size, base_size)      # To
                    )
                    
                    if rule_movements and str(int(point)) in rule_movements:
                        rule = rule_movements[str(int(point))]
                        
                        # Calculate new position (reverse movement if size < base_size)
                        if size < base_size:
                            x = base_coords['PL_POINT_X'] - rule['delta_x']
                            y = base_coords['PL_POINT_Y'] - rule['delta_y']
                        else:
                            x = base_coords['PL_POINT_X'] + rule['delta_x']
                            y = base_coords['PL_POINT_Y'] + rule['delta_y']
                        
                        # Find closest point in source file
                        df['_distance'] = np.sqrt(
                            (df['PL_POINT_X'] - x)**2 + 
                            (df['PL_POINT_Y'] - y)**2
                        )
                        closest_idx = df['_distance'].idxmin()
                        
                        # Add MTM point
                        df.at[closest_idx, 'MTM Points'] = point
                        print(f"  Added point {point} at position ({x:.3f}, {y:.3f})")
                        
                # Remove temporary distance column if it exists
                if '_distance' in df.columns:
                    df = df.drop('_distance', axis=1)
            
            # Save new file
            df.to_excel(target_file, index=False)
            print(f"✅ Generated: {target_file}")

def print_point_analysis(point, current_size, next_size, current_pos, next_pos, rule_movement=None):
    """Print detailed analysis of point movement between sizes"""
    print(f"\n📍 Point {point}")
    print("-" * 30)
    
    # Current position
    print(f"Current Position:")
    print(f"  Size {current_size}: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
    
    # Calculate actual movement
    actual_dx = next_pos[0] - current_pos[0]
    actual_dy = next_pos[1] - current_pos[1]
    actual_distance = np.sqrt(actual_dx**2 + actual_dy**2)
    
    print(f"\nActual Movement:")
    print(f"  ΔX: {actual_dx:.3f}")
    print(f"  ΔY: {actual_dy:.3f}")
    print(f"  Distance: {actual_distance:.3f}")
    
    # If we have grading rules, compare them
    if rule_movement:
        print(f"\nGrading Rule Movement:")
        print(f"  ΔX: {rule_movement['delta_x']:.3f}")
        print(f"  ΔY: {rule_movement['delta_y']:.3f}")
        print(f"  Distance: {rule_movement['distance']:.3f}")
        
        # Calculate differences
        dx_diff = abs(actual_dx - rule_movement['delta_x'])
        dy_diff = abs(actual_dy - rule_movement['delta_y'])
        
        print(f"\nMovement Difference (Actual vs Rule):")
        print(f"  ΔX diff: {dx_diff:.3f}")
        print(f"  ΔY diff: {dy_diff:.3f}")
    
    # Print predicted vs actual positions
    print(f"\nNext Position:")
    print(f"  Size {next_size} (Actual): ({next_pos[0]:.3f}, {next_pos[1]:.3f})")
    if rule_movement:
        predicted_x = current_pos[0] + rule_movement['delta_x']
        predicted_y = current_pos[1] + rule_movement['delta_y']
        print(f"  Size {next_size} (Predicted): ({predicted_x:.3f}, {predicted_y:.3f})")
        
        # Check for significant differences
        if dx_diff > 0.001 or dy_diff > 0.001:
            print(f"\n⚠️ Position Error:")
            print(f"  X error: {dx_diff:.3f}")
            print(f"  Y error: {dy_diff:.3f}")
    print("-" * 30)

def find_closest_unused_point(df, target_x, target_y, used_indices):
    """Find closest point that hasn't been used yet"""
    distances = np.sqrt(
        (df['PL_POINT_X'] - target_x)**2 + 
        (df['PL_POINT_Y'] - target_y)**2
    )
    
    # Mask out already used points - convert set to list for pandas indexing
    distances[list(used_indices)] = np.inf
    return distances.idxmin()

def main():
    # Initialize and load rules
    grading = GradingRules()
    grading.load_rules()
    
    print("\n📝 Generating Labeled Files:")
    print("========================================")
    
    # Define directories
    source_dir = "data/input/graded_mtm_combined_entities/shirt/LGFG-SH-01-CCB-FOA"
    target_dir = "data/input/graded_mtm_combined_entities_labeled/shirt/generated_with_grading_rule"
    os.makedirs(target_dir, exist_ok=True)
    
    # Load base template (pre-labeled size 39)
    base_size = 39
    base_file = "data/input/graded_mtm_combined_entities/shirt/pre_labeled_graded_files/LGFG-SH-01-CCB-FOA-39.dxf_combined_entities.xlsx"
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
    
    print(f"Found {len(mtm_points)} MTM points in base template (size {base_size}):")
    for point in mtm_points:
        print(f"  Point {point['point']}: ({point['x']:.3f}, {point['y']:.3f})")
    
    # Process each size
    sizes = [38, 39, 40]
    for size in sizes:
        source_file = os.path.join(source_dir, f"LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx")
        target_file = os.path.join(target_dir, f"LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx")
        
        print(f"\nProcessing size {size}:")
        df = pd.read_excel(source_file)
        
        # Keep track of used points
        used_indices = set()
        
        # Process each point from base template
        for point_data in mtm_points:
            mtm_point = point_data['point']
            base_x = point_data['x']
            base_y = point_data['y']
            
            if size == base_size:
                # For base size, use exact positions from template
                x, y = base_x, base_y
            else:
                # Try to get grading rule
                rule_movements = grading.get_measurements(
                    piece_name="FFS-V2-SH-01-CCB-FO",
                    points=[str(int(mtm_point))],
                    current_size=min(size, base_size),
                    next_size=max(size, base_size)
                )
                
                if rule_movements and str(int(mtm_point)) in rule_movements:
                    # Calculate new position using grading rule
                    rule = rule_movements[str(int(mtm_point))]
                    if size < base_size:
                        x = base_x - rule['delta_x']
                        y = base_y - rule['delta_y']
                    else:
                        x = base_x + rule['delta_x']
                        y = base_y + rule['delta_y']
                else:
                    # No grading rule - use base position
                    x, y = base_x, base_y
            
            # Find closest unused point
            closest_idx = find_closest_unused_point(df, x, y, used_indices)
            df.at[closest_idx, 'MTM Points'] = mtm_point
            used_indices.add(closest_idx)
            
            print(f"  Added point {mtm_point} at position ({x:.3f}, {y:.3f})")
        
        # Verify all points were transferred
        transferred_points = sorted(df[df['MTM Points'].notna()]['MTM Points'].unique())
        print("\nVerification:")
        print(f"  Expected points: {[p['point'] for p in mtm_points]}")
        print(f"  Transferred points: {transferred_points}")
        if len(transferred_points) != len(mtm_points):
            print("⚠️ WARNING: Not all points were transferred!")
        
        # Save file
        df.to_excel(target_file, index=False)
        print(f"✅ Generated: {target_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
