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
        """Load MTM points from all files in input directory"""
        try:
            logging.info(f"Looking for files in: {self.input_dir}")
            files = [f for f in os.listdir(self.input_dir) if f.endswith('_combined_entities.xlsx')]
            
            for filename in files:
                file_path = os.path.join(self.input_dir, filename)
                logging.info(f"\nProcessing file: {file_path}")
                
                # Read the Excel file
                df = pd.read_excel(file_path)
                
                # Get size from filename
                size = self.extract_size_from_filename(filename)
                if not size:
                    size = self.extract_size_from_text(df)
                    if not size:
                        logging.warning(f"Could not determine size for file: {filename}")
                        continue
                
                # Extract MTM points where they exist (non-NaN values)
                mtm_points = []
                mtm_mask = df['MTM Points'].notna()
                
                if mtm_mask.any():
                    # Get all non-NaN MTM points
                    mtm_rows = df[mtm_mask]
                    mtm_points = mtm_rows['MTM Points'].tolist()
                    
                    if mtm_points:
                        self.mtm_points_by_size[size] = mtm_points
                        logging.info(f"Found {len(mtm_points)} MTM points for size {size}")
                        continue
                
                logging.warning(f"No valid MTM points found in file: {filename}")
                    
        except Exception as e:
            logging.error(f"Error loading MTM points: {str(e)}")
            logging.error(traceback.format_exc())
            raise

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

    def get_point_coordinates(self, point_id, size):
        """Get X,Y coordinates for a specific point in a specific size"""
        try:
            # Get the dataframe for this size
            file_path = os.path.join(
                self.input_dir, 
                f"LGFG-SH-01-CCB-FOA-{size}.dxf_combined_entities.xlsx"
            )
            df = pd.read_excel(file_path)
            
            # Find the row where MTM Points equals our point_id
            point_row = df[df['MTM Points'] == point_id]
            
            if not point_row.empty:
                # Try to get coordinates from PL_POINT_X and PL_POINT_Y first
                x = point_row['PL_POINT_X'].iloc[0] if 'PL_POINT_X' in point_row else None
                y = point_row['PL_POINT_Y'].iloc[0] if 'PL_POINT_Y' in point_row else None
                
                # If those are NaN, try Point_X and Point_Y
                if pd.isna(x) or pd.isna(y):
                    x = point_row['Point_X'].iloc[0] if 'Point_X' in point_row else None
                    y = point_row['Point_Y'].iloc[0] if 'Point_Y' in point_row else None
                
                # If we have valid coordinates, return them
                if not (pd.isna(x) or pd.isna(y)):
                    return (float(x), float(y))
                
                # Debug info about what we found
                logging.debug(f"Point {point_id} in size {size}:")
                logging.debug(f"PL_POINT_X/Y: {point_row['PL_POINT_X'].iloc[0]}, {point_row['PL_POINT_Y'].iloc[0]}")
                logging.debug(f"Point_X/Y: {point_row['Point_X'].iloc[0]}, {point_row['Point_Y'].iloc[0]}")
                
                logging.warning(f"Found NaN coordinates for point {point_id} in size {size}")
                return None
                
            logging.warning(f"No row found for point {point_id} in size {size}")
            return None
            
        except Exception as e:
            logging.error(f"Error getting coordinates for point {point_id} in size {size}: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def analyze_files(self):
        """Analyze MTM points across different sizes"""
        self.load_mtm_points_from_files()
        
        results = {}
        sizes = sorted(self.mtm_points_by_size.keys())
        
        # Print what we found first
        for size in sizes:
            points = self.mtm_points_by_size[size]
            logging.info(f"\nSize {size} contains points: {points}")
        
        # Analyze consecutive sizes
        for i in range(len(sizes) - 1):
            current_size = sizes[i]
            next_size = sizes[i + 1]
            
            current_points = self.mtm_points_by_size[current_size]
            next_points = self.mtm_points_by_size[next_size]
            
            # Compare points between sizes
            measurements = {}
            for point in current_points:
                if point in next_points:
                    current_coords = self.get_point_coordinates(point, current_size)
                    next_coords = self.get_point_coordinates(point, next_size)
                    
                    if current_coords and next_coords:
                        delta_x = next_coords[0] - current_coords[0]
                        delta_y = next_coords[1] - current_coords[1]
                        distance = ((delta_x ** 2) + (delta_y ** 2)) ** 0.5
                        
                        measurements[point] = {
                            'delta_x': delta_x,
                            'delta_y': delta_y,
                            'distance': distance
                        }
                        
                        logging.info(f"\nPoint {point} movement from size {current_size} to {next_size}:")
                        logging.info(f"  X movement: {delta_x:.3f}")
                        logging.info(f"  Y movement: {delta_y:.3f}")
                        logging.info(f"  Distance:   {distance:.3f}")
            
            if measurements:
                results[(current_size, next_size)] = measurements
        
        return results

def main():
    # Initialize and load rules
    grading = GradingRules()
    grading.load_rules()
    
    # Print manual analysis
    print("\n📏 Manual Point Analysis:")
    print("========================================")
    measurements = grading.get_measurements(
        piece_name="FFS-V2-SH-01-CCB-FO",
        points=["8002", "8003", "8004", "8005"],
        current_size=30,
        next_size=31
    )
    
    if measurements:
        for point, values in sorted(measurements.items()):
            print(f"\n📍 Point {point}")
            print("--------------------")
            print(f"  X movement: {values['delta_x']:.3f}")
            print(f"  Y movement: {values['delta_y']:.3f}")
            print(f"  Distance:   {values['distance']:.3f}")
    
    # Print file-based analysis
    print("\n📏 File-Based Analysis:")
    print("========================================")
    results = grading.analyze_files()
    
    if not results:
        print("No results found from file analysis")
    
    for (current_size, next_size), measurements in sorted(results.items()):
        print(f"\n📐 Size Range: {current_size} - {next_size}")
        print("-" * 20)
        
        for point, values in sorted(measurements.items()):
            print(f"\n📍 Point {point}")
            print("--------------------")
            print(f"  X movement: {values['delta_x']:.3f}")
            print(f"  Y movement: {values['delta_y']:.3f}")
            print(f"  Distance:   {values['distance']:.3f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
