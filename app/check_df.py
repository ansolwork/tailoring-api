import pandas as pd
import numpy as np

class CheckTable:
    def __init__(self, input_table_path):
        self.df = pd.read_excel(input_table_path)
        # Standardize column names to lowercase
        self.df.columns = [col.lower() for col in self.df.columns]

    def display_df(self, column=None):
        if column is not None:
            print(self.df[column])
        else:
            print(self.df)

    def filter_data(self, row_filter=None, column_filter=None):
        filtered_df = self.df
        
        if column_filter is not None:
            column_filter = [col.lower() for col in column_filter]
            filtered_df = filtered_df[column_filter]
        
        if row_filter is not None:
            filtered_df = filtered_df.query(row_filter.lower())
        
        return filtered_df
    
    def print_vertices(self, df):
        for idx, row in df.iterrows():
            print("")
            print(f"mtm points: {row['mtm points']}")
            print(f"(x, y): ({row['pl_point_x']}, {row['pl_point_y']})")
            print()
            
            # Calculate the distance from the current point to all other points
            self.df['distance'] = np.sqrt((self.df['pl_point_x'] - row['pl_point_x'])**2 + (self.df['pl_point_y'] - row['pl_point_y'])**2)
            
            # Sort by distance and get the top 3 closest points, excluding the point itself
            closest_points = self.df[self.df.index != idx].nsmallest(3, 'distance')
            
            print("Top 3 Closest Points In Vertex List:")
            for _, closest_row in closest_points.iterrows():
                print(f"(x, y): ({closest_row['pl_point_x']}, {closest_row['pl_point_y']}), Distance: {closest_row['distance']:.2f}")

            print()
            print("Vertices:")
            print(f"  {row['vertices']}")

if __name__ == "__main__":
    # Use the desired input table path
    #input_table_path = "../data/output_tables/merged_with_rule_subset.xlsx"
    input_table_path = "../data/mtm-test-points/LGFG-SH-01-CCB-FO 2_combined_entities.xlsx"
    make_alteration = CheckTable(input_table_path)
    
    # Display the whole dataframe
    make_alteration.display_df()

    # Example usage of filter_data to extract specific columns
    filter_mtm_point = 8014
    column_filter = ["mtm points", "pl_point_x", "pl_point_y", "vertices"]  # List of columns you want to keep
    row_filter = "`mtm points` == " + str(float(filter_mtm_point))

    filtered_df = make_alteration.filter_data(row_filter=row_filter, column_filter=column_filter)
    print(filtered_df)
    
    # Print vertices from the filtered DataFrame
    make_alteration.print_vertices(filtered_df)
