import pandas as pd

class DXFComparer:
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame, pattern1: str, pattern2: str):
        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.df1['Source'] = pattern1
        self.df2['Source'] = pattern2

    def merge_dataframes(self):
        # Merge the dataframes on common columns, adding a suffix to distinguish between them
        common_columns = [col for col in self.df1.columns if col in self.df2.columns]
        merged_df = pd.merge(self.df1, self.df2, on=common_columns, how='outer', suffixes=('_df1', '_df2'))

        return merged_df
    
    def calculate_alteration_diff(self, merged_df):

        for index, row in merged_df.iterrows():
            if row['Type'] == 'POLYLINE':
                print(row['Vertices'])
            break

        return merged_df


# Example usage:
if __name__ == "__main__":
    pattern_1 = "basic_pattern"
    pattern_2 = "minor_alterations"

    table_1 = "../data/output_tables/" + pattern_1 + "_combined_entities.csv" 
    table_2 = "../data/output_tables/" + pattern_2 + "_combined_entities.csv" 

    # Load dataframes from CSV files
    df1 = pd.read_csv(table_1)
    df2 = pd.read_csv(table_2)

    comparer = DXFComparer(df1, df2, pattern_1, pattern_2)
    merged_df = comparer.merge_dataframes()
    alteration_df = comparer.calculate_alteration_diff(merged_df)

    # Save merged results to a CSV file
    merged_df.to_csv("merged_results.csv", index=False)
    merged_df.to_excel('merged_results.xlsx', index=False)

    print(merged_df.head())
