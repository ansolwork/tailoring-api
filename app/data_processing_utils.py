import pandas as pd

class DataProcessingUtils:
    """
    A utility class for common data processing tasks, including CSV operations
    and list manipulations.
    """

    def load_csv(self, table_path):
        """
        Load a CSV file into a pandas DataFrame.
        
        Parameters:
        table_path (str): The file path to the CSV file.

        Returns:
        pd.DataFrame: The loaded DataFrame.
        """
        return pd.read_csv(table_path)
    
    def save_csv(self, df, save_path):
        """
        Save a pandas DataFrame to a CSV file.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to save.
        save_path (str): The file path where the CSV should be saved.
        """
        df.to_csv(save_path, index=False)
    
    def load_all_excel_sheets(self, df):
        return pd.read_excel(df, sheet_name=None)
    
    def drop_columns(self, df, columns_to_drop):
        """
        Drop specified columns from a pandas DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame from which to drop columns.
        columns_to_drop (list): A list of column names to be dropped.

        Returns:
        pd.DataFrame: The DataFrame with the specified columns dropped.
        """
        return df.drop(columns=columns_to_drop)
    
    def remove_duplicates(self, input_list):
        """
        Remove duplicates from a list of tuples or coordinates.

        Parameters:
        input_list (list): A list of tuples where each tuple represents coordinates.

        Returns:
        list: A list with duplicates removed, maintaining the original order.
        """
        seen = set()
        output_list = []
        for item in input_list:
            if item not in seen:
                seen.add(item)
                output_list.append(item)
        return output_list

    def remove_duplicates_preserve_order(self, flattened_list):
        """
        Remove duplicates from a list while preserving the order of elements.
        
        Parameters:
        flattened_list (list): The list from which to remove duplicates.

        Returns:
        list: A list with duplicates removed, preserving the original order.
        """
        unique_list = []
        seen = set()

        for item in flattened_list:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)

        return unique_list
    
    # Function to sort coordinates by x-coordinate
    def sort_by_x(self, coords):
        if isinstance(coords, list) and all(isinstance(coord, (list, tuple)) for coord in coords):
            return sorted(coords, key=lambda coord: coord[0])
        return coords  # Return as is if not a list of tuples

