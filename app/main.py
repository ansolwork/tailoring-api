from apply_alteration import ApplyAlteration
import pandas as pd

class Main:
    def __init__(self):
        pass

    # Figure out how to add a custom alteration
    def create_alteration_table(self, rule_name="", movement_x=0., movement_y=0.):

        # Defining the data as a dictionary
        data = {
            "Rule Name": ["rule_name"] * 10,  # Repeats "4-WAIST" for all rows
            "Alt Type": ["CW Ext", "CCW Ext", "X Y MOVE", "X Y MOVE", "CW Ext", "CCW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE"],
            "First PT": [8015, 8017, 8033, 0, 8108, 8110, 8116, 0, 8014, 8107],
            "Second PT": [8016, 8016, 8033, 0, 8109, 8109, 8116, 0, 8014, 8107],
            "Movement X": ["0.000%"] * 10,  # Repeats "0.000%" for all rows
            "Movement Y": ["12.500%", "12.500%", "25.000%", "0.000%", "12.500%", "12.500%", "25.000%", "0.000%", "-5.000%", "-5.000%"]
        }

        # Creating a pandas DataFrame from the dictionary
        df = pd.DataFrame(data)

        # Displaying the DataFrame
        print(df)

        return ""


if __name__ == "__main__":
    alteration_rules_file = "../data/output_tables/alteration_rules.xlsx"

    # Change this to something else later
    alt_type = "4-WAIST"

    # Get more mtm points for combined entities
    coordinates_table_path = "../data/mtm-combined-entities"
    apply_alteration = ApplyAlteration(alteration_rules_file)
    apply_alteration.load_coordinates_tables(coordinates_table_path)
    apply_alteration.remove_nan_mtm_points()
    apply_alteration.merge_with_rule_subset(alt_type)


    # Save the merged DataFrame with rule subset to an Excel file
    output_merged_with_rule_subset_file = "../data/output_tables/merged_with_rule_subset.xlsx"
    apply_alteration.export_merged_with_rule_subset_to_excel(output_merged_with_rule_subset_file)
