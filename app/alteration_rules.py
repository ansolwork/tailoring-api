import pandas as pd

# Data for Rule 4-WAIST
data_waist = {
    "Rule Name": ["4-WAIST"] * 3,
    "Piece Usage": ["Both"] * 3,
    "Alt Type": ["CW Ext", "CW Ext", "X Y MOVE"],
    "First PT": ["8015", "8016", "8033"],  # Converted to strings
    "Second PT": ["", "", "8033"],
    "Movement X": ["", "", "0.000%"],
    "Movement Y": ["", "", "25.000%"]
}

# Data for Rule 4-HIP
data_hip = {
    "Rule Name": ["4-HIP"] * 10,
    "Piece Usage": ["Both"] * 10,
    "Alt Type": ["CW Ext", "X Y MOVE", "X Y MOVE", "CW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE"],
    "First PT": ["8016", "8035", "0", "8109", "8118", "0", "0", "0", "0", "0"],  # Converted to strings
    "Second PT": ["8017", "8035", "0", "8110", "8118", "0", "0", "0", "0", "0"],
    "Movement X": ["0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%"],
    "Movement Y": ["25.000%", "25.000%", "0.000%", "25.000%", "25.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%"]
}

# Data for Rule 1LTH-FULL
data_1lth_full = {
    "Rule Name": ["1LTH-FULL"] * 22,
    "Piece Usage": ["Both"] * 22,
    "Alt Type": ["X Y MOVE"] * 22,
    "First PT": ["8037", "8017", "8021", "8035", "8028", "8025", "8023", "8016", "8033", "8041", "0", "8122", "8110", "8118", "8116", "8109", "0", "8160", "0", "8060", "0", "0"],  # Converted to strings
    "Second PT": ["8039", "8020", "8022", "8036", "8028", "8025", "8023", "8016", "8034", "8041", "0", "8124", "8111", "8119", "8117", "8109", "0", "8164", "0", "8064", "0", "0"],
    "Movement X": ["100.000%"] * 7 + ["33.000%"] * 2 + ["100.000%"] * 7 + ["33.000%"] * 2 + ["0.000%"] * 4,
    "Movement Y": ["0.000%"] * 22
}

# Data for Rule 4-CHEST
data_chest = {
    "Rule Name": ["4-CHEST"] * 23,
    "Piece Usage": ["Both"] * 23,
    "Alt Type": ["CW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "CW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "CCW Ext", "X Y MOVE", "CW Ext", "X Y MOVE", "CCW Ext", "CW Ext", "X Y MOVE", "X Y MOVE", "CCW Ext", "X Y MOVE"],
    "First PT": ["8011", "8015", "8011", "8031", "8029", "0", "8102", "8107", "8108", "8102", "8114", "8112", "0", "8308", "8306", "8315", "0", "8328", "8331", "8326", "8327", "8305", "0"],  # Converted to strings
    "Second PT": ["8014", "8015", "8011", "8031", "8029", "0", "8106", "8107", "8108", "8102", "8114", "8112", "0", "8307", "8306", "8300", "0", "8307", "8300", "8326", "8327", "8301", "0"],
    "Movement X": ["0.000%"] * 23,
    "Movement Y": ["28.000%", "25.000%", "5.000%", "25.000%", "5.000%", "0.000%", "28.000%", "28.000%", "25.000%", "5.000%", "25.000%", "5.000%", "0.000%", "12.500%", "7.800%", "-12.500%", "0.000%", "12.500%", "-12.500%", "11.500%", "-11.500%", "-7.500%", "0.000%"]
}

# Convert the dictionaries to DataFrames
df_waist = pd.DataFrame(data_waist)
df_hip = pd.DataFrame(data_hip)
df_1lth_full = pd.DataFrame(data_1lth_full)
df_chest = pd.DataFrame(data_chest)

# Combine the DataFrames
df_combined = pd.concat([df_waist, df_hip, df_1lth_full, df_chest], ignore_index=True)

# Save the DataFrame to an Excel file
df_combined.to_excel("../data/output_tables/alteration_rules.xlsx", index=False, engine='openpyxl')

# Display the DataFrame
print(df_combined)
