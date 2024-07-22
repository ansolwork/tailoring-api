import pandas as pd

# Existing data
data_waist = {
    "Rule Name": ["4-WAIST"] * 10,
    "Alt Type": ["CW Ext", "CCW Ext", "X Y MOVE", "X Y MOVE", "CW Ext", "CCW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE"],
    "First PT": [8015, 8017, 8033, 0, 8108, 8110, 8116, 0, 8014, 8107],
    "Second PT": [8016, 8016, 8033, 0, 8109, 8109, 8116, 0, 8014, 8107],
    "Movement X": ["0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%"],
    "Movement Y": ["12.500%", "12.500%", "25.000%", "0.000%", "12.500%", "12.500%", "25.000%", "0.000%", "-5.000%", "-5.000%"]
}

data_hip = {
    "Rule Name": ["4-HIP"] * 10,
    "Alt Type": ["CW Ext", "X Y MOVE", "X Y MOVE", "CW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE"],
    "First PT": [8016, 8035, 0, 8109, 8118, 0, 0, 0, 0, 0],
    "Second PT": [8017, 8035, 0, 8110, 8118, 0, 0, 0, 0, 0],
    "Movement X": ["0.000%"] * 10,
    "Movement Y": ["25.000%", "25.000%", "0.000%", "25.000%", "25.000%", "0.000%", "0.000%", "0.000%", "0.000%", "0.000%"]
}

data_1lth_full = {
    "Rule Name": ["1LTH-FULL"] * 22,
    "Alt Type": ["X Y MOVE"] * 22,
    "First PT": [8037, 8017, 8021, 8035, 8028, 8025, 8023, 8016, 8033, 8041, 0, 8122, 8110, 8118, 8116, 8109, 0, 8160, 0, 8060, 0, 0],
    "Second PT": [8039, 8020, 8022, 8036, 8028, 8025, 8023, 8016, 8034, 8041, 0, 8124, 8111, 8119, 8117, 8109, 0, 8164, 0, 8064, 0, 0],
    "Movement X": ["100.000%"] * 7 + ["33.000%"] * 2 + ["100.000%"] * 7 + ["33.000%"] * 2 + ["0.000%"] * 4,
    "Movement Y": ["0.000%"] * 22
}

data_chest = {
    "Rule Name": ["4-CHEST"] * 23,
    "Alt Type": ["CW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "CW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "CCW Ext", "X Y MOVE", "CW Ext", "X Y MOVE", "CCW Ext", "CW Ext", "X Y MOVE", "X Y MOVE", "CCW Ext", "X Y MOVE"],
    "First PT": [8011, 8015, 8011, 8031, 8029, 0, 8102, 8107, 8108, 8102, 8114, 8112, 0, 8308, 8306, 8315, 0, 8328, 8331, 8326, 8327, 8305, 0],
    "Second PT": [8014, 8015, 8011, 8031, 8029, 0, 8106, 8107, 8108, 8102, 8114, 8112, 0, 8307, 8306, 8300, 0, 8307, 8300, 8326, 8327, 8301, 0],
    "Movement X": ["0.000%"] * 23,
    "Movement Y": ["28.000%", "25.000%", "5.000%", "25.000%", "5.000%", "0.000%", "28.000%", "28.000%", "25.000%", "5.000%", "25.000%", "5.000%", "0.000%", "12.500%", "7.800%", "-12.500%", "0.000%", "12.500%", "-12.500%", "11.500%", "-11.500%", "-7.500%", "0.000%"]
}

# New alterations
data_2sl_hround = {
    "Rule Name": ["2SL-HROUND"] * 12,
    "Alt Type": ["X Y MOVE"] * 12,
    "First PT": [8318, 8328, 8332, 8334, 0, 8330, 8333, 8335, 8319, 0, 0, 0],
    "Second PT": [8318, 8329, 8332, 8334, 0, 8331, 8333, 8335, 8319, 0, 0, 0],
    "Movement X": ["0.60%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.40%", "0.00%", "0.00%", "0.00%"],
    "Movement Y": ["13.50%", "50.00%", "50.00%", "50.00%", "0.00%", "-50.00%", "-50.00%", "-50.00%", "-14.20%", "0.00%", "0.00%", "0.00%"]
}

data_chest_acrs = {
    "Rule Name": ["4CHESTACRS"] * 7,
    "Alt Type": ["X Y MOVE"] * 7,
    "First PT": [8011, 8012, 8013, 8014, 8029, 8031, 0],
    "Second PT": [8011, 8012, 8013, 8015, 8029, 8031, 0],
    "Movement X": ["0.00%"] * 7,
    "Movement Y": ["50.00%", "60.00%", "40.00%", "25.00%", "50.00%", "25.00%", "0.00%"]
}

data_2sl_bicep = {
    "Rule Name": ["2SL-BICEP"] * 22,
    "Alt Type": ["X Y MOVE"] * 22,
    "First PT": [8300, 8301, 8302, 8303, 8304, 8350, 8305, 8351, 8306, 8307, 8318, 8326, 8308, 8320, 8336, 8337, 8319, 8327, 8328, 0, 8399, 8398],
    "Second PT": [8300, 8301, 8302, 8303, 8304, 8350, 8305, 8351, 8306, 8307, 8318, 8326, 8315, 8321, 8336, 8337, 8319, 8327, 8331, 0, 8399, 8398],
    "Movement X": ["14.17%", "32.28%", "43.60%", "47.48%", "72.00%", "95.00%", "100.00%", "95.00%", "57.00%", "13.94%", "13.94%", "0.00%", "100.00%", "100.00%", "100.00%", "100.00%", "14.17%", "0.00%", "100.00%", "0.00%", "53.40%", "53.40%"],
    "Movement Y": ["-46.11%", "-18.51%", "-4.51%", "1.37%", "18.86%", "8.00%", "2.80%", "2.00%", "1.71%", "46.17%", "50.00%", "50.00%", "0.00%", "0.00%", "0.00%", "0.00%", "-50.00%", "-50.00%", "0.00%", "0.00%", "24.00%", "-24.00%"]
}

data_2armholein = {
    "Rule Name": ["2ARMHOLEIN"] * 22,
    "Alt Type": ["X Y MOVE"] * 22,
    "First PT": [8011, 8012, 8031, 0, 8102, 8103, 8114, 0, 8319, 8301, 8302, 8303, 8304, 8306, 8307, 8326, 8327, 0, 8399, 8398, 0, 0],
    "Second PT": [8011, 8015, 8032, 0, 8102, 8102, 8115, 0, 8300, 8301, 8302, 8303, 8304, 8306, 8318, 8326, 8327, 0, 8399, 8398, 0, 0],
    "Movement X": ["30.50%", "50.00%", "50.00%", "0.00%", "37.70%", "50.00%", "50.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "1.50%", "1.50%", "0.00%", "0.00%"],
    "Movement Y": ["0.00%"] * 22
}

data_2armholedn = {
    "Rule Name": ["2ARMHOLEDN"] * 29,
    "Alt Type": ["X Y MOVE"] * 29,
    "First PT": [8011, 8012, 8031, 8016, 8033, 0, 8102, 8103, 8114, 8109, 8116, 0, 8319, 8301, 8302, 8303, 8304, 8306, 8307, 0, 8305, 8328, 8308, 0, 8029, 8112, 0, 8399, 8398],
    "Second PT": [8011, 8015, 8032, 8016, 8034, 0, 8102, 8108, 8115, 8109, 8117, 0, 8300, 8301, 8302, 8303, 8304, 8306, 8318, 0, 8305, 8331, 8315, 0, 8030, 8113, 0, 8399, 8398],
    "Movement X": ["61.00%", "100.00%", "100.00%", "50.00%", "50.00%", "0.00%", "75.40%", "100.00%", "100.00%", "50.00%", "50.00%", "0.00%", "90.00%", "85.00%", "82.60%", "79.20%", "47.70%", "80.00%", "90.00%", "0.00%", "25.00%", "25.00%", "25.00%", "0.00%", "61.00%", "75.40%", "0.00%", "61.30%", "61.30%"],
    "Movement Y": ["0.00%"] * 29
}

data_3_backacrs = {
    "Rule Name": ["3-BACKACRS"] * 5,
    "Alt Type": ["CCW No Ext", "CCW No Ext", "CCW Ext", "X Y MOVE", "X Y MOVE"],
    "First PT": [8107, 8109, 8107, 8112, 8114],
    "Second PT": [8102, 8107, 8103, 8112, 8114],
    "Movement X": ["0.00%"] * 5,
    "Movement Y": ["50.00%", "25.00%", "12.50%", "41.00%", "21.00%"]
}

data_1lth_back = {
    "Rule Name": ["1LTH-BACK"] * 15,
    "Alt Type": ["X Y MOVE"] * 15,
    "First PT": [8199, 8110, 8118, 8122, 8109, 8116, 8160, 0, 8017, 8018, 8035, 8037, 8016, 8033, 8060],
    "Second PT": [8111, 8110, 8119, 8124, 8109, 8117, 8164, 0, 8017, 8018, 8036, 8039, 8016, 8034, 8064],
    "Movement X": ["100.00%", "50.00%", "50.00%", "50.00%", "16.50%", "16.50%", "16.50%", "0.00%", "50.00%", "25.00%", "50.00%", "50.00%", "16.50%", "16.50%", "16.50%"],
    "Movement Y": ["0.00%"] * 15
}

data_1lth_front = {
    "Rule Name": ["1LTH-FRONT"] * 18,
    "Alt Type": ["X Y MOVE"] * 18,
    "First PT": [8019, 8025, 8023, 8017, 8037, 8018, 8035, 8016, 8033, 8060, 0, 8110, 8118, 8122, 8109, 8116, 8160, 0],
    "Second PT": [8022, 8025, 8023, 8017, 8039, 8018, 8036, 8016, 8034, 8064, 0, 8110, 8119, 8124, 8109, 8117, 8164, 0],
    "Movement X": ["100.00%", "100.00%", "100.00%", "50.00%", "50.00%", "75.00%", "50.00%", "16.50%", "16.50%", "16.50%", "0.00%", "50.00%", "50.00%", "50.00%", "16.50%", "16.50%", "16.50%", "0.00%"],
    "Movement Y": ["0.00%"] * 18
}

data_1lth_fslv = {
    "Rule Name": ["1LTH-FSLV"] * 17,
    "Alt Type": ["X Y MOVE"] * 17,
    "First PT": [8308, 8323, 8322, 8324, 8325, 0, 8336, 8328, 8332, 8334, 0, 0, 0, 0, 0, 0, 0],
    "Second PT": [8315, 8323, 8322, 8324, 8325, 0, 8336, 8331, 8333, 8335, 0, 0, 0, 0, 0, 0, 0],
    "Movement X": ["100.00%", "100.00%", "100.00%", "100.00%", "100.00%", "0.00%", "100.00%", "100.00%", "100.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%"],
    "Movement Y": ["0.00%"] * 17
}

data_1lth_hslv = {
    "Rule Name": ["1LTH-HSLV"] * 19,
    "Alt Type": ["X Y MOVE"] * 19,
    "First PT": [8308, 8320, 8323, 8322, 8324, 8325, 0, 8336, 8328, 8332, 8335, 0, 0, 0, 0, 0, 0, 0, 0],
    "Second PT": [8315, 8321, 8323, 8322, 8324, 8325, 0, 8336, 8331, 8333, 8335, 0, 0, 0, 0, 0, 0, 0, 0],
    "Movement X": ["100.00%"] * 19,
    "Movement Y": ["0.00%"] * 19
}

data_2sl_fcuff = {
    "Rule Name": ["2SL-FCUFF"] * 16,
    "Alt Type": ["CW Ext", "CCW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE"],
    "First PT": [8318, 8319, 8313, 8320, 8323, 8325, 0, 8600, 8603, 0, 0, 0, 0, 0, 0, 0],
    "Second PT": [8308, 8315, 8314, 8321, 8323, 8325, 0, 8602, 8605, 0, 0, 0, 0, 0, 0, 0],
    "Movement X": ["0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "-50.00%", "50.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%"],
    "Movement Y": ["50.00%", "-50.00%", "-25.00%", "-25.00%", "-25.00%", "-25.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%"]
}

data_3_collar = {
    "Rule Name": ["3-COLLAR"] * 16,
    "Alt Type": ["X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "CW Ext", "CW Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE", "CW Ext", "X Y MOVE"],
    "First PT": [8500, 8502, 8507, 8509, 0, 8400, 8402, 8406, 0, 8202, 8002, 8205, 0, 8203, 8172, 8175],
    "Second PT": [8501, 8502, 8507, 8509, 0, 8401, 8402, 8406, 0, 8205, 8009, 8205, 0, 8203, 8175, 8175],
    "Movement X": ["-50.00%", "-25.00%", "-25.00%", "-50.00%", "0.00%", "-50.00%", "-25.00%", "-25.00%", "0.00%", "21.00%", "0.00%", "0.00%", "0.00%", "5.00%", "0.00%", "-12.50%"],
    "Movement Y": ["0.00%"] * 16
}

data_3_shoulder = {
    "Rule Name": ["3-SHOULDER"] * 21,
    "Alt Type": ["X Y MOVE", "X Y MOVE", "X Y MOVE", "X Y MOVE", "CCW Ext", "X Y MOVE", "X Y MOVE", "CCW Ext", "X Y MOVE", "X Y MOVE", "CCW Ext", "CW Ext", "X Y MOVE", "CCW Ext", "CW Ext", "X Y MOVE", "X Y MOVE", "CW No Ext", "X Y MOVE", "X Y MOVE", "X Y MOVE"],
    "First PT": [8206, 8207, 8210, 0, 8013, 8029, 0, 8107, 8112, 0, 8308, 8315, 0, 8328, 8331, 0, 8176, 0, 8176, 8176, 8175],
    "Second PT": [8206, 8208, 8210, 0, 8010, 8029, 0, 8101, 8112, 0, 8307, 8300, 0, 8307, 8300, 0, 8177, 0, 8101, 8176, 8175],
    "Movement X": ["50.00%", "50.00%", "50.00%", "0.00%", "0.00%", "4.30%", "0.00%", "0.00%", "-2.20%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%", "0.00%"],
    "Movement Y": ["0.00%"] * 21
}

data_5_pltbkcen = {
    "Rule Name": ["5-PLTBKCEN"] * 2,
    "Alt Type": ["CCW Ext", "X Y MOVE"],
    "First PT": [8107, 8125],
    "Second PT": [8101, 8126],
    "Movement X": ["0.00%", "0.00%"],
    "Movement Y": ["100.00%", "100.00%"]
}

data_5_pltbksde = {
    "Rule Name": ["5-PLTBKSDE"] * 2,
    "Alt Type": ["CCW Ext", "X Y MOVE"],
    "First PT": [8107, 8125],
    "Second PT": [8101, 8126],
    "Movement X": ["0.00%", "0.00%"],
    "Movement Y": ["100.00%", "520.00%"]
}

data_5_dartback = {
    "Rule Name": ["5-DARTBACK"] * 6,
    "Alt Type": ["X Y MOVE"] * 6,
    "First PT": [8161, 8163, 0, 8109, 0, 8016],
    "Second PT": [8161, 8163, 0, 8109, 0, 8016],
    "Movement X": ["0.00%"] * 6,
    "Movement Y": ["25.00%", "-25.00%", "0.00%", "25.00%", "0.00%", "25.00%"]
}

data_5_dartfrt = {
    "Rule Name": ["5-DARTFRT"] * 4,
    "Alt Type": ["X Y MOVE"] * 4,
    "First PT": [8061, 8063, 0, 8016],
    "Second PT": [8061, 8063, 0, 8016],
    "Movement X": ["0.00%"] * 4,
    "Movement Y": ["25.00%", "-25.00%", "0.00%", "50.00%"]
}

# Convert the dictionaries to DataFrames
df_waist = pd.DataFrame(data_waist)
df_hip = pd.DataFrame(data_hip)
df_1lth_full = pd.DataFrame(data_1lth_full)
df_chest = pd.DataFrame(data_chest)
df_2sl_hround = pd.DataFrame(data_2sl_hround)
df_chest_acrs = pd.DataFrame(data_chest_acrs)
df_2sl_bicep = pd.DataFrame(data_2sl_bicep)
df_2armholein = pd.DataFrame(data_2armholein)
df_2armholedn = pd.DataFrame(data_2armholedn)
df_3_backacrs = pd.DataFrame(data_3_backacrs)
df_1lth_back = pd.DataFrame(data_1lth_back)
df_1lth_front = pd.DataFrame(data_1lth_front)
df_1lth_fslv = pd.DataFrame(data_1lth_fslv)
df_1lth_hslv = pd.DataFrame(data_1lth_hslv)
df_2sl_fcuff = pd.DataFrame(data_2sl_fcuff)
df_3_collar = pd.DataFrame(data_3_collar)
df_3_shoulder = pd.DataFrame(data_3_shoulder)
df_5_pltbkcen = pd.DataFrame(data_5_pltbkcen)
df_5_pltbksde = pd.DataFrame(data_5_pltbksde)
df_5_dartback = pd.DataFrame(data_5_dartback)
df_5_dartfrt = pd.DataFrame(data_5_dartfrt)

# Combine the DataFrames
df_combined = pd.concat([df_waist, df_hip, df_1lth_full, df_chest, df_2sl_hround, df_chest_acrs, df_2sl_bicep, df_2armholein, df_2armholedn, df_3_backacrs, df_1lth_back, df_1lth_front, df_1lth_fslv, df_1lth_hslv, df_2sl_fcuff, df_3_collar, df_3_shoulder, df_5_pltbkcen, df_5_pltbksde, df_5_dartback, df_5_dartfrt], ignore_index=True)

# Save the DataFrame to an Excel file
df_combined.to_excel("../data/output_tables/alteration_rules.xlsx", index=False, engine='openpyxl')

# Display the DataFrame
print(df_combined)
