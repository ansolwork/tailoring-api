from dxf_loader import DXFLoader
from matplotlib import pyplot as plt
import pandas as pd
import os

class ApplyAlteration:
    def __init__(self, alteration_rules_file, scale_factor=5):
        self.alteration_rules_df = pd.read_excel(alteration_rules_file)
        self.scale_factor = scale_factor

    def apply_alterations(self, df):
        altered_entities = []
        alterations_log = []

        for index, row in self.alteration_rules_df.iterrows():
            rule_name = row['Rule Name']
            first_pt = str(row['First PT'])  # Ensure it's a string
            second_pt = str(row['Second PT'])  # Ensure it's a string
            movement_x = row['Movement X']
            movement_y = row['Movement Y']
            alt_type = row['Alt Type']

            try:
                movement_x = float(movement_x.replace('%', '')) / 100 if isinstance(movement_x, str) else float(movement_x)
                movement_y = float(movement_y.replace('%', '')) / 100 if isinstance(movement_y, str) else float(movement_y)
            except ValueError:
                print(f"Invalid movement values for rule {rule_name}: {movement_x}, {movement_y}")
                continue

            if alt_type == "X Y MOVE":
                mask = df['Text'].apply(lambda x: first_pt in str(x) if pd.notnull(x) else False)
                if mask.any():
                    for idx in df[mask].index:
                        original_x = df.at[idx, 'Position_X']
                        original_y = df.at[idx, 'Position_Y']
                        new_x = original_x + movement_x
                        new_y = original_y + movement_y
                        alterations_log.append({
                            'Filename': df.at[idx, 'Filename'],
                            'Text': df.at[idx, 'Text'],
                            'Original_X': original_x,
                            'Original_Y': original_y,
                            'New_X': new_x,
                            'New_Y': new_y
                        })
                        df.at[idx, 'Position_X'] = new_x
                        df.at[idx, 'Position_Y'] = new_y

                    altered_entities.extend(df[mask].index.tolist())
                    print(f"Applied {alt_type} to {mask.sum()} entities for rule {rule_name} with point {first_pt}")
                else:
                    print(f"No entities found for {alt_type} with point {first_pt} in rule {rule_name}")

        altered_df = df.loc[altered_entities]
        alterations_log_df = pd.DataFrame(alterations_log)
        return df, altered_df, alterations_log_df

    def generate_comparison_svg(self, original_df, altered_df, output_svg_path):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

        self.plot_entities(axes[0], original_df, title="Original")
        self.plot_entities(axes[1], altered_df, title="Altered")

        plt.savefig(output_svg_path)
        plt.close()

    def generate_altered_svg(self, altered_df, output_svg_path):
        fig, ax = plt.subplots(figsize=(10, 10))

        self.plot_entities(ax, altered_df, title="Altered Only")

        plt.savefig(output_svg_path)
        plt.close()

    def plot_entities(self, ax, df, title):
        ax.set_aspect('equal')
        ax.set_title(title)
        for _, row in df.iterrows():
            if row['Type'] == 'TEXT':
                ax.text(row['Position_X'] * self.scale_factor, row['Position_Y'] * self.scale_factor, row['Text'])
            elif row['Type'] == 'LINE':
                ax.plot([row['Start_X'] * self.scale_factor, row['End_X'] * self.scale_factor], 
                        [row['Start_Y'] * self.scale_factor, row['End_Y'] * self.scale_factor], 'k-')
            elif row['Type'] == 'CIRCLE':
                circle = plt.Circle((row['Center_X'] * self.scale_factor, row['Center_Y'] * self.scale_factor), 
                                    row['Radius'] * self.scale_factor, color='r', fill=False)
                ax.add_artist(circle)
            elif row['Type'] in ['POLYLINE', 'LWPOLYLINE']:
                vertices = [(x * self.scale_factor, y * self.scale_factor) for x, y in row['Vertices']]
                polyline = plt.Polygon(vertices, closed=False, fill=None, edgecolor='b')
                ax.add_patch(polyline)
            elif row['Type'] == 'POINT':
                ax.plot(row['Position_X'] * self.scale_factor, row['Position_Y'] * self.scale_factor, 'bo')  # Blue dot for points

if __name__ == "__main__":
    pattern = "basic_pattern"
    dxf_directory = "../data/ff_pattern_2/" + str(pattern) + "/"
    output_table_directory = "../data/output_tables/" + pattern + "_"
    alteration_rules_file = "../data/output_tables/alteration_rules.xlsx"
    output_svg_path = "../data/output_tables/comparison.svg"
    altered_only_svg_path = "../data/output_tables/altered_only.svg"
    alterations_log_excel_path = "../data/output_tables/alterations_log.xlsx"

    dxf_loader = DXFLoader()
    all_data = []

    for filename in os.listdir(dxf_directory):
        file_path = os.path.join(dxf_directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.dxf'):
            dxf_loader.load_dxf(file_path)
            df = dxf_loader.entities_to_dataframe(filename)
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    sorted_df = combined_df.sort_values(by=['Filename', 'Type', 'Layer'])

    sorted_df.to_csv(output_table_directory + 'combined_entities.csv', index=False)
    sorted_df.to_excel(output_table_directory + 'combined_entities.xlsx', index=False)

    print(sorted_df.head())

    # Apply alterations
    dxf_modifier = ApplyAlteration(alteration_rules_file, scale_factor=5)
    altered_df, altered_only_df, alterations_log_df = dxf_modifier.apply_alterations(sorted_df)

    # Generate comparison SVG
    dxf_modifier.generate_comparison_svg(sorted_df, altered_df, output_svg_path)
    print(f"Comparison SVG saved to {output_svg_path}")

    # Generate altered only SVG
    dxf_modifier.generate_altered_svg(altered_only_df, altered_only_svg_path)
    print(f"Altered only SVG saved to {altered_only_svg_path}")

    # Save alterations log to Excel
    alterations_log_df.to_excel(alterations_log_excel_path, index=False)
    print(f"Alterations log saved to {alterations_log_excel_path}")
