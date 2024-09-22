import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def test_plot_vertices_and_altered(file_path, file_path_vertices, output_dir="data/output/"):
    """
    Plots original vertices, altered points, polygons (vertices), and labels the MTM points.
    Highlights notch points with a unique color if they are available.

    :param file_path: Path to the CSV file containing the alteration data.
    :param file_path_vertices: Path to the CSV file containing the vertices (polygons).
    :param output_dir: Directory path to save the output plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique output file name based on the input file name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.png")

    # Load the data
    df = pd.read_csv(file_path)
    vertices_df = pd.read_csv(file_path_vertices)

    # Extract original and altered points
    original_x = df['pl_point_x']
    original_y = df['pl_point_y']
    altered_x = df['pl_point_altered_x']
    altered_y = df['pl_point_altered_y']
    mtm_points = df['mtm points']  # Keep as numeric for better comparison
    
    # Separate MTM points from non-MTM points
    mtm_mask = df['mtm points'].notna()  # Identify rows where 'mtm points' is not NaN

    # Check if the 'notch_labels' column exists and contains valid string data
    if 'notch_labels' in df.columns and df['notch_labels'].dtype == object:
        # Identify notch points
        notch_mask = df['notch_labels'].str.contains('notch', na=False)
        plot_notches = notch_mask.any()  # Only plot notches if there are any
    else:
        plot_notches = False  # No valid notch data found

    # Set up the plot
    plt.figure(figsize=(25, 15))

    # Plot altered points (excluding MTM points)
    plt.scatter(altered_x, altered_y, color='red', label='Altered Points', marker='x')

    # Plot MTM points with a different color and marker
    plt.scatter(original_x[mtm_mask], original_y[mtm_mask], color='orange', label='MTM Points', marker='D')  # MTM points in orange

    plt.scatter(original_x, original_y, color='blue', label='Original Points', marker='o')

    # Plot notch points if available
    if plot_notches:
        plt.scatter(original_x[notch_mask], original_y[notch_mask], color='purple', label='Notch Points', marker='s')  # Notch points in purple

    # Plot vertices (polygons) and add "Original Points" label only once
    original_label_added = False
    for _, row in vertices_df.iterrows():
        vertices = row['vertices']
        # Convert the string representation of the list of vertices into an actual list
        vertices = ast.literal_eval(vertices)
        
        # Separate the x and y coordinates from the list of tuples
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]

        # Plot the polygon as a line
        if not original_label_added:
            plt.plot(xs, ys, color='blue', alpha=0.6)  # Add the label for the first plot
            original_label_added = True
        else:
            plt.plot(xs, ys, color='blue', alpha=0.6)  # No label for subsequent plots

    # Label MTM point numbers on the plot
    for i in range(len(original_x)):
        if pd.notna(mtm_points[i]):  # Only label MTM points
            plt.text(original_x[i], original_y[i], str(int(mtm_points[i])), fontsize=9, ha='right', color='black')

    # Add labels and title
    plt.title('Original, Altered Points, Vertices, MTM Points, and Notch Points' if plot_notches else 'Original, Altered Points, Vertices, and MTM Points')
    plt.xlabel('X Coordinate [in]')
    plt.ylabel('Y Coordinate [in]')
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # Save the plot to a file
    plt.savefig(output_file, format='png', dpi=300)
    plt.close()

    print(f"Plot saved to {output_file}")

# Example usage
#file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_1LTH-FULL.csv"
#file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_1LTH-BACK.csv"
#file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_2ARMHOLEDN.csv"
file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_4-WAIST.csv"
#file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_4-CHEST.csv"
#file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_7F-SHPOINT.csv"

#file_path = "data/staging_processed/debug/LGFG-V2-SH-01-STBS-F_7F-ERECT.csv"
#file_path = "data/staging_processed/debug/LGFG-V2-SH-01-STBS-F_1LTH-FULL.csv"
#file_path = "data/staging_processed/debug/LGFG-V2-BC1-SH-08_FRT-HEIGHT.csv"
#file_path = "data/staging_processed/debug/LGFG-SH-04FS-FOA_2SL-BICEP.csv"

file_path_vertices = "data/staging_processed/processed_vertices_by_piece/processed_vertices_LGFG-SH-01-CCB-FO.csv"
#file_path_vertices = "data/staging_processed/processed_vertices_by_piece/vertices_LGFG-V2-SH-01-STBS-F.csv"
#file_path_vertices = "data/staging_processed/processed_vertices_by_piece/processed_vertices_LGFG-V2-BC1-SH-08.csv"
#file_path_vertices = "data/staging_processed/processed_vertices_by_piece/processed_vertices_LGFG-SH-04FS-FOA.csv"
test_plot_vertices_and_altered(file_path, file_path_vertices)
