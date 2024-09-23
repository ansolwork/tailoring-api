import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import seaborn as sns
import matplotlib.patheffects as pe  # Correct import for path effects

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

    # Set up Seaborn styling
    sns.set(style="whitegrid", context="notebook")

    # Create a figure with a light background gradient
    fig, ax = plt.subplots(figsize=(50, 30))
    ax.set_facecolor('#f0f0f5')  # Light grey background

    # Plot altered points with larger size, bold lines, and enhanced visibility
    sns.scatterplot(
        x=altered_x, 
        y=altered_y, 
        color='red', 
        label='Altered Points', 
        marker='x', 
        s=300,  # Increased size
        linewidth=4,  # Bolder lines for the 'X' markers
        alpha=0.8,  # Reduced transparency
        edgecolor='black', 
        path_effects=[pe.withStroke(linewidth=5, foreground="white")]  # Shadow effect
    )

    # Plot MTM points with a larger size and transparency
    sns.scatterplot(x=original_x[mtm_mask], y=original_y[mtm_mask], color='orange', 
                    label='MTM Points', marker='D', s=250, alpha=0.7, edgecolor='black')

    # Plot original points with transparency
    sns.scatterplot(x=original_x, y=original_y, color='blue', label='Original Points', 
                    marker='o', s=150, alpha=0.5, edgecolor='black')

    # Plot notch points if available with distinct markers
    if plot_notches:
        sns.scatterplot(x=original_x[notch_mask], y=original_y[notch_mask], color='purple', 
                        label='Notch Points', marker='s', s=250, alpha=0.7, edgecolor='black')

    # Plot vertices (polygons) with gradients and transparency
    original_label_added = False
    for _, row in vertices_df.iterrows():
        vertices = row['vertices']
        # Convert the string representation of the list of vertices into an actual list
        vertices = ast.literal_eval(vertices)
        
        # Separate the x and y coordinates from the list of tuples
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]

        # Plot the polygon with a gradient-like effect and light transparency
        plt.plot(xs, ys, color='#0066ff', alpha=0.4, linewidth=2, linestyle='--')

    # Add fancy labels with shadow effect for MTM point numbers
    offset_x, offset_y = 0.15, 0.15

    for i in range(len(original_x)):
        if pd.notna(mtm_points[i]):  # Only label MTM points
            plt.text(original_x[i] + offset_x, original_y[i] + offset_y, str(int(mtm_points[i])), 
                     fontsize=14, ha='center', color='black', weight='bold', 
                     path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # Label MTM point numbers on the plot for altered points, but only if altered points are available
    for i in range(len(altered_x)):
        if pd.notna(mtm_points[i]) and pd.notna(altered_x[i]) and pd.notna(altered_y[i]):  # Ensure altered points exist
            plt.text(altered_x[i] + offset_x, altered_y[i] + offset_y, f"{int(mtm_points[i])} (A)", 
                     fontsize=14, ha='center', color='red', weight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # Add a better title with shadow and more styling
    plt.title('MTM, Altered, and Original Points with Vertices', 
              fontsize=28, weight='bold', color='#4b4b4b', 
              path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # Enhanced axis labels
    plt.xlabel('X Coordinate [in]', fontsize=20, weight='bold', color='#333333')
    plt.ylabel('Y Coordinate [in]', fontsize=20, weight='bold', color='#333333')

    # Customized legend with larger fonts and markers
    plt.legend(fontsize=22, loc='best', markerscale=2.5, shadow=True, frameon=True, fancybox=True, borderpad=1.5, labelspacing=1.2)

    # Add grid and enhance layout
    plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

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
