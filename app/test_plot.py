import pandas as pd
import matplotlib.pyplot as plt
import ast

def test_plot_vertices_and_altered(file_path, file_path_vertices):
    """
    Plots original vertices, altered points, polygons (vertices), and labels the mtm points.

    :param file_path: Path to the CSV file containing the alteration data.
    :param file_path_vertices: Path to the CSV file containing the vertices (polygons).
    """
    # Load the data
    df = pd.read_csv(file_path)
    vertices_df = pd.read_csv(file_path_vertices)

    # Extract original and altered points
    original_x = df['pl_point_x']
    original_y = df['pl_point_y']
    altered_x = df['pl_point_altered_x']
    altered_y = df['pl_point_altered_y']
    mtm_points = df['mtm points'].astype(str)  # Convert mtm points to strings

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot original points
    plt.scatter(original_x, original_y, color='blue', label='Original Points', marker='o')

    # Plot altered points
    plt.scatter(altered_x, altered_y, color='red', label='Altered Points', marker='x')

    # Plot vertices (polygons)
    for _, row in vertices_df.iterrows():
        vertices = row['vertices']
        # Convert the string representation of the list of vertices into an actual list
        vertices = ast.literal_eval(vertices)
        
        # Separate the x and y coordinates from the list of tuples
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]

        # Plot the polygon as a line
        plt.plot(xs, ys, color='green', alpha=0.6)

    # Plot mtm point labels at the original points' positions (skip NaN and 'nan' string values)
    for i in range(len(original_x)):
        if pd.notna(mtm_points[i]) and mtm_points[i] != 'nan':  # Skip NaN and 'nan' string
            plt.text(original_x[i], original_y[i], mtm_points[i], fontsize=9, ha='right', color='black')

    # Add labels and title
    plt.title('Original, Altered Points, Vertices, and MTM Points Labels')
    plt.xlabel('X Coordinate [in]')
    plt.ylabel('Y Coordinate [in]')
    plt.legend()

    # Show the plot
    plt.show()

# File path to the uploaded CSV
file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_1LTH-FULL.csv"
file_path_vertices = "data/staging_processed/processed_vertices_by_piece/processed_vertices_LGFG-SH-01-CCB-FO.csv"

# Call the test function to plot the data
test_plot_vertices_and_altered(file_path, file_path_vertices)
