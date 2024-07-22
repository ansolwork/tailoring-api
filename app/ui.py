import gradio as gr
import os
import shutil
import tempfile
import pandas as pd
from matplotlib import pyplot as plt
from dxf_loader import DXFLoader
from apply_alteration import ApplyAlteration  # Ensure this import is correct

# Define process_dxf function
def process_dxf(file):
    if file is None:
        return "No file uploaded. Please upload a DXF file.", []

    print(f"Processing DXF file: {file.name}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, os.path.basename(file.name))
        shutil.copy(file.name, temp_file_path)

        dxf_loader = DXFLoader()
        try:
            dxf_loader.load_dxf(temp_file_path)
            df = dxf_loader.entities_to_dataframe(temp_file_path)
        except RuntimeError as e:
            return str(e), []

        sorted_df = df.sort_values(by=['Filename', 'Type', 'Layer'])

        sorted_df['MTM Points'] = ''
        base_filename = os.path.splitext(os.path.basename(temp_file_path))[0]
        output_excel_path = os.path.join(temp_dir, f"{base_filename}_combined_entities.xlsx")
        sorted_df.to_excel(output_excel_path, index=False)

        print(f"Output Excel file generated: {output_excel_path}")

        persistent_output_dir = "./persistent_output"
        os.makedirs(persistent_output_dir, exist_ok=True)
        persistent_excel_path = os.path.join(persistent_output_dir, os.path.basename(output_excel_path))
        shutil.copy(output_excel_path, persistent_excel_path)

        plot_paths = generate_plots(persistent_excel_path)

        return persistent_excel_path, plot_paths

# Define generate_plots function
def generate_plots(excel_file_path):
    output_directory = "./persistent_output/plots"
    os.makedirs(output_directory, exist_ok=True)

    df = pd.read_excel(excel_file_path)

    def draw_lines_and_polylines(group, name):
        plt.figure(figsize=(16, 10))  # Increase figure size

        # Draw lines
        lines = group[group['Type'] == 'LINE']
        for _, row in lines.iterrows():
            if pd.notna(row['Line_Start_X']) and pd.notna(row['Line_End_X']) and pd.notna(row['Line_Start_Y']) and pd.notna(row['Line_End_Y']):
                plt.plot([row['Line_Start_X'], row['Line_End_X']], [row['Line_Start_Y'], row['Line_End_Y']], marker='o', linewidth=0.5, markersize=5)
                plt.text(row['Line_Start_X'], row['Line_Start_Y'], f"({row['Line_Start_X']}, {row['Line_Start_Y']})", fontsize=10, ha='right', va='bottom')
                plt.text(row['Line_End_X'], row['Line_End_Y'], f"({row['Line_End_X']}, {row['Line_End_Y']})", fontsize=10, ha='right', va='bottom')

        # Draw polylines
        polylines = group[group['Type'].isin(['POLYLINE', 'LWPOLYLINE'])]
        unique_points = polylines.drop_duplicates(subset=['PL_POINT_X', 'PL_POINT_Y'])
        for vertex_label in polylines['Vertex Label'].unique():
            vertex_group = polylines[polylines['Vertex Label'] == vertex_label]
            xs = vertex_group['PL_POINT_X'].tolist()
            ys = vertex_group['PL_POINT_Y'].tolist()
            plt.plot(xs, ys, marker='o', linewidth=0.5, markersize=5)

        # Annotate unique points
        for x, y, point_label in zip(unique_points['PL_POINT_X'], unique_points['PL_POINT_Y'], unique_points['Point Label']):
            plt.text(x, y, f'{point_label}', fontsize=10, ha='right', va='bottom')

        plt.title(f'Polyline Plot for {name}', fontsize=16)
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()  # Improve layout

        output_path = os.path.join(output_directory, f"polyline_plot_{name}.png")
        plt.savefig(output_path, dpi=300)  # Increase DPI for better quality
        plt.close()

        return output_path

    grouped = df.groupby('Filename')
    plot_paths = []
    for name, group in grouped:
        plot_path = draw_lines_and_polylines(group, name)
        plot_paths.append(plot_path)

    return plot_paths

# Define process_mtm_points function
def process_mtm_points(file):
    if file is None:
        return "No file uploaded. Please upload an Excel file or a zip file.", []

    print(f"Processing MTM points file: {file.name}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, os.path.basename(file.name))
        shutil.copy(file.name, temp_file_path)

        apply_alteration = ApplyAlteration()
        apply_alteration.load_coordinates_tables(temp_file_path)
        apply_alteration.remove_nan_mtm_points()
        apply_alteration.display_filtered_coordinates_tables()
        output_path = apply_alteration.plot_points()

        if output_path:
            persistent_output_dir = "./persistent_output"
            os.makedirs(persistent_output_dir, exist_ok=True)
            persistent_file_path = os.path.join(persistent_output_dir, os.path.basename(output_path))
            shutil.copy(output_path, persistent_file_path)
            print(f"Output file generated: {persistent_file_path}")
            return persistent_file_path, [persistent_file_path]
        else:
            print("No output file generated.")
            return "No output file generated.", []

# Define process_file function
def process_file(file, file_type):
    if file is None:
        return "No file uploaded. Please upload a file.", []

    if file_type == 'DXF':
        return process_dxf(file)
    elif file_type == 'MTM Points':
        return process_mtm_points(file)
    else:
        return "Invalid file type", []

# Define Gradio interface
iface = gr.Interface(
    fn=process_file,
    inputs=[gr.File(label="Upload File", type="filepath"), gr.Radio(["DXF", "MTM Points"], label="Select File Type")],
    outputs=[gr.File(label="Processed Excel File"), gr.Gallery(label="Generated Plots")],
    title="File Processor",
    description="Upload a DXF file and get a processed Excel file with combined entities and plots. Also, upload an Excel file or a zip file with MTM points to generate output graphs.",
    theme="default"
)

iface.launch(share=True)
