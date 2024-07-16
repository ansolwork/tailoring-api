import gradio as gr
from dxf_loader import DXFLoader
from apply_alteration import ApplyAlteration
import os
import shutil
import tempfile
import zipfile

def process_dxf(file):
    if file is None:
        return "No file uploaded. Please upload a DXF file."
    
    print(f"Processing DXF file: {file.name}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, os.path.basename(file.name))
        shutil.copy(file.name, temp_file_path)

        dxf_loader = DXFLoader()
        try:
            dxf_loader.load_dxf(temp_file_path)
            df = dxf_loader.entities_to_dataframe(temp_file_path)
        except RuntimeError as e:
            return str(e)
        
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

        return persistent_excel_path

def process_mtm_points(file):
    if file is None:
        return "No file uploaded. Please upload an Excel file or a zip file."
    
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
            return persistent_file_path
        else:
            print("No output file generated.")
            return "No output file generated."

def process_file(file, file_type):
    if file is None:
        return "No file uploaded. Please upload a file."
    
    if file_type == 'DXF':
        return process_dxf(file)
    elif file_type == 'MTM Points':
        return process_mtm_points(file)
    else:
        return "Invalid file type"

iface = gr.Interface(
    fn=process_file,
    inputs=[gr.File(label="Upload File", type="filepath"), gr.Radio(["DXF", "MTM Points"], label="Select File Type")],
    outputs="file",
    title="File Processor",
    description="Upload a DXF file and get a processed Excel file with combined entities. Also, upload an Excel file or a zip file with MTM points to generate output graphs.",
    theme="default"
)

iface.launch(share=True)
