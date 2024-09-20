from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import os
import yaml
from utils.aws_utils import AwsUtils
from app.main import Main
from ui.dxf_loader import DXFLoader
from ui.mtm_processor import MTMProcessor
import tempfile
import shutil

# Run from project root (no relative path needed)
config_filepath = "tailoring_api_config.yml"
#test_profile = "183295423477_PowerUserAccess"

app = Flask(__name__, template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 MB as per file upload limit

dxf_loader = DXFLoader()

# Load config file
with open(config_filepath) as f:
    try:
        yaml_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

ALLOWED_EXTENSIONS = yaml_config['ALLOWED_EXTENSIONS']
ALLOWED_MIME_TYPES = yaml_config['ALLOWED_MIME_TYPES']
AWS_S3_BUCKET_NAME = yaml_config['AWS_S3_BUCKET_NAME']
AWS_DXF_DIR_PATH = yaml_config['AWS_DXF_DIR_PATH']
AWS_MTM_DIR_PATH = yaml_config['AWS_MTM_DIR_PATH']
AWS_MTM_DIR_PATH_LABELED = yaml_config['AWS_MTM_DIR_PATH_LABELED']
AWS_OUTPUT_DIR_PATH = yaml_config['AWS_OUTPUT_DIR_PATH']
AWS_S3_SIGNATURE_VERSION = yaml_config['AWS_S3_SIGNATURE_VERSION']
AWS_PLOT_DIR_BASE = yaml_config['AWS_PLOT_DIR_BASE']
AWS_PROFILE= yaml_config['AWS_PROFILE']

aws_utils = AwsUtils(ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, AWS_S3_BUCKET_NAME, AWS_S3_SIGNATURE_VERSION,AWS_PROFILE)

# Function to clear static/plots folder
def clear_static_plots_folder(folder_path="ui/static/plots/"):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory and all its contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@app.route("/home")
def home():
    # return "Hello World!"
    contents_output = aws_utils.list_all_s3_files(AWS_OUTPUT_DIR_PATH)
    contents_mtm = aws_utils.list_all_s3_files(AWS_MTM_DIR_PATH)
    return render_template("Index.html", contents_output=contents_output, contents_mtm=contents_mtm)

@app.route("/upload_file", methods=["POST", "GET"])
def upload_file():
    if "file-to-s3" not in request.files:
        return "No files key in request.files"

    file = request.files["file-to-s3"]
    if not aws_utils.allowed_file(file.filename) or aws_utils.allowed_mime(file):
        return "FILE FORMAT NOT ALLOWED"
    if file.filename == "":
        return "Please select a file"
    if file:
        typeform = request.form['file_choice']
        print("typeform" + typeform)

        if typeform.lower() == 'dxf_file':
            print(f"Processing DXF file: {file.filename}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as temp_file:
                file.save(temp_file.name)  # Save the uploaded file to the temp file

                # Load DXF using the file path
                dxf_loader.load_dxf(temp_file.name)

                # Convert DXF entities to a Pandas DataFrame
                df = dxf_loader.entities_to_dataframe()

                sorted_df = df.sort_values(by=['Filename', 'Type', 'Layer'])
                sorted_df['MTM Points'] = ''

                base_filename = os.path.splitext(os.path.basename(file.filename))[0]
                aws_mtm_dir_path = os.path.join(AWS_MTM_DIR_PATH, f"{base_filename}_combined_entities.csv")

            # Delete the temp file after processing (optional)
            os.remove(temp_file.name)

            # This needs to happen after temp file is closed
            aws_utils.upload_file_to_s3(file, AWS_DXF_DIR_PATH)
            #aws_utils.upload_dataframe_to_s3(sorted_df, aws_mtm_dir_path, file_format="csv")
            aws_utils.upload_dataframe_to_s3(sorted_df, aws_mtm_dir_path, file_format="excel")
        
            # Generate a presigned URL for the CSV file
            presigned_url = aws_utils.generate_presigned_url(aws_mtm_dir_path)

            return render_template("Download.html", download_url=presigned_url)

        if typeform.lower() == 'mtm_points_file':
            print(f"Processing MTM points file: {file.filename}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as temp_file:
                file.save(temp_file.name)
                mtm_processor = MTMProcessor()
                mtm_processor.load_coordinates_tables(temp_file.name)
                mtm_processor.remove_nan_mtm_points()
                #mtm_processor.display_filtered_coordinates_tables()

                # Remove the extension
                table_name = os.path.splitext(file.filename)[0]
                piece_name = table_name.split('_')[0]
                print(f"Piece Name: {piece_name}")

                plot_filename, plot_file_path = mtm_processor.plot_points(rename=piece_name)
                
                aws_mtm_plot_dir_path = os.path.join(AWS_PLOT_DIR_BASE, f"{plot_filename}_base.png")
                aws_utils.upload_file_by_path_to_s3(plot_file_path, aws_mtm_plot_dir_path)
                print(f"Plot Filename {plot_filename}")

            # Delete the temp file after processing (optional)
            os.remove(temp_file.name)     

            aws_utils.upload_file_to_s3(file, AWS_MTM_DIR_PATH_LABELED)

            # Generate a presigned URL for the plot file in S3
            plot_presigned_url = aws_utils.generate_presigned_url(aws_mtm_plot_dir_path)

            # Clear the static/plots folder after everything is processed
            clear_static_plots_folder()

            # Render the plot on the page by passing the presigned URL to the template
            return render_template("display_plot.html", plot_url=plot_presigned_url)   
                 
        return redirect("/home")
    else:
        return "File not uploaded successfully"


@app.route("/download_file_to_dir/<path:filename>", methods=['GET'])
def download_files(s3_filepath, local_filepath):
    if request.method == 'GET':
        output = aws_utils.download_file_from_s3(s3_filepath, local_filepath)
        return output


@app.route("/download_file_as_attachment/<path:s3_filename>", methods=['GET'])
def download_files_as_attachment(s3_filename):
    if request.method == 'GET':
        presigned_url = aws_utils.download_file_as_attachment(s3_filename)
        return redirect(presigned_url)


@app.route("/upload_dxf")
def ui_upload_dxf():
    return "Placeholder"


@app.route("/process", methods=['POST'])
def run_main_process():
    # Get input data from the request
    alteration_filepath = request.json.get('alteration_filepath')
    combined_entities_folder = request.json.get('combined_entities_folder')
    preprocessed_table_path = request.json.get('preprocessed_table_path')
    input_vertices_path = request.json.get('input_vertices_path')
    processed_alterations_path = request.json.get('processed_alterations_path')
    processed_vertices_path = request.json.get('processed_vertices_path')

    # Validate the provided paths
    if not os.path.exists(alteration_filepath):
        return jsonify({"error": "Alteration file path does not exist."}), 400
    if not os.path.isdir(combined_entities_folder):
        return jsonify({"error": "Combined entities folder does not exist."}), 400
    if not os.path.exists(preprocessed_table_path):
        return jsonify({"error": "Preprocessed table path does not exist."}), 400
    if not os.path.exists(input_vertices_path):
        return jsonify({"error": "Input vertices path does not exist."}), 400
    if not os.path.exists(processed_alterations_path):
        return jsonify({"error": "Processed alterations path does not exist."}), 400
    if not os.path.exists(processed_vertices_path):
        return jsonify({"error": "Processed vertices path does not exist."}), 400

    try:
        # Initialize the Main object and run the process
        main_process = Main(alteration_filepath, combined_entities_folder,
                            preprocessed_table_path, input_vertices_path,
                            processed_alterations_path, processed_vertices_path)
        main_process.run()

        return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
