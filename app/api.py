from flask import Flask, request, jsonify, render_template, send_file, redirect
import os
import yaml
from app.aws_utils import AwsUtils
from main import Main

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB as per file upload limit
# Load config file
with open("..\\tailoring_api_config.yml") as f:
    try:
        yaml_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

ALLOWED_EXTENSIONS = yaml_config['ALLOWED_EXTENSIONS']
ALLOWED_MIME_TYPES = yaml_config['ALLOWED_MIME_TYPES']
AWS_S3_BUCKET_NAME = yaml_config['AWS_S3_BUCKET_NAME']
AWS_DXF_DIR_PATH = yaml_config['AWS_DXF_DIR_PATH']
AWS_MTM_DIR_PATH = yaml_config['AWS_MTM_DIR_PATH']
AWS_OUTPUT_DIR_PATH = yaml_config['AWS_OUTPUT_DIR_PATH']

aws_utils = AwsUtils(ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, AWS_S3_BUCKET_NAME)


@app.route("/home")
def home():
    # return "Hello World!"
    contents = aws_utils.list_all_s3_files(AWS_OUTPUT_DIR_PATH)
    return render_template("Index.html", contents=contents)


@app.route("/upload_file", methods=["POST","GET"])
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
        print("typeform"+typeform)
        if typeform.lower() == 'dxf_file':
            aws_utils.upload_file_to_s3(file, AWS_DXF_DIR_PATH)
        if typeform.lower() == 'mtm_points_file':
            aws_utils.upload_file_to_s3(file, AWS_MTM_DIR_PATH)
        return redirect("/home")
    else:
        return "File not uploaded successfully"



@app.route("/download_file/<path:filename>", methods=['GET'])
def download_files(filename):
    if request.method == 'GET':
        output = aws_utils.download_file_from_s3(filename)
        return send_file(f"test_aws.xlsx", as_attachment=True)


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
    app.run(debug=True)
