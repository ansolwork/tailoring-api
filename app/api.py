from flask import Flask, request, jsonify, render_template
import os

from app.aws_utils import AwsUtils
from main import Main

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 megabytes

ALLOWED_EXTENSIONS = ['xlsx', 'csv', 'text']
ALLOWED_MIME_TYPES = ['application/vnd.ms -excel']
AWS_S3_BUCKET_NAME = 'lgfgbucket'
AWS_INPUT_DIR_PATH = "data/input/mtm-combined-entites/"

aws_utils = AwsUtils(ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, AWS_S3_BUCKET_NAME)


@app.route("/")
def hello():
    # return "Hello World!"
    return render_template("Index.html")


@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file-to-s3" not in request.files:
        return "No files key in request.files"
    file = request.files["file-to-s3"]
    if not aws_utils.allowed_file(file.filename) or aws_utils.allowed_mime(file):
        return "FILE FORMAT NOT ALLOWED"
    if file.filename == "":
        return "Please select a file"
    if file:
        aws_utils.upload_file_to_s3(file, AWS_INPUT_DIR_PATH)
        return render_template("Ack.html")
    else:
        return "File not uploaded successfully"


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
