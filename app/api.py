import boto3
from flask import Flask, request, jsonify, render_template
import os
from main import Main

app = Flask(__name__)
# TODO: Keep config outside the code,for now hardcoded
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
AWS_S3_BUCKET_NAME = 'lgfgbucket'
AWS_ACCESS_KEY = 'AKIASVLKCGP2TVCZVKO2'
AWS_SECRET_KEY = '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'
AWS_INPUT_DIR_PATH = "data/input/mtm-combined-entites/"


# Validation to check file extensions
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Method to upload file to S3 bucket
def upload_file_to_s3(file, s3_bucket_name, file_path):
    s3_client = boto3.client(
        service_name='s3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    response = s3_client.upload_fileobj(file, s3_bucket_name, file_path)
    print(f'upload_log_to_aws response: {response}')


@app.route("/")
def hello():
    # return "Hello World!"
    return render_template("Index.html")


@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file-to-s3" not in request.files:
        return "No files key in request.files"
    file = request.files["file-to-s3"]
    if not allowed_file(file.filename):
        return "FILE FORMAT NOT ALLOWED"
    if file.filename == "":
        return "Please select a file"
    if file:
        upload_file_to_s3(file,AWS_S3_BUCKET_NAME,AWS_INPUT_DIR_PATH)
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
