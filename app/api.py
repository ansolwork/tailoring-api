from flask import Flask, request, jsonify
from create_table import CreateTable
import os

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/upload_dxf")
def ui_upload_dxf():
    return "Placeholder"

@app.route("/test", methods=['POST'])
def run_create_table():
    # Get input data from the request
    alteration_filepath = request.json.get('alteration_filepath')
    combined_entities_folder = request.json.get('combined_entities_folder')

    # Check if the provided paths are valid
    if not os.path.exists(alteration_filepath):
        return jsonify({"error": "Alteration file path does not exist."}), 400
    if not os.path.isdir(combined_entities_folder):
        return jsonify({"error": "Combined entities folder does not exist."}), 400
    
    try:
        # Initialize the CreateTable object
        create_table = CreateTable(alteration_filepath, combined_entities_folder)

        # Process the sheets and get the combined DataFrame
        create_table.process_table()
        create_table.process_combined_entities()

        # Create Vertices DF
        create_table.create_vertices_df()

        # Join tables
        create_table.merge_tables()
        
        # Save the combined DataFrame as CSV files 
        
        # TODO: Save to S3
        create_table.save_table_csv()
        create_table.add_other_mtm_points()

        return jsonify({"message": "Table creation and processing completed successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
