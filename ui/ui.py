import os
import secrets
import yaml
import pandas as pd
import psycopg2
import urllib.parse
import tempfile
import shutil
from utils.aws_utils import AwsUtils
from app.main import Main
from ui.dxf_loader import DXFLoader
from ui.mtm_processor import MTMProcessor
from matplotlib import pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash

load_dotenv()
# Run from project root (no relative path needed)
config_filepath = "tailoring_api_config.yml"
# test_profile = "183295423477_PowerUserAccess"


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
AWS_OUTPUT_DIR_PATH = yaml_config['AWS_OUTPUT_DIR_PATH']
AWS_S3_SIGNATURE_VERSION = yaml_config['AWS_S3_SIGNATURE_VERSION']
AWS_PLOT_DIR_BASE = yaml_config['AWS_PLOT_DIR_BASE']
AWS_PROFILE = yaml_config['AWS_PROFILE']
DB_TYPE = yaml_config['DB_TYPE']
DB_API = yaml_config['DB_API']
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

aws_utils = AwsUtils(ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, AWS_S3_BUCKET_NAME, AWS_S3_SIGNATURE_VERSION, AWS_PROFILE)
connection_string = f"{DB_TYPE}+{DB_API}://{DB_USER}:{urllib.parse.quote(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

'''
File operation service functions
'''


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


'''
Input-formatter service functions
'''


def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST
    )
    return conn


def save_mapped_tags_to_db(tag_name, pieces_list, alterations_list, alteration_amnt):
    engine = create_engine(connection_string)
    try:
        print(
            f"Selected Pieces: {pieces_list}, Selected Alterations: {alterations_list}, Entered Alteration amount: {alteration_amnt},Selected Tag: {tag_name}")
        pieces_tuple = tuple(pieces_list)
        alterations_tuple = tuple(alterations_list)
        # Get the tag_ids,piece_ids and alteration_ids using the input
        tag_id_query = text("""
            SELECT tag_id 
            FROM tags
            WHERE tag_name ilike :tag_name 
        """)

        piece_id_query = text("""
            SELECT piece_id 
            FROM pieces
            WHERE piece_name in :pieces_tuple 
        """)

        alteration_id_query = text("""
            SELECT alteration_id 
            FROM alterations
            WHERE alteration_name in :alterations_tuple 
        """)

        tag_df = pd.read_sql(tag_id_query, engine, params={"tag_name": tag_name})
        tag_id = tag_df['tag_id'].tolist()[0]

        if not len(pieces_tuple) == 0:
            piece_df = pd.read_sql(piece_id_query, engine, params={"pieces_tuple": pieces_tuple})
            piece_ids = piece_df['piece_id'].tolist()
            # For each piece add the tag and insert many-to-many relation values into piece_tag_rel table
            for piece_id in piece_ids:
                data = {
                    'tag_id': tag_id,
                    'piece_id': piece_id
                }
                df_to_insert = pd.DataFrame(data, index=[0])
                check_duplicate_df_query = text("""
                    SELECT tag_id ,piece_id
                    FROM tag_piece_rel
                    WHERE tag_id = :tag_id AND piece_id = :piece_id
                """)
                check_duplicate_df = pd.read_sql(check_duplicate_df_query, engine,
                                                 params={"tag_id": tag_id, "piece_id": piece_id})
                if not (df_to_insert.equals(check_duplicate_df)):
                    df_to_insert.to_sql('tag_piece_rel', engine, if_exists='append', index=False)
                    print("Inserted tag into tag_piece_rel table")
                else:
                    print("Duplicate found , skipping insert")
                    flash("Duplicate found , skipping insert")

        if not len(alterations_tuple) == 0:
            alterations_df = pd.read_sql(alteration_id_query, engine, params={"alterations_tuple": alterations_tuple})
            alteration_ids = alterations_df['alteration_id'].tolist()
            # For each alteration add the tag and insert many-to-many relation values into piece_alteration_rel table
            for alteration_id in alteration_ids:
                data = {
                    'tag_id': tag_id,
                    'alteration_id': alteration_id
                }
                df_to_insert = pd.DataFrame(data, index=[0])
                check_duplicate_df_query = text("""
                    SELECT tag_id ,alteration_id
                    FROM tag_alteration_rel
                    WHERE tag_id = :tag_id AND alteration_id = :alteration_id
                """)
                check_duplicate_df = pd.read_sql(check_duplicate_df_query, engine,
                                                 params={"tag_id": tag_id, "alteration_id": alteration_id})
                if not (df_to_insert.equals(check_duplicate_df)):
                    df_to_insert.to_sql('tag_alteration_rel', engine, if_exists='append', index=False)
                    print("Inserted tag into tag_alteration_rel table")
                else:
                    print("Duplicate found , skipping insert")
                    flash("Duplicate found , skipping insert")

        if not alteration_amnt == '':
            data = {
                'tag_id': tag_id,
                'alteration_amnt': alteration_amnt
            }
            # pd.set_option('display.float_format', lambda x: '%.3f' % x)
            df_to_insert = pd.DataFrame(data, index=[0])
            check_duplicate_df_query = text("""
                                SELECT tag_id ,alteration_amnt
                                FROM tag_alteration_amnt_rel
                                WHERE tag_id = :tag_id AND alteration_amnt = :alteration_amnt
                            """)
            check_duplicate_df = pd.read_sql(check_duplicate_df_query, engine,
                                             params={"tag_id": tag_id, "alteration_amnt": alteration_amnt})
            if not (df_to_insert.equals(check_duplicate_df)):
                df_to_insert.to_sql('tag_alteration_amnt_rel', engine, if_exists='append', index=False)
                print("Inserted tag into tag_alteration_rel table")
            else:
                print("Duplicate found , skipping insert")
                flash("Duplicate found , skipping insert")

        return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        engine.dispose()


def save_piece_to_db(entered_piece, entered_item):
    engine = create_engine(connection_string)
    try:
        data = {
            'piece_name': entered_piece,
            'item': entered_item
        }
        df_to_insert = pd.DataFrame(data, index=[0])
        check_duplicate_df_query = text("""
                                       SELECT piece_name ,item
                                       FROM pieces
                                       WHERE piece_name = :piece_name AND item = :item
                                   """)
        check_duplicate_df = pd.read_sql(check_duplicate_df_query, engine,
                                         params={"piece_name": entered_piece, "item": entered_item})
        if not (df_to_insert.equals(check_duplicate_df)):
            df_to_insert.to_sql('pieces', engine, if_exists='append', index=False)
            print("Inserted tag into pieces table")
        else:
            print("Duplicate found , skipping insert")
            flash("Duplicate found , skipping insert")

        return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        engine.dispose()


def save_alteration_to_db(entered_alteration, entered_item):
    engine = create_engine(connection_string)
    try:
        data = {
            'alteration_name': entered_alteration,
            'item': entered_item
        }
        df_to_insert = pd.DataFrame(data, index=[0])
        check_duplicate_df_query = text("""
                                           SELECT alteration_name ,item
                                           FROM alterations
                                           WHERE alteration_name = :alteration_name AND item = :item
                                       """)
        check_duplicate_df = pd.read_sql(check_duplicate_df_query, engine,
                                         params={"alteration_name": entered_alteration, "item": entered_item})
        if not (df_to_insert.equals(check_duplicate_df)):
            df_to_insert.to_sql('alterations', engine, if_exists='append', index=False)
            print("Inserted tag into alterations table")
        else:
            print("Duplicate found , skipping insert")
            flash("Duplicate found , skipping insert")

        return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        engine.dispose()


def save_tag_to_db(entered_tagname, entered_tagsubcategory, entered_tagcategory):
    engine = create_engine(connection_string)
    try:
        data = {
            'tag_name': entered_tagname,
            'tag_subcategory': entered_tagsubcategory,
            'tag_category': entered_tagcategory,
        }
        df_to_insert = pd.DataFrame(data, index=[0])
        check_duplicate_df_query = text("""
                                               SELECT tag_name ,tag_subcategory,tag_category
                                               FROM tags
                                               WHERE tag_name = :tag_name 
                                               AND tag_subcategory = :tag_subcategory 
                                               AND tag_category = :tag_category 
                                           """)
        check_duplicate_df = pd.read_sql(check_duplicate_df_query, engine,
                                         params={'tag_name': entered_tagname, 'tag_subcategory': entered_tagsubcategory,
                                                 'tag_category': entered_tagcategory, })
        if not (df_to_insert.equals(check_duplicate_df)):
            df_to_insert.to_sql('tags', engine, if_exists='append', index=False)
            print("Inserted tag into tags table")
        else:
            print("Duplicate found , skipping insert")
            flash("Duplicate found , skipping insert")

        return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        engine.dispose()


''' 
MAIN PAGE
'''


@app.route("/home")
def home():
    return render_template("ui_main.html")


''' 
Input formatter service routes
'''


@app.route("/input_formatter_home")
def input_formatter_home():
    return render_template("input_formatter_home.html")


@app.route("/add_piece_form", methods=["POST", "GET"])
def add_piece_form():
    return render_template("add_piece.html")


@app.route("/add_alteration_form", methods=["POST", "GET"])
def add_alteration_form():
    return render_template("add_alteration.html")


@app.route("/add_tag_form", methods=["POST", "GET"])
def add_tag_form():
    return render_template("add_tag.html")


@app.route('/get-options/pieces', methods=['GET'])
def get_options_table1():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT piece_name FROM pieces')
    options = cursor.fetchall()
    conn.close()
    return jsonify([option[0] for option in options])


# Route to get options from Table 2
@app.route('/get-options/alterations', methods=['GET'])
def get_options_table2():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT alteration_name FROM alterations')
    options = cursor.fetchall()
    conn.close()
    return jsonify([option[0] for option in options])


# Route to get options from Table 3
@app.route('/get-options/tags', methods=['GET'])
def get_options_table3():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT tag_name FROM tags')
    options = cursor.fetchall()
    conn.close()
    return jsonify([option[0] for option in options])


@app.route("/map_tags", methods=["POST", "GET"])
def map_tags():
    return render_template("tag_mapper.html")


@app.route("/add_piece", methods=["POST", "GET"])
def add_piece():
    entered_piece = request.form.get('piece_name')
    entered_item = request.form.get('item')
    save_piece_to_db(entered_piece, entered_item)
    return redirect("/add_piece_form")


@app.route("/add_alteration", methods=["POST", "GET"])
def add_alteration():
    entered_alteration = request.form.get('alteration_name')
    entered_item = request.form.get('item')
    save_alteration_to_db(entered_alteration, entered_item)
    return redirect("/add_alteration_form")


@app.route("/add_tag", methods=["POST", "GET"])
def add_tag():
    entered_tagname = request.form.get('tag_name')
    entered_tagsubcategory = request.form.get('tag_subcategory')
    entered_tagcategory = request.form.get('tag_category')
    save_tag_to_db(entered_tagname, entered_tagsubcategory, entered_tagcategory)
    return redirect("/add_tag_form")


# Route to handle form submission
@app.route('/submit', methods=['POST'])
def handle_form_submission():
    # Get selected options from the form
    selected_pieces = request.form.getlist('pieces_options')  # Multiple selections for Pieces (Table 1)
    selected_alterations = request.form.getlist('alterations_options')  # Multiple selections for Alterations (Table 2)
    selected_tag = request.form.get('tag_option')  # Single selection for Tags (Table 3)
    entered_alteration_amnt = request.form.get('alteration_amnt')
    save_mapped_tags_to_db(selected_tag, selected_pieces, selected_alterations, entered_alteration_amnt)
    return redirect("/map_tags")


''' 
File operations service routes
'''


@app.route("/file_operations_home")
def file_operations_home():
    # return "Hello World!"
    with open('tailoring_api_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    return render_template("file_operations_home.html", garment_types=config['GARMENT_TYPES'])


@app.route("/upload_file", methods=["POST", "GET"])
def upload_file():
    if "file-to-s3" not in request.files:
        return "No files key in request.files"

    file = request.files["file-to-s3"]
    file_list = request.files.getlist('file-to-s3')
    item_choice = request.form.get('item_choice')

    if not item_choice:
        return "Please select an item type"

    # First update the paths
    try:
        global ITEM, AWS_DXF_DIR_PATH, AWS_DXF_GRADED_DIR_PATH, AWS_MTM_DIR_PATH, AWS_MTM_GRADED_DIR_PATH, AWS_MTM_DIR_PATH_LABELED

        ITEM = item_choice.lower()
        # Update AWS paths with new ITEM value
        AWS_DXF_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['AWS_DXF_DIR_PATH'], ITEM)
        AWS_DXF_GRADED_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['AWS_DXF_GRADED_DIR_PATH'],
                                                                            ITEM)
        AWS_MTM_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['AWS_MTM_DIR_PATH'], ITEM)
        AWS_MTM_GRADED_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['AWS_MTM_GRADED_DIR_PATH'],
                                                                            ITEM)
        AWS_MTM_DIR_PATH_LABELED = aws_utils.concatenate_item_subdirectories(yaml_config['AWS_MTM_DIR_PATH_LABELED'],
                                                                             ITEM)
    except Exception as e:
        return f"Failed to update paths: {str(e)}"

    # Then proceed with file upload
    if not aws_utils.allowed_file(file.filename) or aws_utils.allowed_mime(file):
        return "FILE FORMAT NOT ALLOWED"
    if file.filename == "":
        return "Please select a file"

    typeform = request.form.get('file_choice')
    if not typeform:
        return "Please select a file type"

    if typeform.lower() == 'dxf_file':
        print(f"Processing DXF file: {file.filename}")
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temp file with the same name as the uploaded file
            temp_file_path = os.path.join(tmpdirname, file.filename)

            # Save the uploaded file to the temp file
            file.save(temp_file_path)

            # with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as temp_file:
            # file.save(temp_file.name)  # Save the uploaded file to the temp file

            # Load DXF using the file path
            # dxf_loader.load_dxf(temp_file.name)
            dxf_loader.load_dxf(temp_file_path)

            # Convert DXF entities to a Pandas DataFrame
            df = dxf_loader.entities_to_dataframe()

            # Create plot
            output_plot_path = os.path.join("ui/static/plots", f"{os.path.splitext(file.filename)[0]}_plot.png")

            # Prepare a matplotlib figure
            fig, ax = plt.subplots(figsize=(60, 50))
            ax.set_aspect('equal')

            # Draw lines
            lines = df[df['Type'] == 'LINE']
            for _, row in lines.iterrows():
                if pd.notna(row['Line_Start_X']) and pd.notna(row['Line_End_X']) and pd.notna(
                        row['Line_Start_Y']) and pd.notna(row['Line_End_Y']):
                    plt.plot([row['Line_Start_X'], row['Line_End_X']], [row['Line_Start_Y'], row['Line_End_Y']],
                             marker='o', linewidth=0.5, markersize=5)
                    plt.text(row['Line_Start_X'], row['Line_Start_Y'],
                             f"({row['Line_Start_X']}, {row['Line_Start_Y']})", fontsize=8, ha='right', va='bottom')
                    plt.text(row['Line_End_X'], row['Line_End_Y'], f"({row['Line_End_X']}, {row['Line_End_Y']})",
                             fontsize=8, ha='right', va='bottom')

            # Draw polylines
            polylines = df[df['Type'].isin(['POLYLINE', 'LWPOLYLINE'])]
            unique_points = polylines.drop_duplicates(subset=['PL_POINT_X', 'PL_POINT_Y'])
            for vertex_label in polylines['Vertex Label'].unique():
                vertex_group = polylines[polylines['Vertex Label'] == vertex_label]
                xs = vertex_group['PL_POINT_X'].tolist()
                ys = vertex_group['PL_POINT_Y'].tolist()
                plt.plot(xs, ys, marker='o', linewidth=0.5, markersize=5)

            # Annotate unique points
            for x, y, point_label in zip(unique_points['PL_POINT_X'], unique_points['PL_POINT_Y'],
                                         unique_points['Point Label']):
                plt.text(x, y, f'{point_label}', fontsize=8, ha='right', va='bottom')

            # Configure and save the plot
            plt.title(f'Polyline Plot for {file.filename}', fontsize=26)
            plt.xlabel('X Coordinate', fontsize=24)
            plt.ylabel('Y Coordinate', fontsize=24)

            # Increase x-tick and y-tick label sizes
            plt.xticks(fontsize=20)  # Increase x-tick size
            plt.yticks(fontsize=20)  # Increase y-tick size

            plt.grid(True)
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Adjust margins to reduce whitespace
            fig.savefig(output_plot_path, dpi=150, bbox_inches='tight')  # Save plot locally with minimal whitespace
            plt.close(fig)

            # Upload plot to S3
            s3_plot_path = os.path.join(AWS_PLOT_DIR_BASE + "/for_labeling",
                                        f"{os.path.splitext(file.filename)[0]}_plot.png")
            aws_utils.upload_file_by_path_to_s3(output_plot_path, s3_plot_path)  # Upload plot to S3

            # Generate presigned URL for the plot
            presigned_plot_url = aws_utils.generate_presigned_url(s3_plot_path)

            ## MTM Stuff: Generate Excel
            sorted_df = df.sort_values(by=['Filename', 'Type', 'Layer'])
            sorted_df['MTM Points'] = ''

            base_filename = os.path.splitext(os.path.basename(file.filename))[0]
            aws_mtm_dir_path = os.path.join(AWS_MTM_DIR_PATH, f"{base_filename}_combined_entities.xlsx")

            with open(temp_file_path, 'rb') as tmp:
                # check for duplicate file using hash value of the file
                file_content = tmp.read()
                file_hash = aws_utils.compute_file_hashValue(file_content)
                tmp.seek(0)
                if not aws_utils.check_hash_exists(file_hash, AWS_DXF_DIR_PATH):
                    aws_utils.upload_file_to_s3(tmp, os.path.join(AWS_DXF_DIR_PATH, file.filename))
                    aws_utils.update_hash_file(AWS_DXF_DIR_PATH)
                    print(f'Successfully uploaded {file.filename} to S3!')
                else:
                    print(f'DUPLICATE file : {file.filename} ')
                    return "DUPLICATE file detected. Upload a different file"

        aws_utils.upload_dataframe_to_s3(sorted_df, aws_mtm_dir_path, file_format="excel")

        # Generate a presigned URL for the CSV file
        presigned_csv_url = aws_utils.generate_presigned_url(aws_mtm_dir_path)

        # return render_template("download.html", download_url=presigned_url)
        return render_template("display_and_download_dxf.html", plot_url=presigned_plot_url,
                               csv_url=presigned_csv_url)

    if typeform.lower() == 'graded_dxf_file':
        with tempfile.TemporaryDirectory() as tmpdirname:
            for each_file in file_list:
                file_base_name = each_file.filename

                print(f"Processing Graded DXF file: {file_base_name}")
                # Create a temp file with the same name as the uploaded file
                temp_file_path = os.path.join(tmpdirname, file_base_name)
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                # Save the uploaded file to the temp file
                each_file.save(temp_file_path)

                # with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as temp_file:
                # file.save(temp_file.name)  # Save the uploaded file to the temp file

                # Load DXF using the file path
                # dxf_loader.load_dxf(temp_file.name)
                dxf_loader.load_dxf(temp_file_path)

                # Convert DXF entities to a Pandas DataFrame
                df = dxf_loader.entities_to_dataframe()

                ## MTM Stuff: Generate Excel
                sorted_df = df.sort_values(by=['Filename', 'Type', 'Layer'])
                sorted_df['MTM Points'] = ''

                # base_filename = os.path.splitext(os.path.basename(each_file.filename))[0]
                aws_mtm_graded_dir_path = os.path.join(AWS_MTM_GRADED_DIR_PATH,
                                                       f"{file_base_name}_combined_entities.xlsx")

                aws_utils.upload_dataframe_to_s3(sorted_df, aws_mtm_graded_dir_path, file_format="excel")

                with open(temp_file_path, 'rb') as tmp:
                    # check for duplicate file using hash value of the file
                    file_content = tmp.read()
                    file_hash = aws_utils.compute_file_hashValue(file_content)
                    tmp.seek(0)
                    if not aws_utils.check_hash_exists(file_hash, AWS_DXF_GRADED_DIR_PATH):
                        aws_utils.upload_file_to_s3(tmp, os.path.join(AWS_DXF_GRADED_DIR_PATH, file_base_name))
                        aws_utils.update_hash_file(AWS_DXF_GRADED_DIR_PATH)
                        print(f'Successfully uploaded {file_base_name} to S3!')
                    else:
                        print(f'DUPLICATE file : {file_base_name} ')
                        return f"DUPLICATE file detected:{file_base_name} ,Upload a different file"

    if typeform.lower() == 'mtm_points_file':
        print(f"Processing MTM points file: {file.filename}")
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as temp_file:
        # file.save(temp_file.name)
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a temp file with the same name as the uploaded file
            temp_file_path = os.path.join(tmpdirname, file.filename)

            # Save the uploaded file to the temp file
            file.save(temp_file_path)
            mtm_processor = MTMProcessor()
            mtm_processor.load_coordinates_tables(temp_file_path)
            mtm_processor.remove_nan_mtm_points()
            # mtm_processor.display_filtered_coordinates_tables()

            # Remove the extension
            table_name = os.path.splitext(file.filename)[0]
            piece_name = table_name.split('_')[0]
            print(f"Piece Name: {piece_name}")

            plot_filename, plot_file_path = mtm_processor.plot_points(rename=piece_name)

            aws_mtm_plot_dir_path = os.path.join(AWS_PLOT_DIR_BASE + "/labeled_mtm/", f"{plot_filename}_base.png")
            aws_utils.upload_file_by_path_to_s3(plot_file_path, aws_mtm_plot_dir_path)
            print(f"Plot Filename {plot_filename}")

            with open(temp_file_path, 'rb') as tmp:
                # check for duplicate file using hash value of the file
                file_content = tmp.read()
                file_hash = aws_utils.compute_file_hashValue(file_content)
                tmp.seek(0)
                if not aws_utils.check_hash_exists(file_hash, AWS_MTM_DIR_PATH_LABELED):
                    aws_utils.upload_file_to_s3(tmp, os.path.join(AWS_MTM_DIR_PATH_LABELED, file.filename))
                    aws_utils.update_hash_file(AWS_MTM_DIR_PATH_LABELED)
                    print(f'Successfully uploaded {file.filename} to S3!')
                else:
                    print(f'DUPLICATE file : {file.filename} ')
                    return "DUPLICATE file detected. Upload a different file"

        # Delete the temp file after processing (optional)
        # os.remove(temp_file.name)

        # aws_utils.upload_file_to_s3(file, AWS_MTM_DIR_PATH_LABELED)

        # Generate a presigned URL for the plot file in S3
        plot_presigned_url = aws_utils.generate_presigned_url(aws_mtm_plot_dir_path)

        # Clear the static/plots folder after everything is processed
        clear_static_plots_folder()

        # Render the plot on the page by passing the presigned URL to the template
        return render_template("display_mtm_plot.html", plot_url=plot_presigned_url)

    return redirect("/file_operations_home")


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
    app.secret_key = secrets.token_hex(16)
    app.run(host='0.0.0.0', port=5000)