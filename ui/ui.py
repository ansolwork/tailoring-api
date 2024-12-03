import os
import secrets
import yaml
import pandas as pd
import psycopg2
import urllib.parse
import tempfile
import shutil
import ast
from utils.aws_utils import AwsUtils
from app.main import Main
from ui.dxf_loader import DXFLoader
from ui.mtm_processor import MTMProcessor
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
from matplotlib import pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, send_file
import logging
from werkzeug.utils import secure_filename
import time
from flask import after_this_request
import zipfile

load_dotenv()
# Run from project root (no relative path needed)
config_filepath = "tailoring_api_config.yml"
# test_profile = "183295423477_PowerUserAccess"


app = Flask(__name__, template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 MB as per file upload limit
# Get secret key from environment variable or generate a new one
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))

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
AWS_PROFILE = yaml_config['AWS_PROFILE']
UI_AWS_PLOT_DIR_BASE = yaml_config['UI_AWS_PLOT_DIR_BASE']
UI_AWS_PLOT_DIR_GRADED = yaml_config['UI_AWS_PLOT_DIR_GRADED']
UI_AWS_MTM_LABELED_DIR_PATH = yaml_config['UI_AWS_MTM_LABELED_DIR_PATH']
UI_AWS_MTM_GRADED_LABELED_DIR_PATH = yaml_config['UI_AWS_MTM_GRADED_LABELED_DIR_PATH']
DB_TYPE = yaml_config['DB_TYPE']
DB_API = yaml_config['DB_API']
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')



aws_utils = AwsUtils(ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, AWS_S3_BUCKET_NAME, AWS_S3_SIGNATURE_VERSION, AWS_PROFILE)
connection_string = f"{DB_TYPE}+{DB_API}://{DB_USER}:{urllib.parse.quote(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Setup logging
logging.basicConfig(
    #level=logging.DEBUG,  # Set to logging.DEBUG to see debug messages
    # Set to logging.INFO for production
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

'''
File operation service functions
'''


# Function to clear static/plots folder
def clear_static_plots_folder(folder_path="ui/static/plots/"):
    try:
        # Ensure the directory exists
        os.makedirs(folder_path, exist_ok=True)
        
        logger.info(f"Clearing plots folder: {folder_path}")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.debug(f"Removed directory: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
                    
    except Exception as e:
        logger.error(f"Error clearing static plots folder: {e}")

    


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


def save_mapped_tags_to_db(tag_name, tag_subcategory, pieces_list, alterations_list, alteration_amnt):
    engine = create_engine(connection_string)
    try:
        logger.info(f"Processing tag mapping: {tag_name} ({tag_subcategory})")
        logger.debug(
            f"Full details - Pieces: {pieces_list}, Alterations: {alterations_list}, "
            f"Alteration amount: {alteration_amnt}"
        )
        
        # Validate required parameters
        if not tag_name or tag_subcategory is None:
            error_msg = "Tag name and subcategory are required"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        pieces_tuple = tuple(pieces_list)
        alterations_tuple = tuple(alterations_list)
        
        # Get tag_id with error handling
        tag_id_query = text("""
            SELECT tag_id 
            FROM tags
            WHERE tag_name = :tag_name 
            AND tag_subcategory = :tag_subcategory
        """)

        with engine.connect() as conn:
            tag_df = pd.read_sql(tag_id_query, conn, params={
                "tag_name": tag_name,
                "tag_subcategory": tag_subcategory
            })
            
            if tag_df.empty:
                error_msg = f"No tag found with name '{tag_name}' and subcategory '{tag_subcategory}'"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 404
                
            tag_id = int(tag_df['tag_id'].iloc[0])
            logger.debug(f"Found tag_id: {tag_id}")

            # Process pieces
            if pieces_tuple:
                piece_id_query = text("""
                    SELECT piece_id 
                    FROM pieces
                    WHERE piece_name in :pieces_tuple 
                """)
                piece_df = pd.read_sql(piece_id_query, conn, params={"pieces_tuple": pieces_tuple})
                piece_ids = [int(pid) for pid in piece_df['piece_id'].tolist()]
                
                # Create DataFrame for tag-piece relations
                tag_piece_data = {
                    'tag_id': [tag_id] * len(piece_ids),
                    'piece_id': piece_ids
                }
                tag_piece_df = pd.DataFrame(tag_piece_data)
                
                # Check for duplicates
                check_duplicate_query = text("""
                    SELECT tag_id, piece_id
                    FROM tag_piece_rel
                    WHERE tag_id = :tag_id 
                    AND piece_id IN :piece_ids
                """)
                existing_df = pd.read_sql(check_duplicate_query, conn, 
                                        params={"tag_id": tag_id, "piece_ids": tuple(piece_ids)})
                
                # Remove duplicates
                if not existing_df.empty:
                    tag_piece_df = tag_piece_df.merge(existing_df, 
                                                     on=['tag_id', 'piece_id'], 
                                                     how='left', 
                                                     indicator=True)
                    tag_piece_df = tag_piece_df[tag_piece_df['_merge'] == 'left_only']
                    tag_piece_df = tag_piece_df.drop('_merge', axis=1)
                    
                # Insert new relations
                if not tag_piece_df.empty:
                    tag_piece_df.to_sql('tag_piece_rel', conn, if_exists='append', index=False)
                    logger.info(f"Added {len(tag_piece_df)} new piece mappings for tag: {tag_name}")
                else:
                    logger.info("No new tag-piece relations to insert")

            # Process alterations
            if alterations_tuple:
                alteration_id_query = text("""
                    SELECT alteration_id 
                    FROM alterations
                    WHERE alteration_name in :alterations_tuple 
                """)
                alterations_df = pd.read_sql(alteration_id_query, conn, params={"alterations_tuple": alterations_tuple})
                alteration_ids = [int(aid) for aid in alterations_df['alteration_id'].tolist()]
                
                tag_alteration_data = {
                    'tag_id': [tag_id] * len(alteration_ids),
                    'alteration_id': alteration_ids
                }
                tag_alteration_df = pd.DataFrame(tag_alteration_data)
                
                # Check for duplicates
                check_duplicate_query = text("""
                    SELECT tag_id, alteration_id
                    FROM tag_alteration_rel
                    WHERE tag_id = :tag_id 
                    AND alteration_id IN :alteration_ids
                """)
                existing_df = pd.read_sql(check_duplicate_query, conn, 
                                        params={"tag_id": tag_id, "alteration_ids": tuple(alteration_ids)})
                
                # Remove duplicates
                if not existing_df.empty:
                    tag_alteration_df = tag_alteration_df.merge(existing_df, 
                                                              on=['tag_id', 'alteration_id'], 
                                                              how='left', 
                                                              indicator=True)
                    tag_alteration_df = tag_alteration_df[tag_alteration_df['_merge'] == 'left_only']
                    tag_alteration_df = tag_alteration_df.drop('_merge', axis=1)
                    
                # Insert new relations
                if not tag_alteration_df.empty:
                    tag_alteration_df.to_sql('tag_alteration_rel', conn, if_exists='append', index=False)
                    logger.info(f"Added {len(tag_alteration_df)} new alteration mappings for tag: {tag_name}")
                else:
                    logger.info("No new tag-alteration relations to insert")

            # Process alteration amount
            if alteration_amnt:
                tag_amount_data = {
                    'tag_id': [tag_id],
                    'alteration_amnt': [alteration_amnt.strip()]
                }
                tag_amount_df = pd.DataFrame(tag_amount_data)
                
                # Check for duplicates
                check_duplicate_query = text("""
                    SELECT tag_id, alteration_amnt
                    FROM tag_alteration_amnt_rel
                    WHERE tag_id = :tag_id 
                    AND alteration_amnt = :alteration_amnt
                """)
                existing_df = pd.read_sql(check_duplicate_query, conn, 
                                        params={"tag_id": tag_id, "alteration_amnt": alteration_amnt.strip()})
                
                # Insert if no duplicate
                if existing_df.empty:
                    tag_amount_df.to_sql('tag_alteration_amnt_rel', conn, if_exists='append', index=False)
                    logger.info(f"Added alteration amount {alteration_amnt} for tag: {tag_name}")
                else:
                    logger.debug(f"Skipped duplicate alteration amount for tag: {tag_name}")

            conn.commit()
            logger.info(f"Successfully completed tag mapping for: {tag_name}")
            return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        logger.error(f"Error in save_mapped_tags_to_db: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        engine.dispose()


def save_piece_to_db(entered_piece, entered_item):
    engine = create_engine(connection_string)
    try:
        if not all([entered_piece, entered_item]):
            flash("All fields (Piece Name and Item) are required", "error")
            return jsonify({"error": "Missing required fields"}), 400

        # Clean and prepare input data
        entered_piece = entered_piece.strip().upper()
        entered_item = entered_item.strip().lower()

        # Check for duplicates using direct SQL query
        check_duplicate_df_query = text("""
            SELECT COUNT(*) 
            FROM pieces 
            WHERE UPPER(piece_name) = :piece_name 
            AND LOWER(item) = :item
        """)
        
        with engine.connect() as conn:
            result = conn.execute(
                check_duplicate_df_query, 
                {"piece_name": entered_piece, "item": entered_item}
            ).scalar()
            
            if result > 0:
                logger.info("Duplicate found, skipping insert")
                flash("Duplicate found, skipping insert", "warning")
                return jsonify({"message": "Duplicate found"})
            
            # No duplicate found, proceed with insert
            data = {
                'piece_name': entered_piece,
                'item': entered_item
            }
            df_to_insert = pd.DataFrame(data, index=[0])
            df_to_insert.to_sql('pieces', engine, if_exists='append', index=False)
            logger.info("Inserted piece into pieces table")
            flash("Successfully added new piece!", "success")
            return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        logger.error(f"Error in save_piece_to_db: {str(e)}")
        flash(f"Error: {str(e)}", "error")
        return jsonify({"error": str(e)}), 500

    finally:
        engine.dispose()


def save_alteration_to_db(entered_alteration, entered_item):
    engine = create_engine(connection_string)
    try:
        if not all([entered_alteration, entered_item]):
            flash("All fields (Alteration Name and Item) are required", "error")
            return jsonify({"error": "Missing required fields"}), 400

        # Clean and prepare input data
        entered_alteration = entered_alteration.strip().upper()
        entered_item = entered_item.strip().lower()

        # Check for duplicates using direct SQL query
        check_duplicate_df_query = text("""
            SELECT COUNT(*) 
            FROM alterations 
            WHERE UPPER(alteration_name) = :alteration_name 
            AND LOWER(item) = :item
        """)
        
        with engine.connect() as conn:
            result = conn.execute(
                check_duplicate_df_query, 
                {"alteration_name": entered_alteration, "item": entered_item}
            ).scalar()
            
            if result > 0:
                logger.info("Duplicate found, skipping insert")
                flash("Duplicate found, skipping insert", "warning")
                return jsonify({"message": "Duplicate found"})
            
            # No duplicate found, proceed with insert
            data = {
                'alteration_name': entered_alteration,
                'item': entered_item
            }
            df_to_insert = pd.DataFrame(data, index=[0])
            df_to_insert.to_sql('alterations', engine, if_exists='append', index=False)
            logger.info("Inserted alteration into alterations table")
            flash("Successfully added new alteration!", "success")
            return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        logger.error(f"Error in save_alteration_to_db: {str(e)}")
        flash(f"Error: {str(e)}", "error")
        return jsonify({"error": str(e)}), 500

    finally:
        engine.dispose()


def save_tag_to_db(entered_tagname, entered_tagsubcategory, entered_tagcategory, entered_tag_item):
    engine = create_engine(connection_string)
    try:
        if not all([entered_tagname, entered_tagsubcategory, entered_tagcategory, entered_tag_item]):
            flash("All fields (Tag Name, Subcategory, Category, and Item) are required", "error")
            return jsonify({"error": "Missing required fields"}), 400

        data = {
            'tag_name': entered_tagname.strip(),
            'tag_subcategory': entered_tagsubcategory.strip(),
            'tag_category': entered_tagcategory.strip(),
            'tag_item': entered_tag_item.strip()
        }
        df_to_insert = pd.DataFrame(data, index=[0])
        check_duplicate_df_query = text("""
                                               SELECT tag_name ,tag_subcategory,tag_category
                                               FROM tags
                                               WHERE LOWER(tag_name) = LOWER(:tag_name) 
                                               AND LOWER(tag_subcategory) = LOWER(:tag_subcategory) 
                                               AND LOWER(tag_category) = LOWER(:tag_category) 
                                               AND LOWER(tag_item) = LOWER(:tag_item)
                                           """)
        check_duplicate_df = pd.read_sql(check_duplicate_df_query, engine,
                                         params={'tag_name': entered_tagname, 'tag_subcategory': entered_tagsubcategory,
                                                 'tag_category': entered_tagcategory, 'tag_item': entered_tag_item})
        if not (df_to_insert.equals(check_duplicate_df)):
            df_to_insert.to_sql('tags', engine, if_exists='append', index=False)
            logger.info("Inserted tag into tags table")
            flash("Successfully added new tag!", "success")
        else:
            logger.info("Duplicate found, skipping insert")
            flash("Duplicate found, skipping insert", "warning")

        return jsonify({"message": "Processing completed successfully."})

    except Exception as e:
        flash(f"Error: {str(e)}", "error")
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
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT piece_name FROM pieces')
        options = cursor.fetchall()
        logger.debug(f"Retrieved {len(options)} pieces from database")
        return jsonify([option[0] for option in options])
    except Exception as e:
        logger.error(f"Error retrieving pieces: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# Route to get options from Table 2
@app.route('/get-options/alterations', methods=['GET'])
def get_options_table2():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT alteration_name FROM alterations')
        options = cursor.fetchall()
        logger.debug(f"Retrieved {len(options)} alterations from database")
        return jsonify([option[0] for option in options])
    except Exception as e:
        logger.error(f"Error retrieving alterations: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# Route to get options from Table 3
@app.route('/get-options/tags', methods=['GET'])
def get_options_table3():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT tag_name, tag_subcategory FROM tags')
        options = cursor.fetchall()
        logger.debug(f"Retrieved {len(options)} tags from database")
        return jsonify([{"tag_name": option[0], "tag_subcategory": option[1]} for option in options])
    except Exception as e:
        logger.error(f"Error retrieving tags: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


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
    entered_tag_item = request.form.get('tag_item')
    save_tag_to_db(entered_tagname, entered_tagsubcategory, entered_tagcategory, entered_tag_item)
    return redirect("/add_tag_form")


# Route to handle form submission
@app.route('/submit', methods=['POST'])
def handle_form_submission():
    try:
        # Log the raw form data for debugging
        logger.info("Raw form data received:")
        for key, value in request.form.items():
            logger.info(f"{key}: {value}")

        selected_pieces = request.form.getlist('pieces_options')
        selected_alterations = request.form.getlist('alterations_options')
        tag_combined = request.form.get('tag_option', '')
        entered_alteration_amnt = request.form.get('alteration_amnt')

        # Parse the combined tag value
        if '|' not in tag_combined:
            logger.error(f"Invalid tag format received: {tag_combined}")
            flash("Please select a valid tag with subcategory")
            return redirect("/map_tags")

        selected_tag, selected_tag_subcategory = tag_combined.split('|')
        
        logger.info(f"Parsed values - Tag: {selected_tag}, Subcategory: {selected_tag_subcategory}")

        if not selected_tag or not selected_tag_subcategory:
            flash("Tag and subcategory are required")
            return redirect("/map_tags")

        result = save_mapped_tags_to_db(
            selected_tag.strip(), 
            selected_tag_subcategory.strip(), 
            selected_pieces, 
            selected_alterations, 
            entered_alteration_amnt
        )

        if isinstance(result, tuple):
            response, status_code = result
            if status_code >= 400:
                flash(response.json.get('error', 'An error occurred'))
            else:
                flash("Tags mapped successfully!")
        else:
            flash("Tags mapped successfully!")
            
        return redirect("/map_tags")

    except Exception as e:
        logger.error(f"Error in form submission: {str(e)}")
        flash("An error occurred while processing your request")
        return redirect("/map_tags")


''' 
File operations service routes
'''


@app.route("/file_operations_home")
def file_operations_home():
    # Clear any existing plots and temp files when returning to home
    clear_static_plots_folder()
    cleanup_temp_files()
    
    # Create necessary directories if they don't exist
    os.makedirs('ui/static/plots', exist_ok=True)
    os.makedirs('ui/temp', exist_ok=True)
    
    with open('tailoring_api_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    return render_template("file_operations_home.html", garment_types=config['GARMENT_TYPES'])


@app.route("/upload_file", methods=["POST", "GET"])
def upload_file():
    # Create necessary directories if they don't exist
    os.makedirs('ui/static/plots', exist_ok=True)
    os.makedirs('ui/temp', exist_ok=True)
    
    if "file-to-s3" not in request.files:
        return "No files key in request.files"

    file = request.files["file-to-s3"]
    file_list = request.files.getlist('file-to-s3')
    item_choice = request.form.get('item_choice')
    piece_choice = request.form.get('piece_choice')
    if not item_choice and not piece_choice:
        return "Please select an item type and piece type"

    # First update the paths
    try:
        global ITEM, UI_AWS_DXF_DIR_PATH, UI_AWS_DXF_GRADED_DIR_PATH, UI_AWS_MTM_LABELED_DIR_PATH, UI_AWS_MTM_GRADED_LABELED_DIR_PATH, UI_AWS_PLOT_DIR_BASE, UI_AWS_PLOT_DIR_GRADED, UI_AWS_MTM_FROM_DXF_DIR_PATH, UI_AWS_GRADING_RULE_DIR_PATH,UI_AWS_MTM_FROM_DXF_GRADED_DIR_PATH

        ITEM = item_choice.lower()
        # Update AWS paths with new ITEM value
        #DXF paths
        UI_AWS_DXF_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_DXF_DIR_PATH'], ITEM)
        UI_AWS_DXF_GRADED_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_DXF_GRADED_DIR_PATH'],
                                                                            ITEM)
        
        #MTM paths
        UI_AWS_MTM_LABELED_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_MTM_LABELED_DIR_PATH'],
                                                                             ITEM)
        UI_AWS_MTM_GRADED_LABELED_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_MTM_GRADED_LABELED_DIR_PATH'],
                                                                                      ITEM)
        UI_AWS_MTM_FROM_DXF_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_MTM_FROM_DXF_DIR_PATH'],
                                                                               ITEM)
        UI_AWS_MTM_FROM_DXF_GRADED_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_MTM_FROM_DXF_GRADED_DIR_PATH'],
                                                                                      ITEM)
        
        UI_AWS_GRADING_RULE_DIR_PATH = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_GRADING_RULE_DIR_PATH'],
                                                                               ITEM)
        
        #Plot paths
        UI_AWS_PLOT_DIR_BASE = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_PLOT_DIR_BASE'], ITEM)
        UI_AWS_PLOT_DIR_GRADED = aws_utils.concatenate_item_subdirectories(yaml_config['UI_AWS_PLOT_DIR_GRADED'], ITEM)

        
    except Exception as e:
        return f"Failed to update paths: {str(e)}"

    # Then proceed with file upload
    if not aws_utils.allowed_file(file.filename) or aws_utils.allowed_mime(file):
        return "FILE FORMAT NOT ALLOWED"
    if file.filename == "":
        return "Please select a file"

    file_choice = request.form.get('file_choice')
    if not file_choice:
        return "Please select a file type"

    if file_choice.lower() == 'dxf_file':
        logger.info(f"Processing DXF file(s)")
        piece_choice = request.form.get('piece_choice')
        
        if piece_choice.lower() == 'single_piece':
            # Existing single file processing code
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Create a temp file with the same name as the uploaded file
                temp_file_path = os.path.join(tmpdirname, file.filename)

                # Save the uploaded file to the temp file
                file.save(temp_file_path)

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
                s3_plot_path = os.path.join(UI_AWS_PLOT_DIR_BASE + "/for_labeling",
                                            f"{os.path.splitext(file.filename)[0]}_plot.png")
                aws_utils.upload_file_by_path_to_s3(output_plot_path, s3_plot_path)  # Upload plot to S3

                # Generate presigned URL for the plot
                presigned_plot_url = aws_utils.generate_presigned_url(s3_plot_path)

                ## MTM Stuff: Generate Excel
                sorted_df = df.sort_values(by=['Filename', 'Type', 'Layer'])
                sorted_df['MTM Points'] = ''

                base_filename = os.path.splitext(os.path.basename(file.filename))[0]
                aws_mtm_dir_path = os.path.join(UI_AWS_MTM_FROM_DXF_DIR_PATH, f"{base_filename}_combined_entities.xlsx")

                with open(temp_file_path, 'rb') as tmp:
                    # check for duplicate file using hash value of the file
                    file_content = tmp.read()
                    file_hash = aws_utils.compute_file_hashValue(file_content)
                    tmp.seek(0)
                    if not aws_utils.check_hash_exists(file_hash, UI_AWS_DXF_DIR_PATH):
                        aws_utils.upload_file_to_s3(tmp, os.path.join(UI_AWS_DXF_DIR_PATH, file.filename))
                        aws_utils.update_hash_file(UI_AWS_DXF_DIR_PATH)
                        logger.info(f'Successfully uploaded {file.filename} to S3!')
                    else:
                        logger.warning(f'DUPLICATE file : {file.filename} ')
                        return "DUPLICATE file detected. Upload a different file"

            aws_utils.upload_dataframe_to_s3(sorted_df, aws_mtm_dir_path, file_format="excel")

            # Generate a presigned URL for the CSV file
            presigned_csv_url = aws_utils.generate_presigned_url(aws_mtm_dir_path)

            # return render_template("download.html", download_url=presigned_url)
            return render_template("display_and_download_dxf.html", plot_url=presigned_plot_url,
                                   csv_url=presigned_csv_url)

        elif piece_choice.lower() == 'graded_piece':
            try:
                if not file_list:
                    flash("No files were uploaded", "error")
                    return redirect(url_for('file_operations_home'))

                successful_uploads = []
                
                with tempfile.TemporaryDirectory() as tmpdirname:
                    for each_file in file_list:
                        try:
                            file_base_name = each_file.filename
                            
                            if not aws_utils.allowed_file(file_base_name):
                                flash(f"File type not allowed: {file_base_name}", "error")
                                continue
                                
                            logger.info(f"Processing Graded DXF file: {file_base_name}")
                            
                            # Create temp file path
                            temp_file_path = os.path.join(tmpdirname, file_base_name)
                            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                            # Save the uploaded file
                            each_file.save(temp_file_path)

                            # Load and process DXF
                            dxf_loader.load_dxf(temp_file_path)
                            df = dxf_loader.entities_to_dataframe()
                            sorted_df = df.sort_values(by=['Filename', 'Type', 'Layer'])
                            sorted_df['MTM Points'] = ''

                            # Upload to S3
                            aws_mtm_graded_dir_path = os.path.join(UI_AWS_MTM_FROM_DXF_GRADED_DIR_PATH,
                                                               f"{os.path.splitext(file_base_name)[0]}_combined_entities.xlsx")
                            aws_utils.upload_dataframe_to_s3(sorted_df, aws_mtm_graded_dir_path, file_format="excel")

                            # Clean up DataFrames
                            del df
                            del sorted_df

                            # Check for duplicates and upload
                            with open(temp_file_path, 'rb') as tmp:
                                file_content = tmp.read()
                                file_hash = aws_utils.compute_file_hashValue(file_content)
                                tmp.seek(0)
                                
                                if not aws_utils.check_hash_exists(file_hash, UI_AWS_DXF_GRADED_DIR_PATH):
                                    aws_utils.upload_file_to_s3(tmp, os.path.join(UI_AWS_DXF_GRADED_DIR_PATH, file_base_name))
                                    aws_utils.update_hash_file(UI_AWS_DXF_GRADED_DIR_PATH)
                                    logger.info(f'Successfully uploaded {file_base_name} to S3')
                                    successful_uploads.append(file_base_name)
                                else:
                                    logger.warning(f'Duplicate file detected: {file_base_name}')
                                    flash(f"Duplicate file skipped: {file_base_name}", "warning")

                        except Exception as e:
                            logger.error(f"Error processing file {file_base_name}: {str(e)}")
                            flash(f"Error processing file {file_base_name}: {str(e)}", "error")
                            continue

                if successful_uploads:
                    flash(f"Successfully processed files: {', '.join(successful_uploads)}", "success")
                    logger.info(f"Successfully processed files: {', '.join(successful_uploads)}")
                    logger.info(f"Total files processed: {len(file_list)}")
                
                return redirect(url_for('file_operations_home'))
                
            except Exception as e:
                logger.error(f"Error in graded piece processing: {str(e)}")
                flash("Error processing graded pieces", "error")
                return redirect(url_for('file_operations_home'))

    if file_choice.lower() == 'grading_rule_file':
        logger.info(f"Processing grading rule file: {file.filename}")
        try:
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join('ui', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save the uploaded file to temp directory
            temp_file_path = os.path.join(temp_dir, file.filename)
            file.save(temp_file_path)
            
            # Now process the saved file
            with open(temp_file_path, 'rb') as tmp:
                file_content = tmp.read()
                file_hash = aws_utils.compute_file_hashValue(file_content)
                tmp.seek(0)
                
                if not aws_utils.check_hash_exists(file_hash, UI_AWS_GRADING_RULE_DIR_PATH):
                    aws_utils.upload_file_to_s3(tmp, os.path.join(UI_AWS_GRADING_RULE_DIR_PATH, file.filename))
                    aws_utils.update_hash_file(UI_AWS_GRADING_RULE_DIR_PATH)
                    logger.info(f'Successfully uploaded {file.filename} to S3!')
                    flash('File uploaded successfully!', 'success')
                else:
                    logger.warning(f'DUPLICATE file: {file.filename}')
                    flash('DUPLICATE file detected. Upload a different file', 'warning')
                    
            # Clean up temp file
            os.remove(temp_file_path)
            return redirect(url_for('file_operations_home'))
            
        except Exception as e:
            logger.error(f"Error processing grading rule file: {str(e)}")
            flash(f"Error uploading file: {str(e)}", 'error')
            return redirect(url_for('file_operations_home'))

    if file_choice.lower() == 'unlabeled_file':
        logger.info(f"Processing unlabeled file: {file.filename}")
        try:
            # Save to temp directory
            temp_dir = os.path.join('ui', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, file.filename)
            file.save(temp_file_path)

            # Load the unlabeled file
            df = pd.read_excel(temp_file_path)

            # Create plots directory if it doesn't exist
            plots_dir = os.path.join('ui', 'static', 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            output_plot_path = os.path.join(plots_dir, f"{os.path.splitext(file.filename)[0]}_unlabeled_plot.png")

            # Initialize scaling factor
            scaling_factor = None
            
            # Create larger figure
            plt.figure(figsize=(200, 200))
            
            # Calculate scaling factor based on data extents
            x_min = min(df['Point_X'].min(), df['Line_Start_X'].min(), df['Line_End_X'].min())
            x_max = max(df['Point_X'].max(), df['Line_Start_X'].max(), df['Line_End_X'].max())
            y_min = min(df['Point_Y'].min(), df['Line_Start_Y'].min(), df['Line_End_Y'].min())
            y_max = max(df['Point_Y'].max(), df['Line_Start_Y'].max(), df['Line_End_Y'].max())

            # Calculate scaling factor with safety checks
            data_width = x_max - x_min
            data_height = y_max - y_min
            
            # Add safety checks for scaling factor
            if data_width == 0 or data_height == 0:
                scaling_factor = 1.0  # Use default scaling if dimensions are zero
            else:
                scaling_factor = min(80 / data_width, 80 / data_height) * 0.9
            
            vertex_points = df[pd.notna(df['PL_POINT_X'])]
            
            # Initialize lists to collect ALL coordinates
            x_coords_all = []
            y_coords_all = []
            
            # First plot the polygon/vertices
            for _, row in df.iterrows():
                if pd.notna(row['Vertices']):
                    vertices = ast.literal_eval(row['Vertices'])
                    x_coords, y_coords = zip(*vertices)
                    # Scale the coordinates
                    x_coords = [x * scaling_factor for x in x_coords]
                    y_coords = [y * scaling_factor for y in y_coords]
                    plt.plot(x_coords, y_coords, 'b-', linewidth=8.0)  # Thick blue line
                    x_coords_all.extend(x_coords)
                    y_coords_all.extend(y_coords)
            
            # Then plot the points
            if not vertex_points.empty:
                scaled_x = vertex_points['PL_POINT_X'] * scaling_factor
                scaled_y = vertex_points['PL_POINT_Y'] * scaling_factor
                
                plt.scatter(scaled_x, scaled_y, 
                           c='red', s=2000, zorder=5, alpha=0.7)
                
                # Add labels
                for _, row in vertex_points.iterrows():
                    if pd.notna(row.get('Point Label')):
                        plt.annotate(
                            str(row['Point Label']), 
                            (row['PL_POINT_X'] * scaling_factor, row['PL_POINT_Y'] * scaling_factor),
                            xytext=(40, 40),
                            textcoords='offset points',
                            fontsize=32,
                            color='black',
                            weight='bold',
                            bbox=dict(facecolor='white', 
                                     edgecolor='none',
                                     alpha=0.7,
                                     pad=1.5),
                            ha='center',
                            va='center'
                        )
            
            # Calculate the data extent from all coordinates
            if x_coords_all and y_coords_all:
                x_min, x_max = min(x_coords_all), max(x_coords_all)
                y_min, y_max = min(y_coords_all), max(y_coords_all)
                
                # Add small padding (3%)
                x_padding = (x_max - x_min) * 0.03
                y_padding = (y_max - y_min) * 0.03
                
                # Set limits with padding
                ax = plt.gca()
                ax.set_xlim(x_min - x_padding, x_max + x_padding)
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
                ax.set_aspect('equal', adjustable='box')
            
            # Styling
            plt.grid(True, which='major', linestyle='-', linewidth=1.5, color='gray', alpha=0.5)
            plt.grid(True, which='minor', linestyle=':', linewidth=0.8, color='gray', alpha=0.3)
            plt.minorticks_on()
            
            ax = plt.gca()
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
            ax.xaxis.set_major_locator(plt.MultipleLocator(2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(2))
            
            # Title and labels
            plt.title(f"Unlabeled Plot - {os.path.splitext(file.filename)[0]}", fontsize=80, pad=20)
            plt.xlabel('X', fontsize=100, labelpad=20, weight='bold')
            plt.ylabel('Y', fontsize=100, labelpad=20, weight='bold')
            plt.tick_params(axis='both', which='major', labelsize=60, length=20, width=3, pad=15)
            
            # Save and close the figure
            plt.savefig(output_plot_path, bbox_inches='tight', dpi=72, pad_inches=0.0)
            plt.close()

            # Store paths in session for download
            session['unlabeled_temp_file'] = temp_file_path
            session['unlabeled_plot_path'] = output_plot_path
            session['unlabeled_original_filename'] = file.filename
            session['file_choice'] = 'unlabeled'
            # Generate URL for plot preview
            plot_url = url_for('static', filename=f'plots/{os.path.basename(output_plot_path)}')
            
            return render_template("display_unlabeled_file.html", 
                                 plot_url=plot_url, 
                                 filename=file.filename)

        except Exception as e:
            logger.error(f"Error processing unlabeled file: {str(e)}")
            return f"Error processing file: {str(e)}"

    

    
    if file_choice.lower() == 'labeled_file':
        try:
            # Create a temporary directory in a more persistent location
            temp_dir = os.path.join('ui', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save the file with its original name in our temp directory
            temp_file_path = os.path.join(temp_dir, file.filename)
            file.save(temp_file_path)
            
            # Store AWS paths in session based on the current ITEM
            session['aws_plot_dir_base'] = UI_AWS_PLOT_DIR_BASE
            session['aws_plot_dir_graded'] = UI_AWS_PLOT_DIR_GRADED
            session['aws_mtm_labeled_dir_path'] = UI_AWS_MTM_LABELED_DIR_PATH
            session['aws_mtm_graded_labeled_dir_path'] = UI_AWS_MTM_GRADED_LABELED_DIR_PATH
            
            mtm_processor = MTMProcessor()
            mtm_processor.load_coordinates_tables(temp_file_path)
            mtm_processor.remove_nan_mtm_points()
            #mtm_processor.display_filtered_coordinates_tables()

            table_name = os.path.splitext(file.filename)[0]
            piece_name = table_name.split('_')[0]
            
            # Make sure any existing plots are closed
            plt.close('all')
            
            plot_filename, plot_file_path = mtm_processor.plot_points(rename=piece_name)
            
            # Store file paths in session for later use
            session['labeled_temp_file'] = temp_file_path
            session['labeled_plot_path'] = plot_file_path
            session['piece_choice'] = piece_choice
            session['labeled_original_filename'] = file.filename
            session['plot_filename'] = plot_filename
            
            # Generate a temporary local URL for the plot preview
            plot_preview_url = url_for('static', 
                                     filename=f'plots/{os.path.basename(plot_file_path)}')
            
            # Render template with approval options
            return render_template("approve_mtm_plot.html", 
                                 plot_url=plot_preview_url,
                                 filename=file.filename)
                                 
        except Exception as e:
            logger.error(f"Error processing labeled file: {str(e)}")
            flash(f"Error processing file: {str(e)}", 'error')
            return redirect(url_for('file_operations_home'))

@app.route("/approve_mtm_upload", methods=['POST'])
def approve_mtm_upload():
    try:
        action = request.form.get('action')
        
        if not action or action not in ['approve', 'discard', 'download']:
            logger.error(f"Invalid action received: {action}")
            flash("Invalid action received", "error")
            return redirect(url_for('file_operations_home'))

        # Get stored file paths from session - Updated key names
        temp_mtm_file = session.get('labeled_temp_file')  # Changed from 'temp_mtm_file'
        plot_file_path = session.get('labeled_plot_path')  # Changed from 'plot_file_path'
        piece_choice = session.get('piece_choice')
        original_filename = session.get('labeled_original_filename')  # Changed from 'original_filename'
        plot_filename = session.get('plot_filename')

        # Handle download action by redirecting to download route
        if action == 'download':
            return redirect(url_for('download_labeled_plot'))

        # Validate all required session data is present
        if not all([temp_mtm_file, plot_file_path, piece_choice, original_filename, plot_filename]):
            logger.error("Missing required session data")
            flash("Session data missing. Please try uploading again.", "error")
            return redirect(url_for('file_operations_home'))

        if action == 'approve':
            logger.info(f"Processing approved upload for {original_filename}")
            
            # Get item-specific paths from session
            aws_plot_dir_base = session.get('aws_plot_dir_base')
            aws_plot_dir_graded = session.get('aws_plot_dir_graded')
            aws_mtm_labeled_dir_path = session.get('aws_mtm_labeled_dir_path')
            aws_mtm_graded_labeled_dir_path = session.get('aws_mtm_graded_labeled_dir_path')
            
            # Validate paths exist
            if not all([aws_plot_dir_base, aws_plot_dir_graded, 
                       aws_mtm_labeled_dir_path, aws_mtm_graded_labeled_dir_path]):
                logger.error("Missing AWS paths in session")
                flash("Session data missing. Please try uploading again.", "error")
                return redirect(url_for('file_operations_home'))
            
            # Set appropriate S3 paths based on piece type
            if piece_choice.lower() == 'graded_piece':
                aws_mtm_plot_dir_path = os.path.join(aws_plot_dir_graded, "labeled_mtm_graded", f"{plot_filename}_base.png")
                aws_mtm_dir_path = os.path.join(aws_mtm_graded_labeled_dir_path, original_filename)
                aws_mtm_hash_path = aws_mtm_graded_labeled_dir_path
            else:  # single_piece
                aws_mtm_plot_dir_path = os.path.join(aws_plot_dir_base, "labeled_mtm", f"{plot_filename}_base.png")
                aws_mtm_dir_path = os.path.join(aws_mtm_labeled_dir_path, original_filename)
                aws_mtm_hash_path = aws_mtm_labeled_dir_path
                
            try:
                # Upload plot file
                aws_utils.upload_file_by_path_to_s3(plot_file_path, aws_mtm_plot_dir_path)
                logger.info(f"Successfully uploaded plot to {aws_mtm_plot_dir_path}")
                
                # Upload MTM points file
                with open(temp_mtm_file, 'rb') as tmp:
                    file_content = tmp.read()
                    file_hash = aws_utils.compute_file_hashValue(file_content)
                    tmp.seek(0)
                        
                    if not aws_utils.check_hash_exists(file_hash, aws_mtm_hash_path):
                        aws_utils.upload_file_to_s3(tmp, aws_mtm_dir_path)
                        aws_utils.update_hash_file(aws_mtm_hash_path)
                        flash('Files uploaded successfully!', 'success')
                        logger.info(f"Successfully uploaded MTM file to {aws_mtm_dir_path}")
                    else:
                        flash('Duplicate file detected. Upload skipped.', 'warning')
                        logger.warning(f"Duplicate file detected for {original_filename}")
                        
            except Exception as e:
                logger.error(f"Error uploading files to S3: {str(e)}")
                flash(f"Error uploading files: {str(e)}", 'error')
                return redirect(url_for('file_operations_home'))
        else:  # action == 'discard'
            logger.info(f"Discarding upload for {original_filename}")
            flash('Upload discarded', 'info')
      
        # Clear session data - Add the new keys to clear
        for key in ['labeled_temp_file', 'labeled_plot_path', 'piece_choice', 
                   'labeled_original_filename', 'plot_filename',
                   'aws_plot_dir_base', 'aws_plot_dir_graded',
                   'aws_mtm_labeled_dir_path', 'aws_mtm_graded_labeled_dir_path']:
            session.pop(key, None)
 
        return redirect(url_for('file_operations_home'))
        
    except Exception as e:
        logger.error(f"Error in approve_mtm_upload: {str(e)}")
        flash(f"Error processing files: {str(e)}", 'error')
        return redirect(url_for('file_operations_home'))


@app.route("/download_unlabeled_plot")
def download_unlabeled_plot():
    try:
        plot_path = session.get('unlabeled_plot_path')
        original_filename = session.get('unlabeled_original_filename')
        
        logger.info(f"Attempting to download unlabeled plot from path: {plot_path}")
        
        if not plot_path or not original_filename:
            logger.error("Plot path or original filename not found in session")
            flash("Plot information not found in session", "error")
            return redirect(url_for('file_operations_home'))
        
        plot_path = os.path.abspath(plot_path)
        static_plots_dir = os.path.abspath('ui/static/plots')
        
        if not plot_path.startswith(static_plots_dir):
            logger.error(f"Invalid plot path location: {plot_path}")
            flash("Invalid plot file location", "error")
            return redirect(url_for('file_operations_home'))
            
        if not os.path.isfile(plot_path):
            logger.error(f"Plot file not found at path: {plot_path}")
            flash("Plot file not found on server", "error")
            return redirect(url_for('file_operations_home'))
            
        try:
            base_name = os.path.splitext(original_filename)[0]
            download_name = f"{base_name}_unlabeled_plot.png"
            
            logger.info(f"Sending file: {download_name} from {plot_path}")
            
            return send_file(
                plot_path,
                as_attachment=True,
                download_name=download_name,
                mimetype='image/png'
            )
            
        except Exception as e:
            logger.error(f"Error sending plot file: {str(e)}", exc_info=True)
            flash("Error downloading plot", "error")
            return redirect(url_for('file_operations_home'))

    except Exception as e:
        logger.error(f"Error in download_unlabeled_plot: {str(e)}", exc_info=True)
        flash(f"Error downloading file: {str(e)}", "error")
        return redirect(url_for('file_operations_home'))

@app.route("/download_labeled_plot")
def download_labeled_plot():
    try:
        plot_path = session.get('labeled_plot_path')
        original_filename = session.get('labeled_original_filename')
        
        logger.info(f"Attempting to download labeled plot from path: {plot_path}")
        
        if not plot_path or not original_filename:
            logger.error("Plot path or original filename not found in session")
            flash("Plot information not found in session", "error")
            return redirect(url_for('file_operations_home'))
        
        plot_path = os.path.abspath(plot_path)
        static_plots_dir = os.path.abspath('ui/static/plots')
        
        if not plot_path.startswith(static_plots_dir):
            logger.error(f"Invalid plot path location: {plot_path}")
            flash("Invalid plot file location", "error")
            return redirect(url_for('file_operations_home'))
            
        if not os.path.isfile(plot_path):
            logger.error(f"Plot file not found at path: {plot_path}")
            flash("Plot file not found on server", "error")
            return redirect(url_for('file_operations_home'))
            
        try:
            base_name = os.path.splitext(original_filename)[0]
            download_name = f"{base_name}_labeled_plot.png"
            
            logger.info(f"Sending file: {download_name} from {plot_path}")
            
            return send_file(
                plot_path,
                as_attachment=True,
                download_name=download_name,
                mimetype='image/png'
            )
            
        except Exception as e:
            logger.error(f"Error sending plot file: {str(e)}", exc_info=True)
            flash("Error downloading plot", "error")
            return redirect(url_for('file_operations_home'))

    except Exception as e:
        logger.error(f"Error in download_labeled_plot: {str(e)}", exc_info=True)
        flash(f"Error downloading file: {str(e)}", "error")
        return redirect(url_for('file_operations_home'))


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


# Add this function to clean up temp files
def cleanup_temp_files():
    temp_dir = os.path.join('ui', 'temp')
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            logger.info("Successfully cleaned up temp directory")
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {str(e)}")


@app.route("/download_graded_files/<file_type>")
def download_graded_files(file_type):
    try:
        if file_type == 'plots':
            zip_path = session.get('plots_zip_path')
            filename = 'plots.zip'
        elif file_type == 'excel':
            zip_path = session.get('excel_zip_path')
            filename = 'combined_entities.zip'
        else:
            flash("Invalid file type requested", "error")
            return redirect(url_for('file_operations_home'))
            
        if not zip_path or not os.path.exists(zip_path):
            flash("Requested file not found", "error")
            return redirect(url_for('file_operations_home'))
            
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Error downloading zip file: {str(e)}")
        flash("Error downloading file", "error")
        return redirect(url_for('file_operations_home'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)