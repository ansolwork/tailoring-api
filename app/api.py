from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
import pandas as pd
import yaml
from dotenv import load_dotenv
import os
import urllib.parse

'''
API to apply alterations for the items
USAGE : API call to endpoint 
The API expects a JSON/YAML input:
'''

load_dotenv()
# Run from project root (no relative path needed)
config_filepath = "tailoring_api_config.yml"
with open(config_filepath) as f:
    try:
        yaml_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
DB_TYPE = yaml_config['DB_TYPE']
DB_API = yaml_config['DB_API']
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = yaml_config['DB_HOST']
DB_PORT = yaml_config['DB_PORT']
DB_NAME = yaml_config['DB_NAME']

connection_string = f"{DB_TYPE}+{DB_API}://{DB_USER}:{urllib.parse.quote(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

app = Flask(__name__, template_folder="templates")


# Custom loader to convert all YAML values to strings
class StringLoader(yaml.Loader):
    def construct_scalar(self, node):
        # Convert all scalar values to strings
        value = super().construct_scalar(node)
        # If the value is a boolean-like string, convert it explicitly to string
        if value is True:
            return "true"
        elif value is False:
            return "false"
        return str(value)

    def construct_sequence(self, node):
        return [str(item) for item in super().construct_sequence(node)]

    def construct_mapping(self, node):
        return {str(key): str(value) for key, value in super().construct_mapping(node).items()}


def load_required_fields():
    with open("api-input-required-fields.yml") as f:
        return yaml.safe_load(f)


def apply_alterations(pieces, alteration_names, alteration_amount):
    return "success"


def get_entities_from_db(yaml_data):
    engine = create_engine(connection_string)
    # Extract values into an array
    values = list(yaml_data.values())
    get_pieces_query = text("""
            SELECT pt.piece_name
            FROM pieces pt
            JOIN tag_piece_rel tpr ON pt.piece_id = tpr.piece_id
            JOIN tags t1 ON tpr.tag_id = t1.tag_id
            WHERE t1.tag_name ILIKE ANY(:tag_names)
        """)
    get_alteration_names_query = text("""
            SELECT an.alteration_name
            FROM alterations an
            JOIN tag_alteration_rel tar ON an.alteration_id = tar.alteration_id
            JOIN tags t1 ON tar.tag_id = t1.tag_id
            WHERE t1.tag_name ILIKE ANY(:tag_names)
        """)
    get_alteration_amount_query = text("""
            SELECT taar.alteration_amnt
            FROM tag_alteration_amnt_rel taar
            JOIN tags t1 ON taar.tag_id = t1.tag_id
            WHERE t1.tag_name ILIKE ANY(:tag_names)
        """)
    with engine.connect() as conn:
        pieces_df = pd.read_sql(get_pieces_query, conn, params={"tag_names": values})
        alteration_names_df = pd.read_sql(get_alteration_names_query, conn, params={"tag_names": values})
        alteration_amount_df = pd.read_sql(get_alteration_amount_query, conn, params={"tag_names": values})

    print(pieces_df)
    print(alteration_names_df)
    print(alteration_amount_df)
    return pieces_df, alteration_names_df, alteration_amount_df


@app.route('/api/home')
def home():
    return "Home Page"


# TODO: change the below code as per new changes

'''The API call to get entities from db
'''


@app.route('/api/get_entities', methods=['GET'])
def get_entities():
    try:
        # Handle GET request with YAML data as query parameter
        '''yaml_data = request.args.get('data')
        if not yaml_data:
            return jsonify({
                'error': 'Missing data parameter'
            }), 400
'''
        try:
            yaml_data = request.data.decode('utf-8')
            data = yaml.load(yaml_data, Loader=StringLoader)

            # Validate required fields
            required_fields = load_required_fields()
            missing_fields = [field for field in required_fields['required'] if field not in data]
            if missing_fields:
                return jsonify({
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400

            # Filter data to only include required fields
            filtered_data = {k: data[k] for k in required_fields['required']}

            # Pass filtered data to get_entities_from_db
            pieces_df, alteration_names_df, alteration_amount_df = get_entities_from_db(filtered_data)
        except yaml.YAMLError as e:
            return jsonify({
                'error': f'Invalid YAML format: {str(e)}'
            }), 400

        # Prepare response
        response = {
            'pieces': pieces_df['piece_name'].tolist(),
            'alteration_names': alteration_names_df['alteration_name'].tolist(),
            'alteration_amounts': alteration_amount_df['alteration_amnt'].tolist()
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400


'''The API call to process entities'''


@app.route('/api/process_entities', methods=['POST'])
def process_entities():
    try:
        # Handle POST request with YAML or JSON
        if request.is_json:
            data = request.get_json()
        elif request.headers.get('Content-Type') == 'application/x-yaml':
            data = yaml.safe_load(request.data)
        else:
            return jsonify({'error': 'Unsupported Content-Type'}), 415

        pieces_df, alteration_names_df, alteration_amount_df = get_entities_from_db(data)

        # Prepare response
        response = {
            'pieces': pieces_df['piece_name'].tolist(),
            'alteration_names': alteration_names_df['alteration_name'].tolist(),
            'alteration_amounts': alteration_amount_df['alteration_amnt'].tolist()
        }

        apply_alterations(pieces_df, alteration_names_df, alteration_amount_df)
        return jsonify({'status': 'success', 'message': 'Alterations applied successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
