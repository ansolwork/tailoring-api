from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
import pandas as pd
import yaml
from dotenv import load_dotenv
import os
import urllib.parse
import logging
import math

# Configure logging
logging.basicConfig(
    #level=logging.INFO,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#TODO: remove print statements
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
        logger.error(f"Error loading YAML config: {e}")
DB_TYPE = yaml_config['DB_TYPE']
DB_API = yaml_config['DB_API']
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

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
    logger.info("Loading required fields...")
    with open("api-input-required-fields.yml") as f:
        data = yaml.safe_load(f)
        logger.debug(f"Loaded required fields data: {data}")
        return data

def round_to_nearest_half(value):
    value = float(value)
      # If it's a whole number, round up to the next 0.5 increment
    if value % 1 == 0:
        return value + 0.5
    # If the decimal part is 0.25 or 0.75, round down to the nearest 0.5
    elif value % 1 == 0.25:
        return math.ceil(value * 2) / 2
    elif value % 1 == 0.75:
        return math.floor(value * 2) / 2
    # If the decimal part is 0.5, keep it as it is
    else:
        return round(value * 2) / 2

def apply_alterations(pieces, alteration_names, alteration_amount):
    return "success"


def add_allowance(fit_type, measurement_body_type, input_measurement_value):
    engine = create_engine(connection_string)
    increment_amount_query = text("""
            select allowance_value from body_allowance where measurement_body_type ilike :measurement_body_type and fit_type ilike :fit_type;
        """)
    with engine.connect() as conn:
        increment_amount_df = pd.read_sql(increment_amount_query, conn,
                                          params={"measurement_body_type": measurement_body_type, "fit_type": fit_type})
        if increment_amount_df.empty:
            increment_amount = 0
        else:
            increment_amount = increment_amount_df['allowance_value'].values[0]
        body_measurement_with_allowance = float(input_measurement_value) + float(increment_amount)
        return body_measurement_with_allowance


def get_garment_measurements(measurement_body_type='chest', body_measurement=None):
    engine = create_engine(connection_string)
    garment_size_query = text("""
            select measurement_size from all_size_measurements where measurement_body_type ilike :measurement_body_type and measurement_value = :body_measurement;
        """)
    with engine.connect() as conn:
        garment_size_df = pd.read_sql(garment_size_query, conn, params={"measurement_body_type": measurement_body_type,
                                                                        "body_measurement": str(round_to_nearest_half(body_measurement))})
        if garment_size_df.empty:
            raise ValueError(f"No garment size found for body type: {measurement_body_type} and body measurement: {body_measurement} in the database")
    garment_size = garment_size_df['measurement_size'].values[0]
    logger.debug(f"Garment size: {garment_size}")
    garment_measurement_query = text("""
                select measurement_body_type, measurement_value from all_size_measurements where measurement_size = :garment_size;
        """)
    with engine.connect() as conn:
        garment_measurement_df = pd.read_sql(garment_measurement_query, conn, params={"garment_size": str(garment_size)})
    logger.debug(f"Garment measurement dataframe: {garment_measurement_df}")
    return garment_measurement_df

def get_alteration_name_from_db(body_type):
    engine = create_engine(connection_string)
    get_alterations_query = text("""
            SELECT DISTINCT an.alteration_name as alteration_name
            FROM alterations an
            JOIN tag_alteration_rel tar ON an.alteration_id = tar.alteration_id
            JOIN tags t1 ON tar.tag_id = t1.tag_id
            WHERE t1.tag_name ILIKE :body_type
        """)
    with engine.connect() as conn:
        alterations_df = pd.read_sql(get_alterations_query, conn, params={"body_type": body_type})
    return alterations_df['alteration_name']

def compute_alteration_amount(fit_type, measurement_type, input_measurement_df):
    chest_measurement = input_measurement_df[input_measurement_df['measurement_body_type'] == 'chest']['measurement_value'].values[0]

    # if the measurement type is body, add allowance to the chest measurement
    if measurement_type.lower() == "body":
        body_measurement_with_allowance = add_allowance(fit_type, measurement_body_type='chest',
                                                        input_measurement_value=chest_measurement)
        garment_measurements_df = get_garment_measurements(body_measurement=body_measurement_with_allowance)
    else:
        garment_measurements_df = get_garment_measurements(body_measurement=chest_measurement)

    # Calculate alteration amounts for each measurement_body_type
    alteration_amounts = {
        'measurement_body_type': [],
        'alteration_amount': [],
        'alteration_name': []
    }
    for _, row in input_measurement_df.iterrows():
        body_type = row['measurement_body_type']
        # add allowance to the input value if the measurement_type is body
        if measurement_type.lower() == "body":
            input_value = add_allowance(fit_type, body_type, float(row['measurement_value']))
        else:
            input_value = float(row['measurement_value'])

        # Modified this section to handle case-insensitive comparison
        matching_measurements = garment_measurements_df[
            garment_measurements_df['measurement_body_type'].str.lower() == body_type.lower()
        ]
        if matching_measurements.empty:
            raise ValueError(f"No garment measurement found for body type: {body_type}")

        garment_value = float(matching_measurements['measurement_value'].values[0])
        alteration_amounts['measurement_body_type'].append(body_type)
        alteration_amounts['alteration_amount'].append(input_value - garment_value)
        alteration_name = get_alteration_name_from_db(body_type).values[0]
        alteration_amounts['alteration_name'].append(alteration_name)

     
    return pd.DataFrame(alteration_amounts).to_dict('records')  # Convert final result to dict


# convenience function to create the input measurement df from the yaml data
# here we handle all the customisations like half measurements, left and right measurements
#TODO: Check for chest and half chest measurements and find out which one to use when both are present
def create_input_measurement_df(yaml_data):
    required_fields = load_required_fields()
    measurements = {
        'measurement_body_type': [],
        'measurement_value': []
    }
    '''
    multiplier = 0 means skip this measurement when not in body mode
    multiplier = 1 means process this measurement always
    multiplier = 2 means process this measurement only when half measurements are required
    '''
    measurement_mapping = {
        'half chest': ('chest', 2),
        'half waist measure': ('waist measure', 2),
        'half seat': ('seat', 2),
        'chest': ('chest', 0),  # Skip when not in body mode
        'waist measure': ('waist measure', 0),  # Skip when not in body mode
        'seat': ('seat', 0),    # Skip when not in body mode
        'cuff length right': ('cuff length left or both', 1),
        'cuff length left or both': ('cuff length left or both', 1),
        'right sleeve length': ('sleeve length', 1),
        'sleeve length': ('sleeve length', 1)
    }

    # Track measurements to handle right vs regular preferences
    collected_measurements = {}

    # Only process fields that are both required and present in yaml_data
    for field in required_fields['measurement_fields']:
        if field in yaml_data:
            value = str(yaml_data[field])
            measurement_type = yaml_data.get('Measurement Type', '').lower()
            
            if field.lower() in measurement_mapping:
                body_type, multiplier = measurement_mapping[field.lower()]
                # If multiplier is 0, only process in body measurement mode
                if multiplier == 0 and measurement_type != 'body':
                    continue
                
                processed_value = float(value) * (multiplier or 1)  # Use 1 if multiplier is 0
                
                # Store in collected_measurements for comparison
                if body_type not in collected_measurements:
                    collected_measurements[body_type] = processed_value
                elif 'right' in field.lower():  # Prefer right measurements if larger
                    collected_measurements[body_type] = max(processed_value, collected_measurements[body_type])
            else:
                measurements['measurement_body_type'].append(field.lower())
                measurements['measurement_value'].append(value)

    # Add the final measurements after resolving right vs regular preferences
    for body_type, value in collected_measurements.items():
        measurements['measurement_body_type'].append(body_type)
        measurements['measurement_value'].append(str(value))

    logger.info(f"Created input measurements dataframe with {len(measurements['measurement_body_type'])} entries")
    logger.debug(f"Input measurements dataframe: {measurements}")
    return pd.DataFrame(measurements)


def get_entities_from_db(yaml_data):
    engine = create_engine(connection_string)
    values = list(yaml_data.values())
    get_pieces_query = text("""
            SELECT DISTINCT t1.tag_name AS tag_name, t1.tag_subcategory AS tag_subcategory, pt.piece_name
            FROM pieces pt
            JOIN tag_piece_rel tpr ON pt.piece_id = tpr.piece_id
            JOIN tags t1 ON tpr.tag_id = t1.tag_id
            WHERE t1.tag_name ILIKE ANY(:tag_names)
        """)
    # Updated column aliases to match the expected response format
    get_alterations_query = text("""
            SELECT DISTINCT t1.tag_name as tag_name, t1.tag_subcategory as tag_subcategory, an.alteration_name as alteration_name, COALESCE(cast(taar.alteration_amnt as float), 0) as alteration_amnt
            FROM alterations an
            JOIN tag_alteration_rel tar ON an.alteration_id = tar.alteration_id
            JOIN tags t1 ON tar.tag_id = t1.tag_id
            LEFT JOIN tag_alteration_amnt_rel taar ON t1.tag_id = taar.tag_id
            WHERE t1.tag_name ILIKE ANY(:tag_names)
        """)
    
    # get the tag_names that are not present in tag_piece_rel and tag_alteration_rel tables
    get_untagged_tag_names_query = text("""
            SELECT DISTINCT t1.tag_name as tag_name , t1.tag_subcategory as tag_subcategory
            FROM tags t1
            LEFT JOIN tag_piece_rel tpr ON t1.tag_id = tpr.tag_id   
            LEFT JOIN tag_alteration_rel tar ON t1.tag_id = tar.tag_id
            WHERE t1.tag_name ILIKE ANY(:tag_names) AND tpr.tag_id IS NULL AND tar.tag_id IS NULL
        """)
    with engine.connect() as conn:
        pieces_df = pd.read_sql(get_pieces_query, conn, params={"tag_names": values})
        alterations_df = pd.read_sql(get_alterations_query, conn, params={"tag_names": values})
        untagged_items_df = pd.read_sql(get_untagged_tag_names_query, conn, params={"tag_names": values})

    # Convert results to the requested format
    result = {
        'style_pieces': pieces_df.to_dict('records'),
        'style_alterations': alterations_df.to_dict('records'),
        'untagged_items': untagged_items_df.to_dict('records')
    }

    return result


@app.route('/api/home')
def home():
    return "Home Page"



'''The API call to get entities from db
'''


@app.route('/api/get_entities', methods=['GET'])
def get_entities():
    try:
        try:
            yaml_data = request.data.decode('utf-8')
            data = yaml.load(yaml_data, Loader=StringLoader)

            # Filter data to only include available required fields
            required_fields = load_required_fields()
            filtered_data = {k: data[k] for k in required_fields['required'] if k in data}

            # Check if we have at least one required field
            if not filtered_data:
                return jsonify({
                    'error': 'No required fields found in the request'
                }), 400

            # Get the result dictionary from get_entities_from_db
            result = get_entities_from_db(filtered_data)
            
            return jsonify(result), 200

        except yaml.YAMLError as e:
            return jsonify({
                'error': f'Invalid YAML format: {str(e)}'
            }), 400

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400


'''The API call to process entities'''


@app.route('/api/process_entities', methods=['POST'])
def process_entities():
    try:
        if request.is_json:
            data = request.get_json()
        elif request.headers.get('Content-Type') == 'application/x-yaml':
            data = yaml.load(request.data, Loader=StringLoader)
            
            # Filter data to only include available required fields
            required_fields = load_required_fields()
            filtered_data = {k: data[k] for k in required_fields['required'] if k in data}

            # Check if we have at least one required field
            if not filtered_data:
                return jsonify({
                    'error': 'No required fields found in the request'
                }), 400

        else:
            return jsonify({'error': 'Unsupported Content-Type'}), 415

        style_entities_df = get_entities_from_db(filtered_data)
        
    # compute the alteration amount from the input measurement provided in the request
        input_measurement_df = create_input_measurement_df(filtered_data)
    
        measured_alteration_amounts = compute_alteration_amount(filtered_data['Fit'], filtered_data['Measurement Type'],
                                                      input_measurement_df)
        logger.info(f"Computed alteration amounts: {measured_alteration_amounts}")
        # Create a copy of style_alterations without the tag_name and tag_subcategorycolumn as it is not required in the response for processing
        style_entities_notag_df = {
            'style_pieces': style_entities_df['style_pieces'],
            'style_alterations': [{k: v for k, v in alt.items() if k != 'tag_name' and k != 'tag_subcategory'} for alt in style_entities_df['style_alterations']]
        }
       # same for the measured alterations to be used for processing
        measured_alterations_notag_df = {
            'measured_alterations_amounts': [{k: v for k, v in alt.items() if k != 'measurement_body_type'} for alt in measured_alteration_amounts]
        }
        response = {
            'style_pieces': [piece['piece_name'] for piece in style_entities_df['style_pieces']],  
            'style_alterations': style_entities_notag_df['style_alterations'],
            'measured_alterations_amounts': measured_alterations_notag_df['measured_alterations_amounts']
        }
        
        logger.info(f"Response: {response}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in process_entities: {str(e)}")
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
