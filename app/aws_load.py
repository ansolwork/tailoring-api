import boto3
import pandas as pd
import yaml
from io import BytesIO
from botocore.exceptions import ClientError
from typing import Dict, Optional, List
import logging
import chardet
import csv
import os

class AWSS3Loader:
    def __init__(self, config_path: str = 'tailoring_api_config.yml'):
        self.config = self._load_config(config_path)
        self.s3 = self._initialize_s3_client()
        self.bucket_name = self.config['AWS_S3_BUCKET_NAME']
        self.allowed_extensions = self.config['ALLOWED_EXTENSIONS']

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except FileNotFoundError:
            logging.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            raise

    def _initialize_s3_client(self):
        session = boto3.Session(profile_name='tailoring_api')
        return session.client('s3',
                              region_name=self.config.get('AWS_REGION', 'us-east-1'),
                              config=boto3.session.Config(signature_version=self.config['AWS_S3_SIGNATURE_VERSION']))

    def load_file(self, file_key: str) -> Optional[pd.DataFrame]:
        """
        Load a single file (CSV or Excel) from S3
        """
        file_extension = file_key.split('.')[-1].lower()
        if file_extension not in self.allowed_extensions:
            logging.warning(f"File {file_key} has an unsupported extension.")
            return None

        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
            csv_data = csv.reader(response['Body'].read().decode('utf-8').splitlines())
            headers = next(csv_data)
            data = [row for row in csv_data if len(row) == len(headers)]
            return pd.DataFrame(data, columns=headers)
        except Exception as e:
            print(f"Error loading file {file_key}: {e}")
            return None

    def load_directory(self, directory_key: str) -> Dict[str, pd.DataFrame]:
        """
        Load all supported files from a directory in S3
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=directory_key)
            supported_files = [obj['Key'] for obj in response.get('Contents', []) 
                               if obj['Key'].split('.')[-1].lower() in self.allowed_extensions]
            
            return {file_key: df for file_key in supported_files if (df := self.load_file(file_key)) is not None}
        except ClientError as e:
            logging.error(f"Error listing objects in directory {directory_key}: {e}")
            return {}

    def get_file_list(self, directory_key: str) -> List[str]:
        """
        Get a list of all files in a directory
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=directory_key)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logging.error(f"Error listing objects in directory {directory_key}: {e}")
            return []

    def save_dataframe(self, df, file_key):
        # Extract filename from the file_key
        filename = os.path.basename(file_key)
        
        # Create the directory if it doesn't exist
        save_dir = 'data/input/mtm_combined_entities'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the DataFrame as a CSV file
        save_path = os.path.join(save_dir, filename)
        df.to_csv(save_path, index=False)
        print(f"Saved {filename} to {save_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    loader = AWSS3Loader()

    # Use the directory path from the config file
    directory_key = loader.config['AWS_MTM_DIR_PATH']

    # Example: Load all supported files from the specified directory
    directory_files = loader.load_directory(directory_key)
    if directory_files:
        logging.info(f"Successfully loaded files from {directory_key}")
        for file_key, df in directory_files.items():
            logging.info(f"File: {file_key}")
            logging.info(df.head())
            logging.info("-"*30)
            loader.save_dataframe(df, file_key)
    else:
        logging.warning(f"No supported files found in {directory_key}")

    # Example: Get file list from the specified directory
    file_list = loader.get_file_list(directory_key)
    logging.info(f"Files in directory {directory_key}:")
    for file in file_list:
        logging.info(file)

    # Example: Load the Excel file
    excel_file_key = "data/input/recurrent/mtm_combined_entites/LGFG-V2-SH-01-STBS-F_combined_entities.xlsx"
    excel_file = loader.load_file(excel_file_key)
    if excel_file is not None:
        logging.info(f"Successfully loaded Excel file: {excel_file_key}")
        logging.info(excel_file.head())
    else:
        logging.warning(f"Failed to load Excel file: {excel_file_key}")

    # Example: Load the CSV file
    csv_file_key = "data/input/recurrent/mtm_combined_entites/LGFG-SH-01-CCB-FO_combined_entities.csv"
    csv_file = loader.load_file(csv_file_key)
    if csv_file is not None:
        logging.info(f"Successfully loaded CSV file: {csv_file_key}")
        logging.info(csv_file.head())
    else:
        logging.warning(f"Failed to load CSV file: {csv_file_key}")
