import boto3
import pandas as pd
import yaml
import logging
import os
from botocore.exceptions import ClientError
from io import BytesIO
import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
from tqdm import tqdm

class AWSS3Loader:
    def __init__(self, bucket_name: str, directory_key: str, local_save_dir: str, config_path: str = 'tailoring_api_config.yml'):
        self.config = self._load_config(config_path)
        self.s3 = self._initialize_s3_client()
        self.bucket_name = bucket_name
        self.directory_key = directory_key
        self.local_save_dir = local_save_dir

    def _load_config(self, config_path: str):
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def _initialize_s3_client(self):
        session = boto3.Session(profile_name='tailoring_api')
        return session.client('s3')

    def load_file(self, file_key: str):
        try:
            # Add file size check
            response = self.s3.head_object(Bucket=self.bucket_name, Key=file_key)
            file_size = response['ContentLength']
            if file_size > 50_000_000:  # 50MB
                logging.warning(f"Large file detected {file_key}: {file_size/1_000_000:.2f}MB")
            
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
            excel_data = BytesIO(response['Body'].read())
            # Add optimization for large Excel files
            return pd.read_excel(excel_data, engine='openpyxl')
        except Exception as e:
            logging.error(f"Error loading file {file_key}: {e}")
            return None

    def load_all_files(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            files = {}
            
            # Get all Excel files first
            excel_files = []
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.directory_key):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.xlsx'):
                        excel_files.append(obj['Key'])
            
            logging.info(f"Found {len(excel_files)} Excel files to process")
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(tqdm(
                    executor.map(self.load_file, excel_files),
                    total=len(excel_files),
                    desc="Loading files"
                ))
                
                for file_key, df in zip(excel_files, results):
                    if df is not None:
                        files[file_key] = df
                        
            return files
        except ClientError as e:
            logging.error(f"Error listing objects in directory {self.directory_key}: {e}")
            return {}

    def save_dataframe(self, df, file_key):
        # Remove the base directory from the file_key
        relative_path = file_key.replace(self.directory_key, '', 1)
        
        # Create the full local path
        full_local_path = os.path.join(self.local_save_dir, relative_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(full_local_path), exist_ok=True)
        
        # Save the DataFrame as an Excel file
        df.to_excel(full_local_path, index=False)
        logging.info(f"Saved {relative_path} to {full_local_path}")

def load_and_save_s3_data(config_file):
    try:
        # Load configuration
        logging.info(f"Loading configuration from {config_file}")
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        logging.info(f"Initializing AWSS3Loader with bucket: {config['AWS_S3_BUCKET_NAME']}, "
                     f"directory: {config['DIRECTORY_KEY']}, "
                     f"local dir: {config['LOCAL_SAVE_DIR_DEFAULT'] if 'LOCAL_SAVE_DIR_DEFAULT' in config else config['LOCAL_SAVE_DIR_LABELED']}")
        
        # Initialize AWSS3Loader with parameters from config
        loader = AWSS3Loader(
            bucket_name=config['AWS_S3_BUCKET_NAME'],
            directory_key=config['DIRECTORY_KEY'],
            local_save_dir=config['LOCAL_SAVE_DIR_DEFAULT'] if 'LOCAL_SAVE_DIR_DEFAULT' in config else config['LOCAL_SAVE_DIR_LABELED']
        )

        logging.info("Attempting to load files from S3")
        files = loader.load_all_files()
        
        if not files:
            logging.warning("No files were found or loaded.")
            return

        for file_key, df in files.items():
            logging.info(f"Loaded file: {file_key}")
            logging.info(f"Shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")
            
            # Save the DataFrame to the local directory, maintaining S3 structure
            loader.save_dataframe(df, file_key)
            
            logging.info("-" * 30)
        
        logging.info("Processing completed successfully")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Load and save S3 data')
    parser.add_argument('config_file', help='Path to the YAML configuration file')
    
    args = parser.parse_args()

    logging.info(f"Starting script with config file: {args.config_file}")
    load_and_save_s3_data(args.config_file)
    logging.info("Script execution completed")
