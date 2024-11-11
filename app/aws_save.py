import boto3
import pandas as pd
import yaml
import logging
import os
from botocore.exceptions import ClientError
import argparse

class AWSS3Saver:
    def __init__(self, bucket_name: str, local_dir: str, aws_dir: str, config_path: str = 'config_graded_mtm.yml'):
        self.config = self._load_config(config_path)
        self.s3 = self._initialize_s3_client()
        self.bucket_name = bucket_name
        self.local_dir = local_dir
        self.aws_dir = aws_dir

    def _load_config(self, config_path: str):
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def _initialize_s3_client(self):
        session = boto3.Session(profile_name='tailoring_api')
        return session.client('s3')

    def save_file(self, local_file_path: str, s3_key: str):
        try:
            self.s3.upload_file(local_file_path, self.bucket_name, s3_key)
            logging.info(f"Successfully uploaded {local_file_path} to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            logging.error(f"Error uploading file {local_file_path}: {e}")

    def save_all_files(self):
        try:
            logging.info(f"Processing directory mapping: {self.local_dir} -> {self.aws_dir}")
            
            for root, dirs, files in os.walk(self.local_dir):
                for file in files:
                    if file.endswith('.xlsx') or file.endswith('.csv'):  # Add more extensions if needed
                        local_file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_file_path, self.local_dir)
                        s3_key = os.path.join(self.aws_dir, relative_path)
                        self.save_file(local_file_path, s3_key)
        except Exception as e:
            logging.error(f"Error saving files to S3: {e}")

def save_local_data_to_s3(config_file, local_dir_key: str, aws_dir_key: str):
    try:
        # Load configuration
        logging.info(f"Loading configuration from {config_file}")
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        # Get directory paths from config using the provided keys
        local_dir = config[local_dir_key]
        aws_dir = config[aws_dir_key]
        
        logging.info(f"Initializing AWSS3Saver with bucket: {config['AWS_S3_BUCKET_NAME']}")
        logging.info(f"Local directory: {local_dir}")
        logging.info(f"AWS directory: {aws_dir}")
        
        # Initialize AWSS3Saver with specific directory mapping
        saver = AWSS3Saver(
            bucket_name=config['AWS_S3_BUCKET_NAME'],
            local_dir=local_dir,
            aws_dir=aws_dir,
            config_path=config_file
        )

        logging.info("Attempting to save files to S3")
        saver.save_all_files()
        
        logging.info("Processing completed successfully")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Save local data to S3')
    parser.add_argument('config_file', help='Path to the YAML configuration file')
    parser.add_argument('local_dir_key', help='Key in config file for local directory (e.g., LOCAL_STAGING_DIR)')
    parser.add_argument('aws_dir_key', help='Key in config file for AWS directory (e.g., AWS_STAGING_DIR)')
    
    args = parser.parse_args()

    logging.info(f"Starting script with config file: {args.config_file}")
    save_local_data_to_s3(args.config_file, args.local_dir_key, args.aws_dir_key)
    logging.info("Script execution completed")
