import boto3
import pandas as pd
import yaml
import logging
import os
from botocore.exceptions import ClientError
import argparse

class AWSS3Saver:
    def __init__(self, bucket_name: str, local_save_dir: str, aws_save_dir: str, config_path: str = 'tailoring_api_config.yml'):
        self.config = self._load_config(config_path)
        self.s3 = self._initialize_s3_client()
        self.bucket_name = bucket_name
        self.local_save_dir = local_save_dir
        self.aws_save_dir = aws_save_dir

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
            for root, dirs, files in os.walk(self.local_save_dir):
                for file in files:
                    if file.endswith('.xlsx'):
                        local_file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_file_path, self.local_save_dir)
                        s3_key = os.path.join(self.aws_save_dir, relative_path)
                        self.save_file(local_file_path, s3_key)
        except Exception as e:
            logging.error(f"Error saving files to S3: {e}")

def save_local_data_to_s3(config_file):
    try:
        # Load configuration
        logging.info(f"Loading configuration from {config_file}")
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        logging.info(f"Initializing AWSS3Saver with bucket: {config['AWS_S3_BUCKET_NAME']}, "
                     f"local dir: {config['LOCAL_SAVE_DIR_LABELED']}, "
                     f"AWS save dir: {config['AWS_SAVE_DIR']}")
        
        # Initialize AWSS3Saver with parameters from config
        saver = AWSS3Saver(
            bucket_name=config['AWS_S3_BUCKET_NAME'],
            local_save_dir=config['LOCAL_SAVE_DIR_LABELED'],
            aws_save_dir=config['AWS_SAVE_DIR']
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
    
    args = parser.parse_args()

    logging.info(f"Starting script with config file: {args.config_file}")
    save_local_data_to_s3(args.config_file)
    logging.info("Script execution completed")
