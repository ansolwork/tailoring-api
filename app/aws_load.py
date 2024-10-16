import boto3
import pandas as pd
import yaml
import logging
import os
from botocore.exceptions import ClientError
from io import BytesIO

class AWSS3Loader:
    def __init__(self, config_path: str = 'tailoring_api_config.yml'):
        self.config = self._load_config(config_path)
        self.s3 = self._initialize_s3_client()
        self.bucket_name = self.config['AWS_S3_BUCKET_NAME']
        self.directory_key = self.config['AWS_MTM_DIR_PATH_LABELED']
        self.local_save_dir = 'data/input/mtm_combined_entities_labeled'

    def _load_config(self, config_path: str):
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def _initialize_s3_client(self):
        session = boto3.Session(profile_name='tailoring_api')
        return session.client('s3', region_name='us-east-1')

    def load_file(self, file_key: str):
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
            excel_data = BytesIO(response['Body'].read())
            return pd.read_excel(excel_data)
        except Exception as e:
            logging.error(f"Error loading file {file_key}: {e}")
            return None

    def load_all_files(self):
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.directory_key)
            files = {}
            for obj in response.get('Contents', []):
                file_key = obj['Key']
                if file_key.endswith('.xlsx'):
                    df = self.load_file(file_key)
                    if df is not None:
                        files[file_key] = df
            return files
        except ClientError as e:
            logging.error(f"Error listing objects in directory {self.directory_key}: {e}")
            return {}

    def save_dataframe(self, df, file_name):
        # Create the directory if it doesn't exist
        os.makedirs(self.local_save_dir, exist_ok=True)
        
        # Save the DataFrame as an Excel file
        save_path = os.path.join(self.local_save_dir, file_name)
        df.to_excel(save_path, index=False)
        logging.info(f"Saved {file_name} to {save_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    loader = AWSS3Loader()

    files = loader.load_all_files()
    for file_key, df in files.items():
        logging.info(f"Loaded file: {file_key}")
        logging.info(f"Shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
        
        # Extract the file name from the S3 key
        file_name = os.path.basename(file_key)
        
        # Save the DataFrame to the local directory
        loader.save_dataframe(df, file_name)
        
        logging.info("-" * 30)
