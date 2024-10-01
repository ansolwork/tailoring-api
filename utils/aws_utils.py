import hashlib

import boto3
from botocore.client import Config
import magic
import io

from botocore.exceptions import ClientError


# AWS Utilities
class AwsUtils:
    def __init__(self, allowed_extensions,
                 allowed_mime_types,
                 aws_s3_bucket_name,
                 aws_signature_version,
                 aws_profile_name):

        self.allowed_extensions = allowed_extensions
        self.allowed_mime_types = allowed_mime_types
        self.aws_s3_bucket_name = aws_s3_bucket_name
        self.aws_signature_version = aws_signature_version
        self.aws_profile_name = aws_profile_name  # Add a profile name argument

    # Validation to check file extensions, content types, and file sizes
    def allowed_file(self, filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def allowed_mime(self, file):
        mime = magic.from_buffer(file.stream.read(2048), mime=True)
        file.stream.seek(0)  # Reset file pointer after reading
        return mime in self.allowed_mime_types

    # Below functions are used for duplicate handling in s3
    def compute_file_hashValue(self, file_content):
        hasher = hashlib.md5()
        hasher.update(file_content)
        return hasher.hexdigest()
    def update_hash_file(self, s3_filepath, hash_file_name="md5_hash_file"):
        try:
            s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client(
                service_name='s3',
            )
            response = s3_client.list_objects_v2(Bucket=self.aws_s3_bucket_name, Prefix=s3_filepath)
            hashes = []

            # Iterate through each file in the specified S3 directory
            if 'Contents' in response:
                for obj in response['Contents']:
                    file_key = obj['Key']
                    if file_key.endswith('/'):
                        continue  # Skip directories
                    # Get the file content
                    file_content = s3_client.get_object(Bucket=self.aws_s3_bucket_name, Key=file_key)['Body'].read()
                    file_hash = self.compute_file_hashValue(file_content)
                    hashes.append(f"{file_key}: {file_hash}")

            # Save hashes to the hash file
            s3_client.put_object(Bucket=self.aws_s3_bucket_name, Key=s3_filepath + hash_file_name,
                                 Body='\n'.join(hashes))
            print("Hash file updated successfully.")
        except Exception as e:
            print(f"Error updating hash file: {e}")
    def check_hash_exists(self, file_hash,s3_filepath,hash_file_name="md5_hash_file"):
        try:
            s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client(
                service_name='s3',
            )
            hash_file = s3_client.get_object(Bucket=self.aws_s3_bucket_name, Key=s3_filepath + hash_file_name)[
                'Body'].read().decode(
                'utf-8')
            for line in hash_file.splitlines():
                if file_hash in line:
                    return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False  # Hash file doesn't exist
            else:
                print(f"Error checking hash file: {e}")
                return False

    # Convenience method to upload file to S3 bucket
    def upload_file_to_s3(self, file, s3_file_path):
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client(
            service_name='s3'
        )
        response = s3_client.upload_fileobj(file, self.aws_s3_bucket_name, s3_file_path)
        print(f'upload_log_to_aws response: {response}')

    def upload_file_by_path_to_s3(self, file_path, s3_file_path):
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client(
            service_name='s3'
        )
        # Use upload_file, which works with file paths directly
        response = s3_client.upload_file(file_path, self.aws_s3_bucket_name, s3_file_path)
        print(f'File {file_path} uploaded to {s3_file_path} in S3. Response: {response}')

    def upload_buffer_to_s3(self, buffer, s3_file_path):
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client('s3')

        # Upload the buffer to S3
        response = s3_client.upload_fileobj(buffer, self.aws_s3_bucket_name, s3_file_path)
        print(f'Upload to S3 completed. Response: {response}')

    def upload_dataframe_to_s3(self, dataframe, s3_file_path, file_format):
        # Create an in-memory buffer
        buffer = io.BytesIO()

        if file_format == "csv":
            # Save the DataFrame to the buffer as a CSV
            dataframe.to_csv(buffer, index=False)
        elif file_format == "excel":
            # Save the DataFrame to the buffer as an Excel file
            dataframe.to_excel(buffer, index=False)

        # Move the buffer's position to the start
        buffer.seek(0)

        # Create the S3 client with the desired profile
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client('s3')

        # Upload the in-memory file to S3
        response = s3_client.upload_fileobj(buffer, self.aws_s3_bucket_name, s3_file_path)
        print(f'Upload to S3 completed. Response: {response}')

    # Convenience method to download file from s3 to working directory
    def download_file_from_s3(self, s3_file_path, working_dir_file_path):
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client(
            service_name='s3'
        )
        response = s3_client.download_file(Bucket=self.aws_s3_bucket_name, Key=s3_file_path,
                                           Filename=working_dir_file_path)
        print(f'download_log_from_aws response: {response}')
        return response

    def generate_presigned_url(self, s3_file_path, expiration=3600):
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client('s3')
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.aws_s3_bucket_name, 'Key': s3_file_path},
            ExpiresIn=expiration
        )
        return presigned_url

    # Convenience method to download file from s3 as attachment from the UI
    def download_file_as_attachment(self, s3_file_path):
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client(
            service_name='s3',
            config=Config(signature_version=self.aws_signature_version)
        )
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.aws_s3_bucket_name, 'Key': s3_file_path},
            ExpiresIn=60
        )

    # List the downloadable output files
    def list_all_s3_files(self, s3_output_file_path):
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client(
            service_name='s3',
        )
        response = s3_client.list_objects_v2(
            Bucket=self.aws_s3_bucket_name,
            Prefix=s3_output_file_path
        )
        # Extract the file names (keys)
        files = []
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]

        return files
