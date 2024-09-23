import boto3
from botocore.client import Config
import magic
import io

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

    # Convenience method to upload file to S3 bucket
    def upload_file_to_s3(self, file, s3_file_path):
        s3_client = boto3.session.Session(profile_name=self.aws_profile_name).client(
            service_name='s3'
        )
        response = s3_client.upload_fileobj(file, self.aws_s3_bucket_name, s3_file_path + file.filename)
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
            dataframe.to_excel(buffer, index=False, engine='xlsxwriter')

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
