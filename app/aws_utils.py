import boto3
import magic


# AWS Utilities
class AwsUtils:
    def __init__(self, allowed_extensions, allowed_mime_types, aws_s3_bucket_name):
        self.allowed_extensions = allowed_extensions
        self.allowed_mime_types = allowed_mime_types
        self.aws_s3_bucket_name = aws_s3_bucket_name

    # Validation to check file extensions , Content types and file sizes
    def allowed_file(self, filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def allowed_mime(self, file):
        mime = magic.from_buffer(file.stream.read(2048), mime=True)
        file.stream.seek(0)  # Reset file pointer after reading
        return mime in self.allowed_mime_types

    # Convenience method to upload file to S3 bucket
    def upload_file_to_s3(self, file, s3_file_path):
        s3_client = boto3.session.Session(profile_name='tailoring_api').client(
            service_name='s3',
        )
        response = s3_client.upload_fileobj(file, self.aws_s3_bucket_name, s3_file_path + file.filename)
        print(f'upload_log_to_aws response: {response}')

    # Convenience method to download file from s3 to working directory
    # note file_path is the key without the bucket name
    def download_file_from_s3(self, file_path):
        s3_client = boto3.session.Session(profile_name='tailoring_api').client(
            service_name='s3',
        )
        response = s3_client.download_file(Bucket=self.aws_s3_bucket_name, Key=file_path, Filename="test_aws.xlsx")
        print(f'download_log_from_aws response: {response}')
        return file_path

    # list the downloadable output files
    def list_all_s3_files(self, s3_output_file_path):
        s3_client = boto3.session.Session(profile_name='tailoring_api').client(
            service_name='s3',
        )
        response = s3_client.list_objects_v2(
            Bucket=self.aws_s3_bucket_name,
            Prefix=s3_output_file_path
        )

        # Extract the file names (keys)
        # Extract the file names (keys)
        files = []
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]

        return files
