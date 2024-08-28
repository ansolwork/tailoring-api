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
    def upload_file_to_s3(self, file, file_path):
        s3_client = boto3.session.Session(profile_name='tailoring_api').client(
            service_name='s3',
        )
        response = s3_client.upload_fileobj(file, self.aws_s3_bucket_name, file_path + file.filename)
        print(f'upload_log_to_aws response: {response}')
