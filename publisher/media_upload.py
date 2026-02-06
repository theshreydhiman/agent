"""Media upload to S3/MinIO for Instagram publishing."""

import os
import uuid
from datetime import datetime

import boto3
from botocore.config import Config
import structlog

from config import get_settings

logger = structlog.get_logger()


class MediaUploader:
    """Uploads media to S3-compatible storage and generates accessible URLs."""

    def __init__(self):
        settings = get_settings()
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=Config(signature_version="s3v4"),
        )
        self._bucket = settings.s3_bucket_name
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except Exception:
            try:
                self._client.create_bucket(Bucket=self._bucket)
                logger.info("bucket_created", bucket=self._bucket)
            except Exception as e:
                logger.warning("bucket_create_failed", error=str(e))

    def _generate_key(self, extension: str) -> str:
        """Generate a unique S3 key with date prefix."""
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        unique_id = uuid.uuid4().hex[:12]
        return f"media/{date_prefix}/{unique_id}.{extension}"

    def upload_to_s3(
        self, file_path: str, content_type: str = "image/jpeg"
    ) -> str:
        """Upload a local file to S3/MinIO and return the S3 key."""
        ext = os.path.splitext(file_path)[1].lstrip(".")
        s3_key = self._generate_key(ext)

        self._client.upload_file(
            file_path, self._bucket, s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info("uploaded_to_s3", key=s3_key, size=os.path.getsize(file_path))
        return s3_key

    def upload_image(self, image_path: str) -> str:
        """Upload an image and return a presigned URL."""
        ext = os.path.splitext(image_path)[1].lower()
        content_type = "image/png" if ext == ".png" else "image/jpeg"
        s3_key = self.upload_to_s3(image_path, content_type)
        return self.generate_presigned_url(s3_key)

    def upload_video(self, video_path: str) -> str:
        """Upload a video and return a presigned URL."""
        s3_key = self.upload_to_s3(video_path, "video/mp4")
        return self.generate_presigned_url(s3_key, expires_in=7200)

    def generate_presigned_url(
        self, s3_key: str, expires_in: int = 3600
    ) -> str:
        """Generate a temporary public URL for a stored object."""
        url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": s3_key},
            ExpiresIn=expires_in,
        )
        return url

    def delete_media(self, s3_key: str):
        """Delete a media object from storage."""
        self._client.delete_object(Bucket=self._bucket, Key=s3_key)
        logger.info("deleted_from_s3", key=s3_key)
