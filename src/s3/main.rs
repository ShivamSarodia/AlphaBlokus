use anyhow::Result;
use aws_sdk_s3::Client;

#[derive(Debug)]
pub struct S3Uri {
    pub bucket: String,
    // The components of the path, excluding the bucket name and filename.
    inner_path: Vec<String>,
    pub filename: Option<String>,
}

impl S3Uri {
    pub fn new(path: String) -> Result<Self> {
        if !path.starts_with("s3://") {
            return Err(anyhow::anyhow!("Path must start with s3://"));
        }

        let without_protocol = path.split("://").nth(1).unwrap();
        let mut split_without_protocol = without_protocol.split("/");

        // First, grab the bucket name.
        let bucket = split_without_protocol.next().unwrap();

        // Combine the rest into a path.
        let mut inner_path: Vec<String> = split_without_protocol.map(str::to_string).collect();

        let filename = if let Some(last) = inner_path.last() {
            if last.contains(".") {
                inner_path.pop().map(|s| s.to_string())
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            bucket: bucket.to_string(),
            inner_path,
            filename,
        })
    }

    pub fn key(&self) -> String {
        let mut parts = self.inner_path.clone();
        if let Some(ref filename) = self.filename {
            parts.push(filename.clone());
        }
        parts.join("/")
    }

    /// Returns a new S3Uri with the specified filename.
    /// Ignores any existing filename
    pub fn with_filename(&self, filename: String) -> Result<Self> {
        Ok(Self {
            bucket: self.bucket.clone(),
            inner_path: self.inner_path.clone(),
            filename: Some(filename),
        })
    }
}

/// Creates an AWS S3 client configured with the endpoint URL from the S3_ENDPOINT_URL environment variable.
pub async fn create_s3_client() -> Client {
    let config = aws_config::from_env()
        .endpoint_url(std::env::var("AWS_ENDPOINT_URL").unwrap())
        .load()
        .await;
    Client::new(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_uri_with_file() {
        let uri = S3Uri::new("s3://my-bucket/path/to/file.txt".to_string()).unwrap();
        assert_eq!(uri.bucket, "my-bucket");
        assert_eq!(uri.filename, Some("file.txt".to_string()));
        assert_eq!(uri.key(), "path/to/file.txt");
    }

    #[test]
    fn test_s3_uri_with_nested_path_and_file() {
        let uri = S3Uri::new("s3://my-bucket/deeply/nested/path/model.pth".to_string()).unwrap();
        assert_eq!(uri.bucket, "my-bucket");
        assert_eq!(uri.filename, Some("model.pth".to_string()));
        assert_eq!(uri.key(), "deeply/nested/path/model.pth");
    }

    #[test]
    fn test_s3_uri_directory_only() {
        let uri = S3Uri::new("s3://my-bucket/path/to/directory".to_string()).unwrap();
        assert_eq!(uri.bucket, "my-bucket");
        assert_eq!(uri.filename, None);
        assert_eq!(uri.key(), "path/to/directory");
    }

    #[test]
    fn test_s3_uri_bucket_only() {
        let uri = S3Uri::new("s3://my-bucket".to_string()).unwrap();
        assert_eq!(uri.bucket, "my-bucket");
        assert_eq!(uri.filename, None);
        assert_eq!(uri.key(), "");
    }

    #[test]
    fn test_s3_uri_bucket_with_trailing_slash() {
        let uri = S3Uri::new("s3://my-bucket/".to_string()).unwrap();
        assert_eq!(uri.bucket, "my-bucket");
        assert_eq!(uri.filename, None);
        assert_eq!(uri.key(), "");
    }

    #[test]
    fn test_s3_uri_invalid_protocol() {
        let result = S3Uri::new("https://my-bucket/file.txt".to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must start with s3://")
        );
    }

    #[test]
    fn test_s3_uri_no_protocol() {
        let result = S3Uri::new("my-bucket/file.txt".to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must start with s3://")
        );
    }

    #[test]
    fn test_s3_uri_file_with_multiple_dots() {
        let uri = S3Uri::new("s3://my-bucket/archive.tar.gz".to_string()).unwrap();
        assert_eq!(uri.bucket, "my-bucket");
        assert_eq!(uri.filename, Some("archive.tar.gz".to_string()));
        assert_eq!(uri.key(), "archive.tar.gz");
    }

    #[test]
    fn test_with_filename_on_directory() {
        let uri = S3Uri::new("s3://my-bucket/path/to/directory".to_string()).unwrap();
        let uri_with_file = uri.with_filename("file.txt".to_string()).unwrap();
        assert_eq!(uri_with_file.bucket, "my-bucket");
        assert_eq!(uri_with_file.filename, Some("file.txt".to_string()));
        assert_eq!(uri_with_file.key(), "path/to/directory/file.txt");
    }

    #[test]
    fn test_with_filename_on_bucket_only() {
        let uri = S3Uri::new("s3://my-bucket".to_string()).unwrap();
        let uri_with_file = uri.with_filename("file.txt".to_string()).unwrap();
        assert_eq!(uri_with_file.bucket, "my-bucket");
        assert_eq!(uri_with_file.filename, Some("file.txt".to_string()));
        assert_eq!(uri_with_file.key(), "file.txt");
    }
}
