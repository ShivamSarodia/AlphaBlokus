use anyhow::{Result, anyhow};
use aws_config;
use aws_sdk_s3::Client;

#[derive(Debug, Clone)]
pub struct ParsedUri {
    pub bucket_name: String,
    pub directory: Option<String>,
    pub filename: Option<String>,
}

impl ParsedUri {
    pub fn parse(uri: &str) -> Result<Self> {
        let without_scheme = uri
            .strip_prefix("s3://")
            .ok_or_else(|| anyhow!("S3 URI must start with s3://"))?;

        let mut parts = without_scheme.splitn(2, '/');
        let bucket = parts
            .next()
            .filter(|segment| !segment.is_empty())
            .ok_or_else(|| anyhow!("S3 URI missing bucket: {}", uri))?;

        let remainder = parts.next().unwrap_or("");
        let trimmed = remainder.trim_matches('/');

        if trimmed.is_empty() {
            return Ok(Self {
                bucket_name: bucket.to_string(),
                directory: None,
                filename: None,
            });
        }

        let has_trailing_slash = remainder.ends_with('/') || remainder.is_empty();
        let segments: Vec<&str> = trimmed.split('/').collect();
        let last_segment = segments.last().unwrap();
        let looks_like_file = !has_trailing_slash && last_segment.contains('.');

        if looks_like_file {
            let object = last_segment.to_string();
            let prefix = if segments.len() > 1 {
                Some(segments[..segments.len() - 1].join("/"))
            } else {
                None
            };

            Ok(Self {
                bucket_name: bucket.to_string(),
                directory: prefix,
                filename: Some(object),
            })
        } else {
            Ok(Self {
                bucket_name: bucket.to_string(),
                directory: Some(trimmed.to_string()),
                filename: None,
            })
        }
    }

    pub fn extension(&self) -> Option<String> {
        self.filename.as_ref().and_then(|name| {
            let mut segments = name.splitn(2, '.');
            segments.next();
            segments.next().map(|rest| rest.to_string())
        })
    }

    pub fn object_key(&self) -> Option<String> {
        self.filename
            .as_ref()
            .map(|file| match self.directory.as_deref() {
                Some(prefix) if !prefix.is_empty() => format!("{}/{}", prefix, file),
                _ => file.clone(),
            })
    }

    pub fn go_up(&self) -> Option<Self> {
        if self.filename.is_some() {
            let mut clone = self.clone();
            clone.filename = None;
            return Some(clone);
        }

        if let Some(dir) = &self.directory {
            let mut segments: Vec<&str> = dir
                .split('/')
                .filter(|segment| !segment.is_empty())
                .collect();
            if segments.is_empty() {
                return Some(Self {
                    bucket_name: self.bucket_name.clone(),
                    directory: None,
                    filename: None,
                });
            }

            segments.pop();
            let new_directory = if segments.is_empty() {
                None
            } else {
                Some(segments.join("/"))
            };

            return Some(Self {
                bucket_name: self.bucket_name.clone(),
                directory: new_directory,
                filename: None,
            });
        }

        None
    }

    pub fn go_down(&self, component: &str) -> Result<Self> {
        if self.filename.is_some() {
            return Err(anyhow!(
                "Cannot go down from a URI that already points to a file"
            ));
        }

        if component.is_empty() {
            return Err(anyhow!("Component must not be empty"));
        }

        let is_directory = component.ends_with('/');
        let trimmed = component.trim_matches('/');

        if trimmed.is_empty() {
            return Err(anyhow!("Component must not be empty"));
        }

        if trimmed.contains('/') {
            return Err(anyhow!("Component must not contain '/'"));
        }

        if is_directory {
            let new_dir = match &self.directory {
                Some(prefix) if !prefix.is_empty() => format!("{}/{}", prefix, trimmed),
                _ => trimmed.to_string(),
            };

            Ok(Self {
                bucket_name: self.bucket_name.clone(),
                directory: Some(new_dir),
                filename: None,
            })
        } else {
            Ok(Self {
                bucket_name: self.bucket_name.clone(),
                directory: self.directory.clone(),
                filename: Some(trimmed.to_string()),
            })
        }
    }
}

pub async fn build_client() -> Client {
    let mut loader = aws_config::from_env();

    if let Ok(endpoint) = std::env::var("S3_ENDPOINT_URL") {
        loader = loader.endpoint_url(endpoint);
    }

    let config = loader.load().await;
    Client::new(&config)
}
