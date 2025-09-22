import boto3

# In .env
s3 = boto3.resource(
    "s3",
    aws_access_key_id="",
    aws_secret_access_key="",
    region_name="",
    endpoint_url="",
)
bucket = s3.Bucket("alpha-blokus-data")

print("All content:")
for obj in bucket.objects.all():
    print(obj.key)
print("Done!")
