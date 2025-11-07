import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# ----------------------------
# CONFIGURATION
# ----------------------------
region = "us-east-1"
account = "849661979010"

role = "arn:aws:iam::849661979010:role/MLE"   # <-- Your SageMaker execution role

image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/scikit_bring_your_own:latest"

session = sagemaker.Session()
bucket = session.default_bucket()

# S3 path for training data
input_s3 = f"s3://{bucket}/byoc-training/iris.csv"

# ----------------------------
# UPLOAD TRAINING DATA TO S3
# ----------------------------

print("Uploading training data to S3...")
session.upload_data(
     path="/Users/jaebonglee/Documents/Projects/amazon-sagemaker-examples/advanced_functionality/scikit_bring_your_own/container/local_test/test_dir/input/data/training/iris.csv",
    bucket=bucket,
    key_prefix="byoc-training"
)

print(f"Training data uploaded to: {input_s3}")

# ----------------------------
# SAGEMAKER TRAINING JOB
# ----------------------------

print("Starting SageMaker training job...")

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path=f"s3://{bucket}/byoc-output",
    sagemaker_session=session,
    base_job_name="scikit-bring-your-own"
)

estimator.fit({"training": input_s3})
print("Training job completed!")

# ----------------------------
# DEPLOY MODEL TO ENDPOINT
# ----------------------------

print("Deploying SageMaker endpoint...")
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    serializer=CSVSerializer(),
    deserializer=None,
)

print("Endpoint deployed successfully!")

# ----------------------------
# TEST INFERENCE
# ----------------------------

sample = [5.1, 3.5, 1.4, 0.2]
print(f"Sending sample input {sample} to endpoint...")

result = predictor.predict(sample).decode("utf-8").strip()
print("Prediction:", result)

