from clearml import Task, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the task
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 2 process dataset")

# Connect parameters
args = {
    'dataset_task_id': '',  # Will be set from pipeline
    'test_size': 0.25,
    'random_state': 42
}
task.connect(args)

# Execute the task remotely
task.execute_remotely()

# Get the dataset task ID from pipeline parameters
dataset_task_id = task.get_parameter('General/dataset_task_id')
logger.info(f"Using dataset task ID: {dataset_task_id}")

# Load the raw dataset
dataset_task = Task.get_task(task_id=dataset_task_id)
raw_data = dataset_task.artifacts['iris_dataset'].get()

# Process the data
X = raw_data.drop('target', axis=1)
y = raw_data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args['test_size'], random_state=args['random_state']
)

# Create a new dataset in ClearML
dataset = Dataset.create(
    dataset_name="Iris Processed Dataset",
    dataset_project="AI_Studio_Demo"
)

# Add the processed data to the dataset
dataset.add_dataframe(
    dataframe=pd.DataFrame(X_train),
    name="X_train"
)
dataset.add_dataframe(
    dataframe=pd.DataFrame(X_test),
    name="X_test"
)
dataset.add_dataframe(
    dataframe=pd.DataFrame(y_train),
    name="y_train"
)
dataset.add_dataframe(
    dataframe=pd.DataFrame(y_test),
    name="y_test"
)

# Finalize the dataset
dataset.finalize()
logger.info(f"Dataset created with ID: {dataset.id}")

# Store the dataset ID as a task parameter for other steps to use
task.set_parameter("General/processed_dataset_id", dataset.id)
logger.info(f"Stored processed dataset ID: {dataset.id}")

print("Dataset processing completed and uploaded to ClearML") 