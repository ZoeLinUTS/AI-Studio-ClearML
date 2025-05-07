from clearml import Task, StorageManager
import os

# create an dataset experiment
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 1 dataset artifact")

# only create the task, we will actually execute it later
# task.execute_remotely()

# Check if the local dataset file exists
local_iris_csv_path = 'work_dataset/Iris.csv'
if not os.path.exists(local_iris_csv_path):
    print(f"Local file '{local_iris_csv_path}' not found. Downloading...")
    local_iris_pkl = StorageManager.get_local_copy(
        remote_url='https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl'
    )
else:
    print(f"Using existing local file: '{local_iris_csv_path}'")

# Add and upload the dataset file
task.upload_artifact('dataset', artifact_object=local_iris_csv_path)
print('uploading artifacts in the background')

# we are done
print('Done')