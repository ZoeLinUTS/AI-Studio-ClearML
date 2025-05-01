from clearml import Task, StorageManager

# create an dataset experiment
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 1 dataset artifact")

# only create the task, we will actually execute it later
# task.execute_remotely()

# simulate local dataset, download one, so we have something local
# local_iris_pkl = StorageManager.get_local_copy(
#     remote_url='https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl')
# local_iris_csv_path = 'work_dataset/Iris.csv'

# add and upload local file containing our toy dataset
task.upload_artifact('dataset', artifact_object='work_dataset/Iris.csv')

print('uploading artifacts in the background')

# we are done
print('Done')