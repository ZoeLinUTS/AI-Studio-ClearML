import pickle
from clearml import Task, StorageManager
from sklearn.model_selection import train_test_split
import pandas as pd

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 2 process dataset")

# program arguments
# Use either dataset_task_id to point to a tasks artifact or
# use a direct url with dataset_url
args = {
    'dataset_task_id': '3ceaa4409a70486b846e35bcbf229eab', #update id if it needs running locally
    # 'dataset_task_id': '',  # update id if it needs running locally
    'dataset_url': '',
    'random_state': 42,
    'test_size': 0.2,
}

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later
task.execute_remotely()

# get dataset from task's artifact
if args['dataset_task_id']:
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    print('Input task id={} artifacts {}'.format(args['dataset_task_id'], list(dataset_upload_task.artifacts.keys())))
    # download the artifact
    iris_csv = dataset_upload_task.artifacts['dataset'].get_local_copy()
# # get the dataset from a direct url
# elif args['dataset_url']:
#     iris_pickle = StorageManager.get_local_copy(remote_url=args['dataset_url'])
else:
    raise ValueError("Missing dataset link")

iris_df = pd.read_csv(iris_csv)


# "process" data
# Extract features (X) and target (y)
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = iris_df['Species'].astype('category').cat.codes.values  # Convert species to numeric codes

species_mapping = dict(enumerate(iris_df['Species'].astype('category').cat.categories))
print(species_mapping)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args['test_size'], random_state=args['random_state'])

# upload processed data
print('Uploading process dataset')
task.upload_artifact('X_train', X_train)
task.upload_artifact('X_test', X_test)
task.upload_artifact('y_train', y_train)
task.upload_artifact('y_test', y_test)

print('Notice, artifacts are uploaded in the background')
print('Done')
