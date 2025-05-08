import matplotlib.pyplot as plt
from clearml import Task, Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import time
import os

# Create necessary directories
os.makedirs('assets', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Connecting ClearML with the current process,
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 3 train model")
logger = Logger.current_logger()

# Connect parameters
args = {
    'dataset_task_id': '',  # Will be set from pipeline
    'num_epochs': 20,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
}

task.connect(args)

# Execute the task remotely
task.execute_remotely()

# Get the dataset task ID from pipeline parameters
dataset_task_id = task.get_parameter('General/dataset_task_id')
print(f"All task parameters: {task.get_parameters()}")
print(f"Raw dataset task ID from parameters: {dataset_task_id}")

if not dataset_task_id:
    raise ValueError("Dataset task ID not found in parameters. Please ensure it's passed from the pipeline.")

print('Retrieving Iris dataset')

# Wait for artifacts to be available
max_retries = 20  # Increased from 10
retry_delay = 30  # Increased from 20
for attempt in range(max_retries):
    try:
        print(f'Attempt {attempt + 1}/{max_retries} to load artifacts...')
        print(f'Dataset task ID: {dataset_task_id}')
        dataset_task = Task.get_task(task_id=dataset_task_id)
        print(f'Dataset task name: {dataset_task.name}')
        print(f'Dataset task project: {dataset_task.project}')
        print(f'Available artifacts: {list(dataset_task.artifacts.keys())}')
        
        # Add explicit wait for artifacts to be ready
        if not dataset_task.artifacts:
            print('No artifacts found, waiting...')
            time.sleep(retry_delay)
            continue
            
        # Try to get each artifact individually with better error handling
        try:
            X_train = dataset_task.artifacts['X_train'].get()
            print('Successfully loaded X_train')
        except Exception as e:
            print(f'Error loading X_train: {str(e)}')
            raise
            
        try:
            X_test = dataset_task.artifacts['X_test'].get()
            print('Successfully loaded X_test')
        except Exception as e:
            print(f'Error loading X_test: {str(e)}')
            raise
            
        try:
            y_train = dataset_task.artifacts['y_train'].get()
            print('Successfully loaded y_train')
        except Exception as e:
            print(f'Error loading y_train: {str(e)}')
            raise
            
        try:
            y_test = dataset_task.artifacts['y_test'].get()
            print('Successfully loaded y_test')
        except Exception as e:
            print(f'Error loading y_test: {str(e)}')
            raise
            
        print('Iris dataset loaded successfully')
        break
    except (KeyError, Exception) as e:
        print(f'Error loading artifacts: {str(e)}')
        if attempt < max_retries - 1:
            print(f'Artifacts not ready yet, waiting {retry_delay} seconds...')
            time.sleep(retry_delay)
        else:
            print('Failed to load artifacts after maximum retries')
            raise

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size=X_train.shape[1], num_classes=len(set(y_train)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=args['learning_rate'],
    weight_decay=args['weight_decay']
)

for epoch in tqdm(range(args['num_epochs']), desc='Training Epochs'):
    epoch_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    logger.report_scalar(title='train', series='epoch_loss', value=avg_loss, iteration=epoch)

# Save model
model_path = 'assets/model.pkl'
torch.save(model.state_dict(), model_path)
task.upload_artifact(name='model', artifact_object=model_path)
print('Model saved and uploaded as artifact')

# Load model for evaluation
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean().item()
    logger.report_scalar("validation", "accuracy", value=accuracy, iteration=0)

print(f'Model trained & stored with accuracy: {accuracy:.4f}')

# Plotting confusion matrix
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
y_test_names = [species_mapping[label.item()] for label in y_test]
predicted_names = [species_mapping[label.item()] for label in predicted]

cm = confusion_matrix(y_test_names, predicted_names, labels=list(species_mapping.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(species_mapping.values()))
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.savefig('figs/confusion_matrix.png')

print('Confusion matrix plotted and saved as confusion_matrix.png')