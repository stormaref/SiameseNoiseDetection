import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, FashionMNIST
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import LinearLR
from models.dataset import CleanDatasetLoader, CleanWrapperDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from models.preact import PreActResNet18, PreActResNet34
import numpy as np

class FinalModelTester:
    def __init__(self, train_dataset_path: str, train_transform: transforms.transforms, test_transform: transforms.transforms,
                 train_batch_size=256, val_batch_size=256, test_batch_size=64, pretrained=True, lr=0.001, warmup_epochs=5,
                 patience=5, weight_decay=1e-5, use_default_train=False, milestones=[80, 120], use_lr_scheduler=True,
                 freeze=True, smoothing=0, test='cifar', val_ratio=0.1):

        self.freeze = freeze
        self.model_checkpoint_path = 'best-final-model.pth'
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained = pretrained
        self.cleaned_dataset = CleanDatasetLoader(train_dataset_path, None)
        if use_default_train:
            self.cleaned_dataset = CIFAR10(root='data', train=True, download=True, transform=train_transform)
        print(f"Dataset size: {self.cleaned_dataset.__len__()}")

        train_indices, val_indices = train_test_split(
            range(len(self.cleaned_dataset)),
            test_size=val_ratio,
            stratify=self.cleaned_dataset.labels if not use_default_train else self.cleaned_dataset.targets
        )

        self.train_dataset = CleanWrapperDataset(self.cleaned_dataset, train_indices, train_transform)
        if use_default_train:
            self.train_dataset = Subset(self.cleaned_dataset, train_indices)
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True)

        self.val_dataset = CleanWrapperDataset(self.cleaned_dataset, val_indices, test_transform)
        if use_default_train:
            self.val_dataset = Subset(self.cleaned_dataset, val_indices)
        self.val_loader = DataLoader(self.val_dataset, batch_size=val_batch_size)

        if test == 'cifar':
            self.test_dataset = CIFAR10(root='data', train=False, download=True, transform=test_transform)
        elif test == 'fmnist':
            self.test_dataset = FashionMNIST(root='data', train=False, download=True, transform=test_transform)
        else:
            raise 'wtf'
        self.test_loader = DataLoader(self.test_dataset, batch_size=test_batch_size)

        self.model = self.get_model().to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.weight_decay, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay)

        # Warmup scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)

        # Main scheduler
        self.use_lr_scheduler = use_lr_scheduler
        self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200 - warmup_epochs)

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.patience = patience
        self.best_val_accuracy = 0
        self.early_stop_counter = 0

    def get_model(self):
        # base = resnet34(weights=ResNet34_Weights.DEFAULT) if self.pretrained else resnet34()
        # if self.pretrained and self.freeze:
        #     # Freeze all layers except the fully connected layer
        #     for param in base.parameters():
        #         param.requires_grad = False
        #     # Ensure the fully connected layer is trainable
        #     # base.fc = torch.
        #     base.fc = torch.nn.Sequential(
        #         nn.Linear(base.fc.in_features, 128),
        #         nn.Linear(128, 10)
        #     )
        #     for param in base.fc.parameters():
        #         param.requires_grad = True
        # else:
        #     # base.fc = torch.nn.Linear(base.fc.in_features, 10)
        #     base.fc = torch.nn.Sequential(
        #         nn.Linear(base.fc.in_features, 128),
        #         nn.Linear(128, 10)
        #     )
        # return base
        # return torchvision.models.densenet121()
        return PreActResNet34()
        # model = CNNModel(num_classes=10, img_channels=3)
        # return model

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, epochs):
        progress_bar = tqdm(range(epochs))
        for epoch in progress_bar:
            epoch_loss = 0.0
            correct = 0
            total = 0
            self.model.train()
            for data in self.train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            accuracy = correct / total

            self.train_losses.append(avg_epoch_loss)
            self.train_accuracies.append(accuracy)
            progress_bar.set_postfix({'Training Loss': avg_epoch_loss, 'Training Accuracy': accuracy})

            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            progress_bar.set_postfix({'Training Loss': avg_epoch_loss, 'Training Accuracy': accuracy, 
                                      'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy})

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.early_stop_counter = 0
                self.save_model(self.model_checkpoint_path)
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                tqdm.write("Early stopping triggered")
                break

            # Adjust learning rate
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            elif self.use_lr_scheduler:
                self.main_scheduler.step()

            # Record learning rate for plotting
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

        self.model.load_state_dict(torch.load(self.model_checkpoint_path))

    def validate(self):
        epoch_loss = 0.0
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_epoch_loss = epoch_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_epoch_loss, accuracy

    def plot_learning_rate(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.learning_rates)), self.learning_rates, marker='o', linestyle='-')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.show()

    def test(self):
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')

        # Plot confusion matrix
        # self.plot_confusion_matrix(all_labels, all_preds)
        return accuracy

    def plot_confusion_matrix(self, true_labels, pred_labels, normalize=None):
        cm = confusion_matrix(true_labels, pred_labels, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.test_dataset.classes)

        plt.figure(figsize=(10, 7))
        disp.plot(cmap=plt.cm.Blues, values_format='.2f' if normalize else 'd')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.show()


    def objective(self, trial):
        print(trial)
        # Suggest hyperparameters
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        model_type = trial.suggest_categorical('model', ['resnet18', 'resnet34', 'preact_resnet18', 'preact_resnet34'])
        smoothing = trial.suggest_categorical('smoothing', ['0.1', 'disable'])
        
        # Initialize the model based on the suggestion
        if model_type == 'resnet18':
            model = resnet18(weights=ResNet18_Weights.DEFAULT) if self.pretrained else resnet18()
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
        elif model_type == 'resnet34':
            model = resnet34(weights=ResNet34_Weights.DEFAULT) if self.pretrained else resnet34()
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
        elif model_type == 'preact_resnet18':
            model = PreActResNet18()
        elif model_type == 'preact_resnet34':
            model = PreActResNet34()
        
        model = model.to(self.device)
    
        # Define optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if smoothing == 'disable':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
        # Training and validation
        best_val_accuracy = 0
        patience = 5
        early_stop_counter = 0
        
        for epoch in range(15):  # Limiting to 10 epochs for faster optimization
            model.train()
            for data in self.train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.val_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = correct / total
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                break
        
        return -best_val_accuracy  # Return negative accuracy for minimization

class FinalEvaluator:
    def __init__(self, train_dataset_path: str, train_transform: transforms.transforms, test_transform: transforms.transforms,
                 train_batch_size=256, val_batch_size=256, test_batch_size=64, pretrained=True, lr=0.001, warmup_epochs=5,
                 patience=5, weight_decay=1e-5, use_default_train=False, milestones=[80, 120], use_lr_scheduler=True,
                 freeze=True, smoothing=0, test='cifar', val_ratio=0.1):
        
        self.train_dataset_path = train_dataset_path
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.pretrained = pretrained
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.use_default_train = use_default_train
        self.milestones = milestones
        self.use_lr_scheduler = use_lr_scheduler
        self.freeze = freeze
        self.smoothing = smoothing
        self.test = test
        self.val_ratio = val_ratio
        self.accuracies = []

    def evaluate(self, n_trials=5):
        accuracies = []
        for _ in range(n_trials):
            final_model_tester = FinalModelTester(train_dataset_path=self.train_dataset_path, train_transform=self.train_transform, test_transform=self.test_transform,
                                                train_batch_size=self.train_batch_size, val_batch_size=self.val_batch_size, test_batch_size=self.test_batch_size,
                                                pretrained=self.pretrained, lr=self.lr, warmup_epochs=self.warmup_epochs, patience=self.patience,
                                                weight_decay=self.weight_decay, use_default_train=self.use_default_train, milestones=self.milestones,
                                                use_lr_scheduler=self.use_lr_scheduler, freeze=self.freeze, smoothing=self.smoothing, test=self.test, val_ratio=self.val_ratio)
            
            final_model_tester.train(epochs=200)
            accuracy = final_model_tester.test()
            accuracies.append(accuracy)
        
        self.accuracies = accuracies
        return accuracies
        
    def calculate_mean_std(self):
        return np.mean(self.accuracies), np.std(self.accuracies)
    
    def get_best_accuracy(self):
        return max(self.accuracies)
    
    def report(self):
        mean, std = self.calculate_mean_std()
        print(f"Result: {mean} ± {std}")
        print(f"Best accuracy: {self.get_best_accuracy()}")