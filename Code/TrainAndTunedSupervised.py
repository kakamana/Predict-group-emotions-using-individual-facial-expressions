import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from SupervisedModelTraining import FacialExpressionModel, SVMModel, RandomForestModel
from FeatureEngineering import extract_features
from DataSetPrep import FacialExpressionDataset  

def train_cnn(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_cnn_model.pth')

    return model

def train_and_evaluate_cnn(train_data, val_data, test_data, params):
    model = FacialExpressionModel(num_classes=7).to(params['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'])
    test_loader = DataLoader(test_data, batch_size=params['batch_size'])

    model = train_cnn(model, train_loader, val_loader, criterion, optimizer, params['device'], params['num_epochs'])
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(params['device'])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

def train_and_evaluate_svm(train_data, test_data, params):
    X_train = np.array([extract_features(img) for img, _ in train_data])
    y_train = np.array([label for _, label in train_data])
    X_test = np.array([extract_features(img) for img, _ in test_data])
    y_test = np.array([label for _, label in test_data])

    svm = SVMModel(**params)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

def train_and_evaluate_rf(train_data, test_data, params):
    X_train = np.array([extract_features(img) for img, _ in train_data])
    y_train = np.array([label for _, label in train_data])
    X_test = np.array([extract_features(img) for img, _ in test_data])
    y_test = np.array([label for _, label in test_data])

    rf = RandomForestModel(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

def hyperparameter_tuning():
    # Load your dataset   
    train_data = FacialExpressionDataset('dataset/trainCelb')
    val_data = FacialExpressionDataset('dataset/valCelb')
    test_data = FacialExpressionDataset('dataset/testCelb')

    # CNN hyperparameter tuning
    cnn_param_grid = {
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64],
        'num_epochs': [10, 20],
        'device': [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    }

    best_cnn_accuracy = 0
    best_cnn_params = None
    for params in cnn_param_grid:
        accuracy, _ = train_and_evaluate_cnn(train_data, val_data, test_data, params)
        if accuracy > best_cnn_accuracy:
            best_cnn_accuracy = accuracy
            best_cnn_params = params

    print(f"Best CNN params: {best_cnn_params}")
    print(f"Best CNN accuracy: {best_cnn_accuracy}")

    # SVM hyperparameter tuning
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }

    svm = SVMModel()
    svm_grid_search = GridSearchCV(svm.model, svm_param_grid, cv=3, scoring='accuracy')
    X_train = np.array([extract_features(img) for img, _ in train_data])
    y_train = np.array([label for _, label in train_data])
    svm_grid_search.fit(X_train, y_train)

    print(f"Best SVM params: {svm_grid_search.best_params_}")
    print(f"Best SVM accuracy: {svm_grid_search.best_score_}")

    # Random Forest hyperparameter tuning
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestModel()
    rf_grid_search = GridSearchCV(rf.model, rf_param_grid, cv=3, scoring='accuracy')
    rf_grid_search.fit(X_train, y_train)

    print(f"Best Random Forest params: {rf_grid_search.best_params_}")
    print(f"Best Random Forest accuracy: {rf_grid_search.best_score_}")

if __name__ == "__main__":
    hyperparameter_tuning()