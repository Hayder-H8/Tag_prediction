import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os 
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score ,accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier



from utils import hamming_loss


def train_and_evaluate(model , features ,tags,num_epochs=20 ,threshold=0.3):
    train_size = int(0.8 * len(features))  # 80% for training
    
    

    # Ensure data is in PyTorch tensor format
    X_train_tensor = features[:train_size]
    y_train_tensor = tags[:train_size]
    X_test_tensor = features[train_size+1:]
    y_test_tensor = tags[train_size+1:]
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
        
    
    # Evaluation loop
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            # Apply threshold to get binary predictions
            predicted = (outputs > threshold).float()  
            y_true.append(batch_y)
            y_pred.append(predicted)
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    
    hamming = hamming_loss(y_true, y_pred)
    per_class_precision = precision_score(y_true, y_pred, average=None,zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None,zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None,zero_division=0)
    
    print(f"Hamming Loss: {hamming:.4f}")

    print("\nPer-Class Metrics:")
    for i, (p, r, f1) in enumerate(zip(per_class_precision, per_class_recall, per_class_f1)):
        print(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")
    
    
    
    return per_class_f1, per_class_precision, per_class_recall, hamming






def train_logistic_regression_classifiers(X, Y, threshold=0.5):

    num_classes = Y.shape[1]
    models = []

    all_predictions = []
    all_true_labels = []

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    for i in range(num_classes):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train[:, i])  
        models.append(model)

        probabilities = model.predict_proba(X_test)[:, 1]  
        predictions = (probabilities >= threshold).astype(int)  
        
        all_predictions.append(predictions)
        all_true_labels.append(Y_test[:, i])

    all_predictions = np.array(all_predictions).T  
    all_true_labels = np.array(all_true_labels).T  

    hamming = hamming_loss(all_true_labels, all_predictions)
    per_class_precision = precision_score(all_true_labels, all_predictions, average=None,zero_division=0)
    per_class_recall = recall_score(all_true_labels, all_predictions, average=None,zero_division=0)
    per_class_f1 = f1_score(all_true_labels, all_predictions, average=None,zero_division=0)

    # Combine metrics into a dictionary
    metrics = {
        "Hamming Loss": hamming,
        "Per-Class Precision": per_class_precision,
        "Per-Class Recall": per_class_recall,
        "Per-Class F1-Score": per_class_f1
    }



    return models, metrics


from sklearn.model_selection import train_test_split, StratifiedKFold

def cross_validate_threshold(X, Y, thresholds, num_splits=5):
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    avg_metrics = {threshold: {
        "Hamming Loss": [],
        "Per-Class Precision": [],
        "Per-Class Recall": [],
        "Per-Class F1-Score": []
    } for threshold in thresholds}
    
    for train_index, test_index in skf.split(X, Y.argmax(axis=1)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        for threshold in thresholds:
            _, metrics = train_logistic_regression_classifiers(X_train, Y_train, threshold)
            avg_metrics[threshold]["Hamming Loss"].append(metrics["Hamming Loss"])
            avg_metrics[threshold]["Per-Class Precision"].append(metrics["Per-Class Precision"])
            avg_metrics[threshold]["Per-Class Recall"].append(metrics["Per-Class Recall"])
            avg_metrics[threshold]["Per-Class F1-Score"].append(metrics["Per-Class F1-Score"])

    # Calculate average metrics over all folds for each threshold
    avg_metrics = {
        threshold: {
            "Hamming Loss": np.mean(values["Hamming Loss"]),
            "Per-Class Precision": np.mean(values["Per-Class Precision"], axis=0),
            "Per-Class Recall": np.mean(values["Per-Class Recall"], axis=0),
            "Per-Class F1-Score": np.mean(values["Per-Class F1-Score"], axis=0),
        } for threshold, values in avg_metrics.items()
    }

    # Find the best threshold based on the lowest Hamming Loss
    best_threshold = min(avg_metrics, key=lambda t: avg_metrics[t]["Hamming Loss"])

    # Print cross-validation results
    print("Cross-Validation Results:")
    for threshold, metrics in avg_metrics.items():
        print(f"\nThreshold {threshold:.2f} -> Hamming Loss: {metrics['Hamming Loss']:.4f}")
        print("Per-Class Metrics:")
        for i, (p, r, f1) in enumerate(zip(metrics["Per-Class Precision"], metrics["Per-Class Recall"], metrics["Per-Class F1-Score"])):
            print(f"  Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")

    print(f"\nBest Threshold: {best_threshold:.2f}")

    return best_threshold, avg_metrics


# Train and evaluate multiple classifier chains
def train_multiple_classifier_chains(X, Y, threshold=0.5, num_chains=5):
    metrics_list = []
    all_chain_predictions = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    for chain_index in range(num_chains):
        print(f"\nTraining Classifier Chain {chain_index + 1}...")
        base_classifier = LogisticRegression(max_iter=1000)
        chain = ClassifierChain(base_classifier, order='random', random_state=chain_index)
        chain.fit(X_train, Y_train)
        Y_prob = chain.predict_proba(X_test)
        Y_pred = (Y_prob >= threshold).astype(int)
        all_chain_predictions.append(Y_pred)

        

    all_chain_predictions = np.array(all_chain_predictions)
    Y_final_pred = np.mean(all_chain_predictions, axis=0) >= 0.5  
    # Evaluate 
    hamming = hamming_loss(Y_test, Y_final_pred)
    per_class_precision = precision_score(Y_test, Y_final_pred, average=None,zero_division=0)
    per_class_recall = recall_score(Y_test, Y_final_pred, average=None,zero_division=0)
    per_class_f1 = f1_score(Y_test, Y_final_pred, average=None,zero_division=0)

    metrics = {
        "Hamming Loss": hamming,
        "Per-Class Precision": per_class_precision,
        "Per-Class Recall": per_class_recall,
        "Per-Class F1-Score": per_class_f1
    }

    print(f"Hamming Loss: {hamming:.4f}")
    print("\nPer-Class Metrics:")
    for i, (p, r, f1) in enumerate(zip(per_class_precision, per_class_recall, per_class_f1)):
        print(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")
    

from sklearn.svm import SVC

def train_svm_classifiers(X, Y, threshold=0.5):

    num_classes = Y.shape[1]
    models = []

    all_predictions = []
    all_true_labels = []

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    for i in range(num_classes):
        model = SVC(probability=True, kernel="linear", random_state=42)
        model.fit(X_train, Y_train[:, i])  
        models.append(model)

        # Predict probabilities and apply the threshold
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        all_predictions.append(predictions)
        all_true_labels.append(Y_test[:, i])

    all_predictions = np.array(all_predictions).T  
    all_true_labels = np.array(all_true_labels).T  
    hamming = hamming_loss(all_true_labels, all_predictions)
    per_class_precision = precision_score(all_true_labels, all_predictions, average=None,zero_division=0)
    per_class_recall = recall_score(all_true_labels, all_predictions, average=None,zero_division=0)
    per_class_f1 = f1_score(all_true_labels, all_predictions, average=None,zero_division=0)

    metrics = {
        "Hamming Loss": hamming,
        "Per-Class Precision": per_class_precision,
        "Per-Class Recall": per_class_recall,
        "Per-Class F1-Score": per_class_f1
    }

    return models, metrics


def cross_validate_svm_threshold(X, Y, thresholds, num_splits=5):
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    # Initialize dictionary to store metrics for each threshold
    avg_metrics = {threshold: {
        "Hamming Loss": [],
        "Per-Class Precision": [],
        "Per-Class Recall": [],
        "Per-Class F1-Score": []
    } for threshold in thresholds}
    
    for train_index, test_index in skf.split(X, Y.argmax(axis=1)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        for threshold in thresholds:
            _, metrics = train_svm_classifiers(X_train, Y_train, threshold)
            avg_metrics[threshold]["Hamming Loss"].append(metrics["Hamming Loss"])
            avg_metrics[threshold]["Per-Class Precision"].append(metrics["Per-Class Precision"])
            avg_metrics[threshold]["Per-Class Recall"].append(metrics["Per-Class Recall"])
            avg_metrics[threshold]["Per-Class F1-Score"].append(metrics["Per-Class F1-Score"])

    avg_metrics = {
        threshold: {
            "Hamming Loss": np.mean(values["Hamming Loss"]),
            "Per-Class Precision": np.mean(values["Per-Class Precision"], axis=0),
            "Per-Class Recall": np.mean(values["Per-Class Recall"], axis=0),
            "Per-Class F1-Score": np.mean(values["Per-Class F1-Score"], axis=0),
        } for threshold, values in avg_metrics.items()
    }

    best_threshold = min(avg_metrics, key=lambda t: avg_metrics[t]["Hamming Loss"])

    print("Cross-Validation Results:")
    for threshold, metrics in avg_metrics.items():
        print(f"\nThreshold {threshold:.2f} -> Hamming Loss: {metrics['Hamming Loss']:.4f}")
        print("Per-Class Metrics:")
        for i, (p, r, f1) in enumerate(zip(metrics["Per-Class Precision"], metrics["Per-Class Recall"], metrics["Per-Class F1-Score"])):
            print(f"  Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")

    print(f"\nBest Threshold: {best_threshold:.2f}")

    return best_threshold, avg_metrics



def train_and_evaluate_random_forest(X, Y, test_size=0.2, random_state=42, n_estimators=100):
    """
    Train and evaluate a Random Forest model for multilabel classification.
    
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    num_classes = Y.shape[1]
    models = []
    y_pred = []

    for i in range(num_classes):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, Y_train[:, i])  
        models.append(model)

        # Predict on the test set
        predictions = model.predict(X_test)
        y_pred.append(predictions)

    y_pred = np.array(y_pred).T

    # Evaluate 
    hamming = hamming_loss(Y_test, y_pred)
    per_class_precision = precision_score(Y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(Y_test, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(Y_test, y_pred, average=None, zero_division=0)

    metrics = {
        "Hamming Loss": hamming,
        "Per-Class Precision": per_class_precision,
        "Per-Class Recall": per_class_recall,
        "Per-Class F1-Score": per_class_f1
    }
    # Print evaluation metrics
    print(f"Hamming Loss: {hamming:.4f}")

    print("\nPer-Class Metrics:")
    for i, (p, r, f1) in enumerate(zip(per_class_precision, per_class_recall, per_class_f1)):
        print(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")

    return models, metrics

        
        
