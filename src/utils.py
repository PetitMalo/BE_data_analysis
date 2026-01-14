import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def predict_classifier(classifier, x_data, y_true, plot_cm=False):
    """
    Predict the labels using the given classifier and compute the confusion matrix.

    Parameters:
    classifier: Trained classifier with a predict method.
    x_data: Features to predict.
    y_true: True labels.

    Returns:
    cm: Confusion matrix as a 2D array.
    """
    y_pred = classifier.predict(x_data)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true')

    if plot_cm:
        plt.matshow(cm)
        plt.colorbar()
        plt.show()
    return cm


def metrics_classifier(cm,  plot=False):
    """
    Compute accuracy, precision, recall, and F1-score from the confusion matrix.

    Parameters:
    cm: Confusion matrix as a 2D array.

    Returns:
    A dictionary with accuracy, precision, recall, and F1-score.
    """
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    tp_per_class = np.diag(cm)
    fp_per_class = np.sum(cm, axis=0) - tp_per_class
    fn_per_class = np.sum(cm, axis=1) - tp_per_class

    # Calcul des métriques par classe
    precision_per_class = np.divide(tp_per_class, tp_per_class + fp_per_class, 
                                    out=np.zeros_like(tp_per_class, dtype=float), 
                                    where=(tp_per_class + fp_per_class) != 0)
    
    recall_per_class = np.divide(tp_per_class, tp_per_class + fn_per_class, 
                                 out=np.zeros_like(tp_per_class, dtype=float), 
                                 where=(tp_per_class + fn_per_class) != 0)

    # 3. Moyenne "Macro" (moyenne simple des scores de chaque classe)
    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    
    if (precision + recall) > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0
    if plot:
        print(f"Accuracy: {100*accuracy:.2f}%")
        print(f"Precision: {100*precision:.2f}%")
        print(f"Recall: {100*recall:.2f}%")
        print(f"F1-score: {100*f1_score:.2f}%")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def compare_models(models_dict, x_test, y_test):
    """
    Compare different models and return a DataFrame of metrics.
    
    Parameters:
    models_dict: Dictionnary { "Name of the model": trained_model_instance }
    x_test: Test data
    y_test: Test labels
    """
    all_results = {}

    for name, model in models_dict.items():
        y_pred = model.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = metrics_classifier(cm, plot=False)
        
        all_results[name] = metrics

    df_results = pd.DataFrame(all_results).T
    
    df_results = df_results.sort_values(by='f1_score', ascending=False)
    
    return df_results