from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np


import csv
import os

def metrics(answers, gold_labels, fallacy_list):

    fallacy_list_lower = [f.lower() for f in fallacy_list]
    predicted_labels = []

    for answer in answers:
        answer = answer.strip().lower()
        found_label = None
        for label, l_label in zip(fallacy_list, fallacy_list_lower):
            if l_label in answer:
                found_label = label
                break
        predicted_labels.append(found_label if found_label else answer)

    accuracy = accuracy_score(gold_labels, predicted_labels)
    micro_f1 = f1_score(gold_labels, predicted_labels, average='micro',  zero_division=0, labels=fallacy_list)
    f1 = f1_score(gold_labels, predicted_labels, average='macro',  zero_division=0, labels=fallacy_list)
    precision = precision_score(gold_labels, predicted_labels, average='macro', zero_division = 0, labels=fallacy_list)
    recall = recall_score(gold_labels, predicted_labels, average='macro', zero_division = 0, labels=fallacy_list)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Macro Precision: {precision:.4f}"), 
    print(f"Macro Recall: {recall:.4f}")

    return accuracy, micro_f1, f1, precision, recall


def classwise_eval(y_true, y_pred, path, fallacy_list):
    label_to_id = {label: i for i, label in enumerate(fallacy_list)}
    unknown_class = len(fallacy_list)

    fallacy_list_lower = [f.lower() for f in fallacy_list]
    proc_y_pred = []

    for answer in y_pred:
        answer = answer.strip().lower()
        found_label = None
        for label, l_label in zip(fallacy_list, fallacy_list_lower):
            if l_label in answer:
                found_label = label
                break
        proc_y_pred.append(found_label if found_label else answer)

    y_true_ids = [label_to_id.get(label, unknown_class) for label in y_true]
    y_pred_ids = [label_to_id.get(label, unknown_class) for label in proc_y_pred]

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Fallacy', 'Precision', 'Recall', 'F1-Score'])

        for c in range(unknown_class + 1):  # include unknown class
            y_true_c = np.where(np.array(y_true_ids) == c, 1, 0)
            y_pred_c = np.where(np.array(y_pred_ids) == c, 1, 0)

            precision = precision_score(y_true_c, y_pred_c, zero_division=0)
            recall = recall_score(y_true_c, y_pred_c, zero_division=0)
            f1 = f1_score(y_true_c, y_pred_c, zero_division=0)

            label_name = fallacy_list[c] if c < unknown_class else 'Unknown'
            writer.writerow([c, label_name, precision, recall, f1])


def log_metrics_to_csv(csv_file, gold_labels, predicted_labels, accuracy, micro_f1, macro_f1, recall, precision, prompting):

    fieldnames = ["prompting", "accuracy", "micro_f1", "macro_f1", "recall", "precision"]
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "prompting": prompting,
            "accuracy": accuracy,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "recall": recall,
            "precision": precision
        })


    fallacy_list = list(set(gold_labels))
    accuracy = accuracy_score(gold_labels, predicted_labels)
    micro_f1 = f1_score(gold_labels, predicted_labels, average='micro',  zero_division=0, labels=fallacy_list)
    f1 = f1_score(gold_labels, predicted_labels, average='macro',  zero_division=0, labels=fallacy_list)
    precision = precision_score(gold_labels, predicted_labels, average='macro', zero_division = 0, labels=fallacy_list)
    recall = recall_score(gold_labels, predicted_labels, average='macro', zero_division = 0, labels=fallacy_list)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
