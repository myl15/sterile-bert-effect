import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_pos_metrics(eval_pred):
    """
    Compute accuracy and macro/weighted F1 for token classification.
    Used as compute_metrics callback in HuggingFace Trainer.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # Flatten and filter out padding tokens (label == -100)
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)

    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro",
                        zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted",
                           zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
