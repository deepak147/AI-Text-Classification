import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_val, y_val, label_encoder):
    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    y_val_pred = model.predict(X_val)
    y_val_pred_labels = label_encoder.inverse_transform(np.argmax(y_val_pred, axis=1))
    y_val_true_labels = label_encoder.inverse_transform(np.argmax(y_val, axis=1))

    # Create a confusion matrix
    confusion_mat = confusion_matrix(y_val_true_labels, y_val_pred_labels)

    print("Confusion Matrix:")
    print(confusion_mat)

    df_cm = pd.DataFrame(
        confusion_mat,
        index=[i for i in ["AI", "AI, Owned", "AI-TextBook", "Student"]],
        columns=[i for i in ["AI", "AI, Owned", "AI-TextBook", "Student"]],
    )
    plt.figure(figsize=(7, 5))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
