import fasttext
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = fasttext.load_model("model/fasttext_model.ftz")

y_true = []
y_pred = []

with open("data/test.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        label = parts[0].replace("__label__", "")
        text = parts[1]
        pred_label, _ = model.predict(text)
        pred_label = pred_label[0].replace("__label__", "")
        y_true.append(label)
        y_pred.append(pred_label)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))


cm = confusion_matrix(y_true, y_pred)

labels = sorted(list(set(y_true)))

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,              
    fmt='d',                
    cmap='Blues',            
    xticklabels=labels,
    yticklabels=labels,
    cbar=True
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()