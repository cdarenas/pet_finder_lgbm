# plot_confusion.py
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Leer argumentos
y_true_path = sys.argv[1]
y_pred_path = sys.argv[2]
output_path = sys.argv[3]

# Cargar etiquetas reales y predichas
y_true = pd.read_csv(y_true_path, header=None)[0]
y_pred = pd.read_csv(y_pred_path, header=None)[0]

# Calcular matriz de confusión
cm = confusion_matrix(y_true, y_pred)
labels = list(range(cm.shape[0]))

# Graficar
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.savefig(output_path)
