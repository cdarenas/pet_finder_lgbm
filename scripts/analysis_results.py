import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# --- Conexión a la base de datos SQLite ---
conn = sqlite3.connect("resultados.db")

# --- Leer la tabla de resultados ---
df = pd.read_sql_query("SELECT id, fecha, accuracy, f1_macro, kappa, modelo FROM resultados", conn)
conn.close()

# --- Asegurar orden por ID (folds) ---
df = df.sort_values("id").reset_index(drop=True)

# --- Mostrar resumen tabular ---
print("\nResumen de resultados por fold:")
print(df)

# --- Gráfico de barras: Accuracy, F1 Macro y Kappa ---
plt.figure(figsize=(12, 6))
bar_width = 0.25
x = range(len(df))

plt.bar([i - bar_width for i in x], df["accuracy"], width=bar_width, label="Accuracy")
plt.bar(x, df["f1_macro"], width=bar_width, label="F1 Macro")
plt.bar([i + bar_width for i in x], df["kappa"], width=bar_width, label="Kappa")

plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Accuracy, F1 Macro y Kappa por Fold")
plt.xticks(ticks=x, labels=df["id"])
plt.legend()
plt.tight_layout()
plt.savefig("metricas_por_fold.png")
plt.show()