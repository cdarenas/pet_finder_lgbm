import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Conexión a SQLite
conn = sqlite3.connect("resultados.db")

# Leer tabla resultados
df = pd.read_sql_query("SELECT * FROM resultados ORDER BY id ASC", conn)
conn.close()

if df.empty:
    print("No hay resultados en la base de datos.")
    exit()

# Mostrar tabla
print("\n=== Resultados disponibles ===")
print(df[["id", "fecha", "accuracy", "f1_macro", "kappa", "modelo", "config"]])

# --- Gráfico de evolución temporal ---
plt.figure(figsize=(10, 5))
plt.plot(df["id"], df["accuracy"], marker='o', label="Accuracy")
plt.plot(df["id"], df["f1_macro"], marker='s', label="F1 Macro")
plt.plot(df["id"], df["kappa"], marker='^', label="Kappa")
plt.xlabel("ID de Experimento")
plt.ylabel("Métrica")
plt.title("Evolución de Accuracy, F1 Macro y Kappa")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("evolucion_metricas.png")
plt.close()

# --- Ranking por Kappa (además del anterior por F1) ---
df_sorted_kappa = df.sort_values(by="kappa", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_sorted_kappa, x="kappa", y="modelo", palette="magma")
plt.xlabel("Kappa")
plt.ylabel("Modelo")
plt.title("Ranking de Experimentos por Kappa")
plt.tight_layout()
plt.savefig("ranking_kappa.png")
plt.close()

# --- Ranking por F1 Macro (original) ---
df_sorted_f1 = df.sort_values(by="f1_macro", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_sorted_f1, x="f1_macro", y="modelo", palette="viridis")
plt.xlabel("F1 Macro")
plt.ylabel("Modelo")
plt.title("Ranking de Experimentos por F1 Macro")
plt.tight_layout()
plt.savefig("ranking_f1_macro.png")
plt.close()

# --- Guardar CSV resumen ---
df.to_csv("resumen_experimentos.csv", index=False)
print("\n✅ Se generaron los gráficos y resumen de experimentos correctamente.")