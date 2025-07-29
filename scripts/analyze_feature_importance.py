import os
import sys
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer argumentos desde línea de comandos
fold_dir = sys.argv[1]
output_img = sys.argv[2]

# Nombres reales de las features, en el mismo orden que se usaron en el archivo .txt
feature_names = [
    'Type', 'Age', 'Breed1', 'Breed2', 'Gender',
    'MaturitySize', 'FurLength', 'Health', 'Quantity', 'State',
    'CareLevel', 'ColorCount', 'IsMixedBreed', 'HasPhotoVideo', 'TotalMedia', 'FeeBin'
]

# Mapeo seguro: Column_0 → nombre de variable
column_map = {f"Column_{i}": name for i, name in enumerate(feature_names)}

importance_df = pd.DataFrame()

for fold in range(5):
    model_file = os.path.join(fold_dir, f"model_fold_{fold}.txt")
    booster = lgb.Booster(model_file=model_file)

    gain = booster.feature_importance(importance_type='gain')
    model_features = booster.feature_name()  # Ej: ['Column_1', 'Column_3', ...]

    # Mapear a nombres reales
    mapped_features = [column_map.get(name, name) for name in model_features]

    df = pd.DataFrame({
        'feature': mapped_features,
        'importance_gain': gain
    })
    df['fold'] = fold
    importance_df = pd.concat([importance_df, df], axis=0)

# Promediar importancia por variable
mean_importance = importance_df.groupby('feature')['importance_gain'].mean().reset_index()
mean_importance = mean_importance.sort_values(by='importance_gain', ascending=False)

# Gráfico
plt.figure(figsize=(12, 7))
sns.barplot(data=mean_importance, y='feature', x='importance_gain', palette='viridis')
plt.title("Importancia promedio de variables (ganancia)")
plt.tight_layout()
plt.savefig(output_img)
