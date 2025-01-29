import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Charger les fichiers Excel
file_t1 = "resultats-par-niveau-dpt-t1-france-entiere.xlsx"
file_t2 = "resultats-par-niveau-dpt-t2-france-entiere.xlsx"
file_socio = "données_socio_economique.xlsx"

xls_t1 = pd.ExcelFile(file_t1)
xls_t2 = pd.ExcelFile(file_t2)
xls_socio = pd.ExcelFile(file_socio)

df_t1 = xls_t1.parse(xls_t1.sheet_names[0])
df_t2 = xls_t2.parse(xls_t2.sheet_names[0])
df_socio = xls_socio.parse(xls_socio.sheet_names[0])

# Nettoyage et fusion des données
df_socio_clean = df_socio.rename(columns={
    "code": "Code du département",
    "departement": "Libellé du département",
    "taux de pauvereté": "Taux de pauvreté",
    "taux emploi": "Taux d'emploi",
    "taux de chômage": "Taux de chômage"
})
df_socio_clean["Code du département"] = df_socio_clean["Code du département"].astype(str).str.zfill(2)

df_t1_clean = df_t1[["Code du département", "Libellé du département", "Inscrits", "Abstentions", "% Abs/Ins",
                      "Votants", "% Vot/Ins", "Blancs", "% Blancs/Ins", "Nuls", "% Nuls/Ins"]]

df_t2_clean = df_t2[["Code du département", "Libellé du département", "Inscrits", "Abstentions", "% Abs/Ins",
                      "Votants", "% Vot/Ins", "Blancs", "% Blancs/Ins", "Nuls", "% Nuls/Ins",
                      "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp", "Nom.1", "Voix.1", "% Voix/Ins.1", "% Voix/Exp.1"]].rename(columns={"Prénom": "Candidat 1", "Voix": "Voix Candidat 1",
                                       "% Voix/Ins": "% Voix/Ins Candidat 1", "% Voix/Exp": "% Voix/Exp Candidat 1",
                                       "Nom.1": "Candidat 2", "Voix.1": "Voix Candidat 2",
                                       "% Voix/Ins.1": "% Voix/Ins Candidat 2", "% Voix/Exp.1": "% Voix/Exp Candidat 2"})

df_final = df_t1_clean.merge(df_t2_clean, on=["Code du département", "Libellé du département"], how="left")
df_final = df_final.merge(df_socio_clean, on=["Code du département", "Libellé du département"], how="left")

# Conversion des colonnes numériques
features = ["Taux de pauvreté", "Taux d'emploi", "Taux de chômage", "% Abs/Ins_x", "% Vot/Ins_x"]
for col in features:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
# Analyse de corrélation
plt.figure(figsize=(10, 6))
sns.heatmap(df_final[["% Abs/Ins_x", "% Voix/Ins Candidat 1", "% Voix/Ins Candidat 2",
                      "Taux de pauvreté", "Taux d'emploi", "Taux de chômage"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation entre les Variables")
plt.show()
# Graphiques
## Graphique en barres - Taux de participation par département
# Graphiques en une seule figure
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Graphique en barres - Taux de participation par département
df_final.sort_values('% Vot/Ins_x', ascending=False).head(10).plot.bar(
    x='Libellé du département', y='% Vot/Ins_x', ax=axes[0], legend=False, color='blue')
axes[0].set_title("Taux de Participation par Département")
axes[0].set_ylabel("% Participation")
axes[0].tick_params(axis='x', rotation=45)

# Graphique en barres - Taux de pauvreté par département
df_final.sort_values('Taux de pauvreté', ascending=False).head(10).plot.bar(
    x='Libellé du département', y='Taux de pauvreté', ax=axes[1], legend=False, color='red')
axes[1].set_title("Taux de Pauvreté par Département")
axes[1].set_ylabel("% Pauvreté")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

## Nuages de points - Corrélation entre taux de pauvreté et vote
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_final['Taux de pauvreté'], y=df_final['% Voix/Ins Candidat 2'])
plt.xlabel("Taux de Pauvreté")
plt.ylabel("% Vote Le Pen")
plt.title("Corrélation entre le taux de pauvreté et le vote Le Pen")
plt.show()


## Graphique linéaire - Taux de participation par département
df_final.sort_values('% Vot/Ins_x',ascending=False).head(10).plot.bar(x='Libellé du département', y='Taux de pauvreté', legend=False)
plt.title("Évolution du taux de participation par département")
plt.ylabel("% Participation")
plt.xticks(rotation=90)
plt.show()



# Entraînement et évaluation des modèles
## Sélection des variables cibles
target_macron = "% Voix/Ins Candidat 1"
target_lepen = "% Voix/Ins Candidat 2"
df_model = df_final.dropna(subset=features + [target_macron, target_lepen])
X = df_model[features]
y_macron = df_model[target_macron]
y_lepen = df_model[target_lepen]

## Division en ensembles d'entraînement et de test
X_train, X_test, y_macron_train, y_macron_test, y_lepen_train, y_lepen_test = train_test_split(
    X, y_macron, y_lepen, test_size=0.2, random_state=42)

## Entraînement des modèles
model_rf_macron = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf_lepen = RandomForestRegressor(n_estimators=100, random_state=42)
model_xgb_macron = XGBRegressor(n_estimators=100, random_state=42)
model_xgb_lepen = XGBRegressor(n_estimators=100, random_state=42)

model_rf_macron.fit(X_train, y_macron_train)
model_rf_lepen.fit(X_train, y_lepen_train)
model_xgb_macron.fit(X_train, y_macron_train)
model_xgb_lepen.fit(X_train, y_lepen_train)

## Prédictions
y_macron_pred_rf = model_rf_macron.predict(X_test)
y_lepen_pred_rf = model_rf_lepen.predict(X_test)
y_macron_pred_xgb = model_xgb_macron.predict(X_test)
y_lepen_pred_xgb = model_xgb_lepen.predict(X_test)

## Évaluation des modèles
mae_macron_rf = mean_absolute_error(y_macron_test, y_macron_pred_rf)
r2_macron_rf = r2_score(y_macron_test, y_macron_pred_rf)
mae_lepen_rf = mean_absolute_error(y_lepen_test, y_lepen_pred_rf)
r2_lepen_rf = r2_score(y_lepen_test, y_lepen_pred_rf)

mae_macron_xgb = mean_absolute_error(y_macron_test, y_macron_pred_xgb)
r2_macron_xgb = r2_score(y_macron_test, y_macron_pred_xgb)
mae_lepen_xgb = mean_absolute_error(y_lepen_test, y_lepen_pred_xgb)
r2_lepen_xgb = r2_score(y_lepen_test, y_lepen_pred_xgb)

print("Random Forest - Macron: MAE =", mae_macron_rf, "R² =", r2_macron_rf)
print("Random Forest - Le Pen: MAE =", mae_lepen_rf, "R² =", r2_lepen_rf)
print("XGBoost - Macron: MAE =", mae_macron_xgb, "R² =", r2_macron_xgb)
print("XGBoost - Le Pen: MAE =", mae_lepen_xgb, "R² =", r2_lepen_xgb)
# Récupération des noms de départements correspondants
departements_test = df_model.loc[X_test.index, "Libellé du département"]

# Visualisation des prédictions
plt.figure(figsize=(12, 6))
plt.plot(departements_test, y_macron_pred_rf, label="Prédictions Macron (RF)", marker='o')
plt.plot(departements_test, y_lepen_pred_rf, label="Prédictions Le Pen (RF)", marker='s')
plt.xticks(rotation=90)
plt.ylabel("% Voix/Ins")
plt.title("Prédictions des résultats électoraux par département (Random Forest)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(departements_test, y_macron_pred_xgb, label="Prédictions Macron (XGB)", marker='o')
plt.plot(departements_test, y_lepen_pred_xgb, label="Prédictions Le Pen (XGB)", marker='s')
plt.xticks(rotation=90)
plt.ylabel("% Voix/Ins")
plt.title("Prédictions des résultats électoraux par département (XGBoost)")
plt.legend()
plt.show()