"""Chargement et nettoyage du dataset Telco Customer Churn."""
import pandas as pd


def charger(chemin="../data/raw/Telco-Customer-Churn.csv"):
    df = pd.read_csv(chemin)

    # 11 valeurs vides dans TotalCharges : clients à tenure = 0, pas encore facturés
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Cible en 0/1
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Identifiant sans valeur prédictive
    return df.drop(columns=["customerID"])
    