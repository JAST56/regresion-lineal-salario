import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Mapeo ordinal de educacion
EDUCATION_ORDER = {
    "High School": 0,
    "Unknown": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4,
}

# Variables globales del modelo entrenado
trained_model = None
feature_names = None
job_title_encoding = None  # {job_title: mean_salary}
residual_std = None
model_metrics = None


def prepare_features(df, fit=True):
    """
    Transforma el DataFrame crudo en features numericas para el modelo.
    Si fit=True, calcula los encodings. Si fit=False, usa los existentes.
    """
    global job_title_encoding

    result = pd.DataFrame()

    # Numericas directas
    result["age"] = df["age"].astype(float)
    result["years_of_experience"] = df["years_of_experience"].astype(float)

    # Education Level -> ordinal
    result["education_ordinal"] = df["education_level"].map(EDUCATION_ORDER).fillna(1).astype(float)

    # Gender -> one-hot (drop first para evitar multicolinealidad)
    gender_dummies = pd.get_dummies(df["gender"], prefix="gender", drop_first=True)
    for col in gender_dummies.columns:
        result[col] = gender_dummies[col].astype(float)

    # Job Title -> target encoding
    if fit:
        job_title_encoding = df.groupby("job_title")["salary"].mean().to_dict()

    global_mean = df["salary"].mean() if fit else np.mean(list(job_title_encoding.values()))
    result["job_title_encoded"] = df["job_title"].map(job_title_encoding).fillna(global_mean)

    return result


def prepare_single_input(age, gender, education_level, job_title, years_of_experience):
    """Prepara un solo input para prediccion, usando los encodings entrenados."""
    row = pd.DataFrame([{
        "age": float(age),
        "gender": gender,
        "education_level": education_level,
        "job_title": job_title,
        "years_of_experience": float(years_of_experience),
        "salary": 0,  # dummy, no se usa
    }])

    features = prepare_features(row, fit=False)

    # Asegurar que las columnas coincidan con el modelo entrenado
    for col in feature_names:
        if col not in features.columns:
            features[col] = 0.0

    return features[feature_names]


def train_model(df):
    """Entrena el modelo de regresion multiple y calcula residuos."""
    global trained_model, feature_names, residual_std, model_metrics

    # Preparar features
    X_df = prepare_features(df, fit=True)
    feature_names = list(X_df.columns)
    X = X_df.values
    y = df["salary"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo
    trained_model = LinearRegression()
    trained_model.fit(X_train, y_train)

    # Predicciones en test
    y_pred_test = trained_model.predict(X_test)

    # Metricas en test
    r2 = float(r2_score(y_test, y_pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))

    # Cross-validation (5-fold)
    cv_scores = cross_val_score(trained_model, X, y, cv=5, scoring="r2")

    # Residuos del modelo completo (para Monte Carlo)
    y_pred_all = trained_model.predict(X)
    residuals = y - y_pred_all
    residual_std = float(np.std(residuals))

    # Feature importance (coeficiente * desviacion estandar de la feature)
    # Esto mide el impacto real de cada feature en la prediccion
    feature_stds = np.std(X, axis=0)
    coefs_impact = np.abs(trained_model.coef_) * feature_stds
    impact_sum = coefs_impact.sum()
    importance = (coefs_impact / impact_sum * 100) if impact_sum > 0 else coefs_impact

    feature_importance = {
        name: round(float(imp), 2) for name, imp in zip(feature_names, importance)
    }

    # Ecuacion del modelo
    terms = []
    for name, coef in zip(feature_names, trained_model.coef_):
        terms.append(f"{coef:.2f} x {name}")
    equation = "Salary = " + " + ".join(terms) + f" + {trained_model.intercept_:.2f}"

    model_metrics = {
        "r2": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "residual_std": round(residual_std, 2),
        "n_features": len(feature_names),
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "feature_importance": feature_importance,
        "feature_names": feature_names,
        "equation": equation,
        "coefficients": {
            name: round(float(c), 2) for name, c in zip(feature_names, trained_model.coef_)
        },
        "intercept": round(float(trained_model.intercept_), 2),
    }

    print(f"Modelo entrenado: RÂ²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")
    print(f"Cross-validation RÂ²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Desviacion estandar de residuos: {residual_std:.2f}")

    return model_metrics


def predict(age, gender, education_level, job_title, years_of_experience):
    """Predice un salario puntual."""
    X_input = prepare_single_input(age, gender, education_level, job_title, years_of_experience)
    prediction = float(trained_model.predict(X_input.values)[0])
    return prediction


def get_residual_std():
    return residual_std


def get_model_metrics():
    return model_metrics
