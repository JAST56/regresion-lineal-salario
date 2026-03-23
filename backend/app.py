from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from database import (
    init_db, load_csv_to_db, get_all_data, get_stats,
    get_job_titles, get_education_levels,
    save_simulation, get_simulation_history, save_model_metrics
)
from model import train_model, get_model_metrics
from simulation import run_monte_carlo, run_comparison

app = Flask(__name__)
CORS(app)

# =========================
# INICIALIZACION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Salary_Data.csv")

print("Inicializando aplicacion...")
init_db()
n_records = load_csv_to_db(DATASET_PATH)
df = get_all_data()
metrics = train_model(df)
save_model_metrics("multiple_regression", metrics)
print(f"Aplicacion lista. {len(df)} registros en BD.")


# =========================
# ENDPOINTS
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API de Simulacion Monte Carlo - Salarios",
        "version": "2.0",
        "endpoints": ["/stats", "/job-titles", "/education-levels",
                      "/model-info", "/simulate", "/compare", "/history"]
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Estadisticas del dataset via SQL queries."""
    data = get_stats()
    return jsonify({
        "total_records": data["total_records"],
        "total_jobs": data["total_jobs"],
        "salary_min": round(data["salary_min"], 2),
        "salary_max": round(data["salary_max"], 2),
        "salary_mean": round(data["salary_mean"], 2),
        "exp_min": round(data["exp_min"], 2),
        "exp_max": round(data["exp_max"], 2),
        "exp_mean": round(data["exp_mean"], 2),
        "age_min": round(data["age_min"], 2),
        "age_max": round(data["age_max"], 2),
        "age_mean": round(data["age_mean"], 2),
    })


@app.route("/job-titles", methods=["GET"])
def job_titles():
    """Lista de profesiones desde la BD."""
    titles = get_job_titles()
    return jsonify(titles)


@app.route("/education-levels", methods=["GET"])
def education_levels():
    """Niveles de educacion validos."""
    levels = get_education_levels()
    return jsonify(levels)


@app.route("/model-info", methods=["GET"])
def model_info():
    """Informacion y metricas del modelo entrenado."""
    m = get_model_metrics()
    return jsonify({
        "model_name": "Regresion Lineal Multiple",
        "r2_score": m["r2"],
        "rmse": m["rmse"],
        "mae": m["mae"],
        "residual_std": m["residual_std"],
        "n_features": m["n_features"],
        "n_samples": m["n_samples"],
        "n_train": m["n_train"],
        "n_test": m["n_test"],
        "cv_r2_mean": m["cv_r2_mean"],
        "cv_r2_std": m["cv_r2_std"],
        "feature_importance": m["feature_importance"],
        "feature_names": m["feature_names"],
        "equation": m["equation"],
        "coefficients": m["coefficients"],
        "intercept": m["intercept"],
    })


@app.route("/simulate", methods=["POST"])
def simulate():
    """Ejecutar simulacion Monte Carlo."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos JSON"}), 400

        # Validar campos requeridos
        required = ["age", "gender", "education_level", "job_title", "years_of_experience"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Campo requerido: {field}"}), 400

        age = float(data["age"])
        gender = data["gender"]
        education_level = data["education_level"]
        job_title = data["job_title"]
        years_of_experience = float(data["years_of_experience"])
        n_simulations = int(data.get("n_simulations", 10000))
        threshold = data.get("threshold")

        if threshold is not None:
            threshold = float(threshold)

        # Limitar simulaciones
        n_simulations = max(1000, min(n_simulations, 100000))

        # Ejecutar simulacion
        results = run_monte_carlo(
            age, gender, education_level, job_title,
            years_of_experience, n_simulations, threshold
        )

        # Guardar en BD
        params = {
            "age": age,
            "gender": gender,
            "education_level": education_level,
            "job_title": job_title,
            "years_of_experience": years_of_experience,
            "n_simulations": n_simulations,
            "threshold": threshold or 0,
        }
        save_simulation(params, results)

        # Incluir metricas del modelo
        m = get_model_metrics()

        return jsonify({
            "input": params,
            "simulation": results,
            "model_metrics": {
                "r2": m["r2"],
                "rmse": m["rmse"],
                "residual_std": m["residual_std"],
            }
        })

    except ValueError as e:
        return jsonify({"error": f"Valor invalido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """Comparar dos perfiles con simulacion Monte Carlo."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos JSON"}), 400

        profile_a = data.get("profile_a")
        profile_b = data.get("profile_b")

        if not profile_a or not profile_b:
            return jsonify({"error": "Se requieren profile_a y profile_b"}), 400

        n_simulations = int(data.get("n_simulations", 10000))
        threshold = data.get("threshold")
        if threshold is not None:
            threshold = float(threshold)

        n_simulations = max(1000, min(n_simulations, 100000))

        results = run_comparison(profile_a, profile_b, n_simulations, threshold)

        return jsonify(results)

    except ValueError as e:
        return jsonify({"error": f"Valor invalido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def history():
    """Historial de simulaciones desde la BD."""
    limit = request.args.get("limit", 20, type=int)
    limit = max(1, min(limit, 100))
    rows = get_simulation_history(limit)
    return jsonify(rows)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
