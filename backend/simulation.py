import numpy as np
from model import predict, get_residual_std


def run_monte_carlo(age, gender, education_level, job_title,
                    years_of_experience, n_simulations=10000, threshold=None):
    """
    Ejecuta simulacion Monte Carlo para predecir distribucion de salarios.

    Proceso:
    1. El modelo de regresion predice un salario base (y_hat)
    2. Los residuos del modelo siguen ~N(0, sigma_residuos)
    3. Para cada iteracion: salario_simulado = y_hat + N(0, sigma)
    4. Se calculan estadisticas descriptivas sobre los resultados
    """
    # Prediccion puntual del modelo
    predicted_salary = predict(age, gender, education_level, job_title, years_of_experience)

    # Desviacion estandar de los residuos del modelo
    sigma = get_residual_std()

    # Generar N salarios simulados: y_hat + ruido gaussiano
    np.random.seed(None)  # Seed aleatorio para cada simulacion
    noise = np.random.normal(0, sigma, size=n_simulations)
    simulated_salaries = predicted_salary + noise

    # Asegurar que no haya salarios negativos
    simulated_salaries = np.maximum(simulated_salaries, 0)

    # Estadisticas descriptivas
    sim_mean = float(np.mean(simulated_salaries))
    sim_median = float(np.median(simulated_salaries))
    sim_std = float(np.std(simulated_salaries))
    sim_min = float(np.min(simulated_salaries))
    sim_max = float(np.max(simulated_salaries))

    # Intervalos de confianza via percentiles
    ci_90 = [float(np.percentile(simulated_salaries, 5)),
             float(np.percentile(simulated_salaries, 95))]
    ci_95 = [float(np.percentile(simulated_salaries, 2.5)),
             float(np.percentile(simulated_salaries, 97.5))]

    # Skewness y Kurtosis
    if sim_std > 0:
        skewness = float(np.mean(((simulated_salaries - sim_mean) / sim_std) ** 3))
        kurtosis = float(np.mean(((simulated_salaries - sim_mean) / sim_std) ** 4))
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Probabilidad de superar umbral
    prob_above_threshold = 0.0
    if threshold is not None and threshold > 0:
        prob_above_threshold = float(np.mean(simulated_salaries > threshold))

    # Histograma (bins y conteos para el frontend)
    n_bins = 40
    counts, bin_edges = np.histogram(simulated_salaries, bins=n_bins)
    histogram = {
        "bins": [round(float(b), 2) for b in bin_edges[:-1]],
        "bin_edges": [round(float(b), 2) for b in bin_edges],
        "counts": [int(c) for c in counts],
    }

    return {
        "predicted_salary": round(predicted_salary, 2),
        "mean": round(sim_mean, 2),
        "median": round(sim_median, 2),
        "std": round(sim_std, 2),
        "min": round(sim_min, 2),
        "max": round(sim_max, 2),
        "ci_90": [round(ci_90[0], 2), round(ci_90[1], 2)],
        "ci_95": [round(ci_95[0], 2), round(ci_95[1], 2)],
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "prob_above_threshold": round(prob_above_threshold, 4),
        "histogram": histogram,
        "n_simulations": n_simulations,
    }


def run_comparison(profile_a, profile_b, n_simulations=10000, threshold=None):
    """
    Compara dos perfiles ejecutando Monte Carlo para cada uno.
    Retorna ambos resultados + analisis comparativo.
    """
    result_a = run_monte_carlo(
        profile_a["age"], profile_a["gender"], profile_a["education_level"],
        profile_a["job_title"], profile_a["years_of_experience"],
        n_simulations, threshold
    )

    result_b = run_monte_carlo(
        profile_b["age"], profile_b["gender"], profile_b["education_level"],
        profile_b["job_title"], profile_b["years_of_experience"],
        n_simulations, threshold
    )

    # Diferencia entre perfiles
    diff_mean = result_a["mean"] - result_b["mean"]
    diff_median = result_a["median"] - result_b["median"]

    comparison = {
        "profile_a": result_a,
        "profile_b": result_b,
        "difference": {
            "mean": round(diff_mean, 2),
            "median": round(diff_median, 2),
            "a_higher_probability": round(
                1.0 if diff_mean > 0 else 0.0 if diff_mean == 0 else 0.0, 2
            ),
        }
    }

    if threshold:
        comparison["difference"]["prob_above_threshold_a"] = result_a["prob_above_threshold"]
        comparison["difference"]["prob_above_threshold_b"] = result_b["prob_above_threshold"]

    return comparison
