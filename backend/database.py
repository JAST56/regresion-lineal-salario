import sqlite3
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Guardar la BD en la carpeta temp del usuario para evitar que
# Live Server de VS Code detecte cambios y recargue la pagina
DB_DIR = os.path.join(os.path.expanduser("~"), ".salary_simulation")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "salary_simulation.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS salary_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            gender TEXT,
            education_level TEXT,
            job_title TEXT,
            years_of_experience REAL,
            salary REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            gender TEXT,
            education_level TEXT,
            job_title TEXT,
            years_of_experience REAL,
            n_simulations INTEGER,
            predicted_salary REAL,
            sim_mean REAL,
            sim_median REAL,
            sim_std REAL,
            ci_90_lower REAL,
            ci_90_upper REAL,
            ci_95_lower REAL,
            ci_95_upper REAL,
            threshold REAL,
            prob_above_threshold REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE,
            r2_score REAL,
            rmse REAL,
            mae REAL,
            n_features INTEGER,
            n_samples INTEGER,
            residual_std REAL,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print(f"Base de datos inicializada: {DB_PATH}")


def load_csv_to_db(csv_path):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM salary_data")
    count = cursor.fetchone()[0]

    if count > 0:
        print(f"Base de datos ya tiene {count} registros. No se recarga el CSV.")
        conn.close()
        return count

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # Limpiar Education Level
    edu_map = {
        "Bachelor's Degree": "Bachelor's",
        "Master's Degree": "Master's",
        "phD": "PhD",
    }
    df["Education Level"] = df["Education Level"].replace(edu_map)

    # Limpiar NaN
    df["Gender"] = df["Gender"].fillna("Unknown")
    df["Education Level"] = df["Education Level"].fillna("Unknown")

    # Convertir tipos
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["Years of Experience"] = pd.to_numeric(df["Years of Experience"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Eliminar filas con nulos o salarios invalidos
    df = df.dropna(subset=["Salary", "Years of Experience", "Age"])
    df = df[df["Salary"] > 1000]

    for _, row in df.iterrows():
        cursor.execute(
            "INSERT INTO salary_data (age, gender, education_level, job_title, years_of_experience, salary) VALUES (?, ?, ?, ?, ?, ?)",
            (row["Age"], row["Gender"], row["Education Level"], row["Job Title"], row["Years of Experience"], row["Salary"])
        )

    conn.commit()
    total = len(df)
    print(f"CSV cargado a SQLite: {total} registros insertados")
    conn.close()
    return total


def get_all_data():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM salary_data", conn)
    conn.close()
    return df


def get_stats():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT job_title) as total_jobs,
            MIN(salary) as salary_min,
            MAX(salary) as salary_max,
            AVG(salary) as salary_mean,
            MIN(years_of_experience) as exp_min,
            MAX(years_of_experience) as exp_max,
            AVG(years_of_experience) as exp_mean,
            MIN(age) as age_min,
            MAX(age) as age_max,
            AVG(age) as age_mean
        FROM salary_data
    """)
    row = cursor.fetchone()
    conn.close()
    return dict(row)


def get_job_titles():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT job_title FROM salary_data ORDER BY job_title")
    titles = [row[0] for row in cursor.fetchall()]
    conn.close()
    return titles


def get_education_levels():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT education_level FROM salary_data ORDER BY education_level")
    levels = [row[0] for row in cursor.fetchall()]
    conn.close()
    return levels


def save_simulation(params, results):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO simulations
        (age, gender, education_level, job_title, years_of_experience,
         n_simulations, predicted_salary, sim_mean, sim_median, sim_std,
         ci_90_lower, ci_90_upper, ci_95_lower, ci_95_upper,
         threshold, prob_above_threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        params["age"], params["gender"], params["education_level"],
        params["job_title"], params["years_of_experience"],
        params["n_simulations"], results["predicted_salary"],
        results["mean"], results["median"], results["std"],
        results["ci_90"][0], results["ci_90"][1],
        results["ci_95"][0], results["ci_95"][1],
        params.get("threshold", 0), results.get("prob_above_threshold", 0)
    ))
    conn.commit()
    conn.close()


def get_simulation_history(limit=50):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM simulations ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def save_model_metrics(model_name, metrics):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO model_metrics
        (model_name, r2_score, rmse, mae, n_features, n_samples, residual_std)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name, metrics["r2"], metrics["rmse"], metrics["mae"],
        metrics["n_features"], metrics["n_samples"], metrics["residual_std"]
    ))
    conn.commit()
    conn.close()
