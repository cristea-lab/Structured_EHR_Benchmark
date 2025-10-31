
import os, sqlite3, pandas as pd

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_labels(assets_root: str, task_folder: str):
    p = os.path.join(assets_root, "benchmark", task_folder, "labeled_patients.csv")
    df = pd.read_csv(p)
    return df.rename(columns={"patient_id": "person_id"})

def attach_labels_table(con: sqlite3.Connection, df_labels, task_folder: str, anchor: str):
    table = f"labels_{task_folder}"
    df = df_labels.copy()
    df["prediction_time"] = pd.to_datetime(df["prediction_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_sql(table, con, if_exists="replace", index=False)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table} ON {table}(person_id, prediction_time);")
    con.commit()
    agg = "MIN" if anchor == "earliest" else "MAX"
    return table, agg
