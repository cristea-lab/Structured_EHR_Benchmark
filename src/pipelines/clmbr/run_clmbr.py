
"""
CLMBR pipeline: produce per-patient vectors r_i from the OMOP event stream up to the anchor,
then fit a small classifier. If a CLMBR backend is not available, falls back to a simple
frequency embedding to keep the pipeline runnable.
"""
import argparse, os, sqlite3, pandas as pd, numpy as np, json
from tqdm import tqdm
from lightgbm import LGBMClassifier
from src.eval.metrics import compute_metrics
from src.utils.io import read_labels, attach_labels_table

def fetch_ordered_events(con, labels_table, agg):
    q = f"""
    WITH label_t AS (
        SELECT t.*
        FROM {labels_table} AS t
        JOIN (
            SELECT person_id, {agg}(datetime(prediction_time)) AS prediction_time
            FROM {labels_table}
            GROUP BY person_id
        ) m
        ON m.person_id = t.person_id
        AND datetime(t.prediction_time) = m.prediction_time
    ),
    ev AS (
        SELECT l.person_id, l.prediction_time, l.value AS label_value,
               co.condition_concept_id AS concept_id, co.condition_start_date AS event_date, 'condition' AS event_type
        FROM label_t l LEFT JOIN condition_occurrence co ON co.person_id = l.person_id
        WHERE co.condition_start_date <= l.prediction_time
        UNION ALL
        SELECT l.person_id, l.prediction_time, l.value AS label_value,
               de.drug_concept_id AS concept_id, de.drug_exposure_start_date AS event_date, 'drug' AS event_type
        FROM label_t l LEFT JOIN drug_exposure de ON de.person_id = l.person_id
        WHERE de.drug_exposure_start_date <= l.prediction_time
    )
    SELECT * FROM ev ORDER BY person_id, event_date;
    """
    return pd.read_sql_query(q, con, parse_dates=["prediction_time","event_date"])

def backend_frequency(df):
    # simple bag-of-events over concept_id
    grp = (df.groupby(["person_id","prediction_time","concept_id"])["event_type"]
             .size().unstack(fill_value=0))
    emb = grp.reset_index()
    y = df.drop_duplicates(["person_id","prediction_time"])[["person_id","prediction_time","label_value"]]
    Xy = emb.merge(y, on=["person_id","prediction_time"], how="left")
    return Xy

def main(args):
    task_folder = {"los":"guo_los","readmission":"readmission","pancreatic_cancer":"pancreatic_cancer","acute_mi":"acute_mi"}[args.task]
    labels = read_labels(args.assets_root, task_folder)
    con = sqlite3.connect(args.sqlite_path)
    labels_table, agg = attach_labels_table(con, labels, task_folder, args.visit_anchor)

    df = fetch_ordered_events(con, labels_table, agg)

    # For artifact review we default to the fallback. To use CLMBR, implement your encoder and set --backend clmbr.
    if args.backend == "clmbr":
        raise NotImplementedError("Please integrate your CLMBR encoder here or use --backend fallback.")
    else:
        Xy = backend_frequency(df)

    # Split by assets
    sp = pd.read_csv(os.path.join(args.assets_root, "splits", "person_id_map.csv"))
    sp = sp.rename(columns={c:c.lower() for c in sp.columns})
    Xy = Xy.merge(sp[["person_id","split"]], on="person_id", how="left")
    tr, va, te = Xy[Xy.split=="train"], Xy[Xy.split=="val"], Xy[Xy.split=="test"]

    # Train small classifier
    feat_cols = [c for c in Xy.columns if c not in ("person_id","prediction_time","label_value","split")]
    clf = LGBMClassifier(random_state=args.seed)
    clf.fit(tr[feat_cols], tr["label_value"])
    prob = clf.predict_proba(te[feat_cols])[:,1]
    metrics = compute_metrics(te["label_value"], prob)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"lightgbm": metrics}, f, indent=2)
    print("Results:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite_path", required=True)
    ap.add_argument("--assets_root", required=True)
    ap.add_argument("--task", choices=["los","readmission","pancreatic_cancer","acute_mi"], required=True)
    ap.add_argument("--visit_anchor", choices=["earliest","latest"], default="earliest")
    ap.add_argument("--backend", choices=["fallback","clmbr"], default="fallback")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
