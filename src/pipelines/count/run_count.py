
"""
Count-based pipeline with time binning (recent/history) and LightGBM/TabPFN classifiers.
"""
import argparse, os, sqlite3, pandas as pd, numpy as np, json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from src.eval.metrics import compute_metrics
from src.utils.io import ensure_dir, read_labels, attach_labels_table

TASK_MAP = {
    "los": "guo_los",
    "readmission": "readmission",
    "pancreatic_cancer": "pancreatic_cancer",
    "acute_mi": "acute_mi",
}

def fetch_events(con, labels_table, agg, days_recent=365):
    query = f"""
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
        SELECT
            l.person_id, l.prediction_time, l.value AS label_value,
            co.condition_concept_id AS concept_id, cc.concept_code, cc.vocabulary_id,
            co.condition_start_date AS event_date,
            'condition' AS event_type
        FROM label_t l
        LEFT JOIN condition_occurrence co ON co.person_id = l.person_id
        LEFT JOIN concept cc ON cc.concept_id = co.condition_concept_id
        WHERE co.condition_start_date <= l.prediction_time

        UNION ALL
        SELECT
            l.person_id, l.prediction_time, l.value AS label_value,
            de.drug_concept_id AS concept_id, cc.concept_code, cc.vocabulary_id,
            de.drug_exposure_start_date AS event_date,
            'drug' AS event_type
        FROM label_t l
        LEFT JOIN drug_exposure de ON de.person_id = l.person_id
        LEFT JOIN concept cc ON cc.concept_id = de.drug_concept_id
        WHERE de.drug_exposure_start_date <= l.prediction_time
    )
    SELECT * FROM ev
    ORDER BY person_id, event_date;
    """
    df = pd.read_sql_query(query, con, parse_dates=["prediction_time", "event_date"])
    # compute recency
    df["days_before_pred"] = (pd.to_datetime(df["prediction_time"]) - pd.to_datetime(df["event_date"])).dt.days
    df["bin"] = np.where(df["days_before_pred"] <= days_recent, "recent", "history")
    return df

def map_condition_to_icd10(con):
    """Return a dict concept_id -> ICD10 category (3-char) when possible."""
    # Try to map SNOMED etc. to ICD10 via concept_relationship (to an ICD10 concept)
    q = """
    SELECT cr.concept_id_1 AS src_id, c2.vocabulary_id AS vocab2, c2.concept_code AS code2
    FROM concept_relationship cr
    JOIN concept c2 ON c2.concept_id = cr.concept_id_2
    WHERE c2.vocabulary_id LIKE 'ICD10%'
    """
    try:
        df = pd.read_sql_query(q, con)
        df["icd10_cat"] = df["code2"].str.replace(".", "", regex=False).str.slice(0, 3)
        return dict(zip(df["src_id"], df["icd10_cat"]))
    except Exception:
        return {}

def map_drug_to_atc(con, level=2):
    """Return a dict concept_id -> ATC prefix (level 2 = 3 chars)."""
    q = """
    SELECT cr.concept_id_1 AS src_id, c2.vocabulary_id AS vocab2, c2.concept_code AS code2
    FROM concept_relationship cr
    JOIN concept c2 ON c2.concept_id = cr.concept_id_2
    WHERE c2.vocabulary_id LIKE 'ATC%'
    """
    try:
        df = pd.read_sql_query(q, con)
        n = 1 if level == 1 else (3 if level == 2 else 4)
        df["atc_prefix"] = df["code2"].str.slice(0, n)
        return dict(zip(df["src_id"], df["atc_prefix"]))
    except Exception:
        return {}

def build_features(df, con):
    # Build maps
    cond2icd = map_condition_to_icd10(con)
    drug2atc = map_drug_to_atc(con, level=2)

    # Apply rollups
    dfc = df.copy()
    dfc["cond_icd10"] = np.where(
        dfc["event_type"] == "condition",
        dfc["concept_id"].map(cond2icd).fillna(dfc["concept_code"].str.replace(".", "", regex=False).str.slice(0, 3)),
        None
    )
    dfc["drug_atc"] = np.where(
        dfc["event_type"] == "drug",
        dfc["concept_id"].map(drug2atc).fillna(dfc["concept_code"].astype(str).str.slice(0, 3)),
        None
    )
    # Construct group keys
    dfc["group"] = np.where(
        dfc["event_type"] == "condition",
        "ICD10_" + dfc["cond_icd10"].fillna("UNK"),
        "ATC_" + dfc["drug_atc"].fillna("UNK")
    )

    # Count matrix per (person_id, prediction_time) × (bin, group)
    dfc["one"] = 1
    piv = (dfc
           .groupby(["person_id", "prediction_time", "bin", "group"])["one"]
           .sum()
           .unstack(["bin", "group"])
           .fillna(0.0))

    # Flatten columns to strings
    piv.columns = [f"{b}__{g}" for b, g in piv.columns]
    piv = piv.reset_index()

    # Labels (one per (person, anchor))
    y = (dfc.drop_duplicates(["person_id", "prediction_time"])
            [["person_id", "prediction_time", "label_value"]]
            .set_index(["person_id", "prediction_time"]))
    Xy = piv.set_index(["person_id", "prediction_time"]).join(y, how="left").reset_index()
    return Xy

def split_by_assets(Xy, assets_root, task_folder):
    # Expect splits/person_id_map.csv with columns: person_id, split
    sp = pd.read_csv(os.path.join(assets_root, "splits", "person_id_map.csv"))
    sp = sp.rename(columns={col: col.lower() for col in sp.columns})
    assert "person_id" in sp.columns and "split" in sp.columns, "person_id_map.csv must have person_id, split"

    Xy = Xy.merge(sp[["person_id", "split"]], on="person_id", how="left")
    tr = Xy[Xy["split"] == "train"]
    va = Xy[Xy["split"] == "val"]
    te = Xy[Xy["split"] == "test"]
    return tr, va, te

def train_eval_models(tr, va, te, out_dir, models="both", seed=42):
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = [c for c in tr.columns if c not in ("person_id","prediction_time","label_value","split")]
    ytr, yva, yte = tr["label_value"].values, va["label_value"].values, te["label_value"].values
    Xtr, Xva, Xte = tr[feature_cols].values, va[feature_cols].values, te[feature_cols].values

    results = {}

    if models in ("both","lightgbm"):
        clf_lgb = LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=0.03,
            num_leaves=64,
            feature_fraction=0.8,
            lambda_l1=0.01, lambda_l2=0.01,
            random_state=seed,
            n_estimators=2000
        )
        clf_lgb.fit(Xtr, ytr)
        prob = clf_lgb.predict_proba(Xte)[:,1]
        results["lightgbm"] = compute_metrics(yte, prob)

    if models in ("both","tabpfn"):
        clf_pf = TabPFNClassifier(fit_mode='low_memory', memory_saving_mode=True, ignore_pretraining_limits=True)
        clf_pf.fit(Xtr, ytr)
        prob = clf_pf.predict_proba(Xte)[:,1]
        results["tabpfn"] = compute_metrics(yte, prob)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Results:", results)

def main(args):
    task_folder = TASK_MAP[args.task]
    labels = read_labels(args.assets_root, task_folder)

    con = sqlite3.connect(args.sqlite_path)
    labels_table, agg = attach_labels_table(con, labels, task_folder, args.visit_anchor)

    df = fetch_events(con, labels_table, agg, days_recent=args.days_recent)
    Xy = build_features(df, con)
    tr, va, te = split_by_assets(Xy, args.assets_root, task_folder)
    train_eval_models(tr, va, te, args.out_dir, models=args.models, seed=args.seed)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite_path", required=True)
    ap.add_argument("--assets_root", required=True)
    ap.add_argument("--task", choices=list(TASK_MAP.keys()), required=True)
    ap.add_argument("--visit_anchor", choices=["earliest","latest"], default="earliest")
    ap.add_argument("--days_recent", type=int, default=365)
    ap.add_argument("--models", choices=["lightgbm","tabpfn","both"], default="both")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
