
"""
Build a minimal SQLite database from raw OMOP CSVs sufficient to run the pipelines.

Expected CSVs under --omop_csv_dir (if present):
  - person.csv
  - concept.csv
  - concept_relationship.csv
  - concept_ancestor.csv
  - condition_occurrence.csv
  - drug_exposure.csv
"""
import argparse, os, pandas as pd, sqlite3
from tqdm import tqdm

def chunked_csv_to_sqlite(csv_path, con, table, chunksize=200_000):
    it = pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)
    first = True
    for chunk in it:
        chunk.to_sql(table, con, if_exists="replace" if first else "append", index=False)
        first = False

def main(args):
    os.makedirs(os.path.dirname(args.out_sqlite), exist_ok=True)
    con = sqlite3.connect(args.out_sqlite)

    wanted = [
        "person", "concept", "concept_relationship", "concept_ancestor",
        "condition_occurrence", "drug_exposure"
    ]
    for tbl in wanted:
        p = os.path.join(args.omop_csv_dir, f"{tbl}.csv")
        if os.path.exists(p):
            print(f"Ingesting {tbl} ...")
            chunked_csv_to_sqlite(p, con, tbl, chunksize=args.chunksize)
        else:
            print(f"Skipping {tbl} (not found)")

    # Indexes
    con.execute("CREATE INDEX IF NOT EXISTS idx_person ON person(person_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_concept ON concept(concept_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_concept_vocab ON concept(vocabulary_id, concept_code);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_conrel_1 ON concept_relationship(concept_id_1);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_conrel_2 ON concept_relationship(concept_id_2);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_conanc ON concept_ancestor(ancestor_concept_id, descendant_concept_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cond ON condition_occurrence(person_id, condition_start_date, condition_concept_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_drug ON drug_exposure(person_id, drug_exposure_start_date, drug_concept_id);")
    con.commit()
    con.close()
    print(f"SQLite DB written to {args.out_sqlite}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--omop_csv_dir", required=True)
    ap.add_argument("--out_sqlite", required=True)
    ap.add_argument("--chunksize", type=int, default=200000)
    args = ap.parse_args()
    main(args)
