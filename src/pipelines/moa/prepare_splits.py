
"""
Given Qwen summaries CSV + person->row mapping pickle, create train/val/test CSVs aligned to EHRSHOT splits.
"""
import argparse, os, pickle, pandas as pd

TASK_MAP = {"los":"guo_los", "readmission":"readmission", "pancreatic_cancer":"pancreatic_cancer", "acute_mi":"acute_mi"}

def main(args):
    task_folder = TASK_MAP[args.task]
    csv_in  = os.path.join(args.intermediate_root, f"df_event_MoA_qwen_{task_folder}_{args.run_tag}.csv")
    pkl_in  = os.path.join(args.intermediate_root, f"dic_pid_row_llm_MoA_qwen_{task_folder}_{args.run_tag}.pickle")
    splits  = os.path.join(args.assets_root, "splits", "person_id_map.csv")

    df = pd.read_csv(csv_in)
    with open(pkl_in, "rb") as f:
        pid2row = pickle.load(f)

    sp = pd.read_csv(splits)
    sp = sp.rename(columns={c:c.lower() for c in sp.columns})
    assert "person_id" in sp.columns and "split" in sp.columns

    rows = pd.Series(pid2row, name="row").reset_index().rename(columns={"index":"person_id"})
    rows = rows.merge(sp[["person_id","split"]], on="person_id", how="left")

    for split in ["train","val","test"]:
        idxs = rows[rows["split"]==split]["row"].tolist()
        df_split = df.iloc[idxs].reset_index(drop=True)
        out = os.path.join(args.intermediate_root, f"{args.task}_{args.run_tag}_{split}.csv")
        df_split.to_csv(out, index=False)
        print("Wrote:", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets_root", required=True)
    ap.add_argument("--intermediate_root", default="./outputs/intermediate")
    ap.add_argument("--task", choices=list(TASK_MAP.keys()), required=True)
    ap.add_argument("--run_tag", default="seed2")
    args = ap.parse_args()
    main(args)
