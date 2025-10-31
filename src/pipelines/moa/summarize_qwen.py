
"""
Summarize structured events with Qwen and produce JSON-like summaries per patient anchor.
See manuscript Section: Mixture-of-Agents.
"""
import argparse, os, sqlite3, pickle, pandas as pd, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_MAP = {
    "los": ("guo_los", "Long Length of Stay"),
    "readmission": ("readmission", "30-day Readmission"),
    "pancreatic_cancer": ("pancreatic_cancer", "Pancreatic Cancer"),
    "acute_mi": ("acute_mi", "Acute Myocardial Infarction"),
}

def get_attn_impl():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"

def build_query(task_table: str, anchor_agg: str):
    return f"""
        WITH label_t AS (
            SELECT t.*
            FROM {task_table} AS t
            JOIN (
                SELECT person_id, {anchor_agg}(datetime(prediction_time)) AS prediction_time
                FROM {task_table}
                GROUP BY person_id
            ) m
            ON m.person_id = t.person_id
            AND datetime(t.prediction_time) = m.prediction_time
        ),  
        t1 AS
        (
            SELECT DISTINCT l.person_id, l.prediction_time, 
            co.condition_concept_id AS concept_id, cc.concept_code, cc.concept_name, cc.vocabulary_id,
            co.condition_start_date AS event_date,
            'condition_occurrence' AS event_type,
            CAST(strftime('%Y', co.condition_start_date) AS INTEGER) - p.year_of_birth AS age_at_event,
            CAST(strftime('%Y', l.prediction_time) AS INTEGER) - p.year_of_birth AS age_at_predict,
            l.value AS label_value
            FROM label_t AS l
            LEFT JOIN person p
            ON p.person_id = l.person_id
            LEFT JOIN condition_occurrence AS co
            ON co.person_id = l.person_id
            LEFT JOIN concept AS cc
            ON co.condition_concept_id = cc.concept_id
            WHERE co.condition_start_date <= l.prediction_time

            UNION ALL 

            SELECT DISTINCT l.person_id, l.prediction_time, 
            de.drug_concept_id AS concept_id, cc.concept_code, cc.concept_name, cc.vocabulary_id,
            de.drug_exposure_start_date AS event_date,
            'drug_exposure' AS event_type,
            CAST(strftime('%Y', de.drug_exposure_start_date) AS INTEGER) - p.year_of_birth AS age_at_event,
            CAST(strftime('%Y', l.prediction_time) AS INTEGER) - p.year_of_birth AS age_at_predict,
            l.value AS label_value
            FROM label_t AS l
            LEFT JOIN person p
            ON p.person_id = l.person_id
            LEFT JOIN drug_exposure AS de
            ON de.person_id = l.person_id
            LEFT JOIN concept AS cc
            ON de.drug_concept_id = cc.concept_id
            WHERE de.drug_exposure_start_date <= l.prediction_time
        )
        SELECT * FROM t1
        ORDER BY person_id ASC, event_date ASC
    """

def main(args):
    task_folder, task_name = TASK_MAP[args.task]

    # Ingest labels to SQLite table labels_<task>
    labels_path = os.path.join(args.assets_root, "benchmark", task_folder, "labeled_patients.csv")
    df_labels = pd.read_csv(labels_path).rename(columns={"patient_id": "person_id"})
    df_labels["prediction_time"] = pd.to_datetime(df_labels["prediction_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    con = sqlite3.connect(args.sqlite_path)
    tbl = f"labels_{task_folder}"
    df_labels.to_sql(tbl, con, if_exists="replace", index=False)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl} ON {tbl}(person_id, prediction_time);")
    con.commit()

    anchor_agg = "MIN" if args.visit_anchor == "earliest" else "MAX"
    df = pd.read_sql_query(build_query(tbl, anchor_agg), con)
    con.close()

    # Load Qwen
    attn_impl = get_attn_impl()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map={"": 0} if args.device.startswith("cuda") else "cpu",
        cache_dir=args.cache_dir
    ).eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    def llm_generate(sample_text, system_message):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample_text},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
        max_input_tokens = max(1, args.max_context - args.max_new_tokens)
        # Keep head (matches the uploaded script logic)
        if input_ids.shape[1] > max_input_tokens:
            input_ids = input_ids[:, :max_input_tokens]
            attention_mask = attention_mask[:, :max_input_tokens]
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(gen_out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip()

    # Prompt template (JSON-only contract)
    def build_system_message(age_now):
        return (
            f"You are a medical expert evaluating a patient who is currently {age_now} years old. "
            f"Based on the following medical history (formatted as EVENT at AGE), your task is to "
            f"write a concise summary of this patient's risk profile for {task_name} during this admission.\n"
            "\n"
            "Return JSON only:\n"
            "{\n"
            "  'risk_category': 'Low'|'Moderate'|'High',\n"
            "  'risk_score': <0..1>,\n"
            "  'drivers_positive': ['substrings from medical history'],\n"
            "  'drivers_negative': ['substrings from medical history'],\n"
            "  'justification': '2–4 sentences referencing drivers and timing',\n"
            "  'insufficient_evidence': true|false\n"
            "}\n"
            "\n"
            "Guidance:\n"
            "- Do not add external facts.\n"
            "- Prefer explicit outcome-related conditions/medications as positive drivers.\n"
            "- Weigh recent items more (use recent event when present).\n"
            "- If evidence is sparse/ambiguous, prefer Low and set insufficient_evidence=true.\n"
            "- Map Low≈0.10–0.33, Moderate≈0.34–0.66, High≈0.67–0.90.\n"
        )

    # Group by (person, anchor) and generate
    out_csv = os.path.join(args.intermediate_root, f"df_event_MoA_qwen_{task_folder}_{args.run_tag}.csv")
    out_pkl = os.path.join(args.intermediate_root, f"dic_pid_row_llm_MoA_qwen_{task_folder}_{args.run_tag}.pickle")
    os.makedirs(args.intermediate_root, exist_ok=True)

    rows_text, rows_label = [], []
    pid_to_row, idx = {}, 0

    for (pid, anchor_dt), g in tqdm(df.groupby(["person_id", "prediction_time"])):
        age_now = int(g.age_at_predict.values[0])
        sys_msg = build_system_message(age_now)
        g2 = g.drop_duplicates(subset=["concept_name","age_at_event"])
        sample_text = "; ".join([f"{r.concept_name} at {int(r.age_at_event)}" for r in g2.itertuples()]) + ";"
        summary = llm_generate(sample_text, sys_msg)
        rows_text.append(summary)
        rows_label.append(int(g.label_value.values[0]))
        pid_to_row[int(pid)] = idx
        idx += 1

    pd.DataFrame({"text": rows_text, "label": rows_label}).to_csv(out_csv, index=False)
    with open(out_pkl, "wb") as f:
        pickle.dump(pid_to_row, f)

    print("Wrote:", out_csv)
    print("Wrote:", out_pkl)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite_path", required=True)
    ap.add_argument("--assets_root", required=True)
    ap.add_argument("--intermediate_root", default="./outputs/intermediate")
    ap.add_argument("--task", choices=list(TASK_MAP.keys()), required=True)
    ap.add_argument("--visit_anchor", choices=["earliest","latest"], default="earliest")
    ap.add_argument("--run_tag", default="seed2")

    # Qwen
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_dir", default="./hf_cache")
    ap.add_argument("--max_context", type=int, default=7500)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    args = ap.parse_args()
    main(args)
