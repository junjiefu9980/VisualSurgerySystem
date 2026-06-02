"""
Build LLM feedback for selected agent cases.

If GEMINI_API_KEY is available in the environment or repo .env file, this
module sends llm_input.json and the agent evidence pack to Gemini and writes
feedback.md.
The old rule-based text is still saved as rule_feedback.md for fallback.
"""
import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


# ===============================================
# config
# ================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_AGENT_DIR = REPO_ROOT / "output" / "agent"
EVAL_DIR = REPO_ROOT / "output" / "evaluation"

IN_JSON_PATH = OUT_AGENT_DIR / "evidence_pack.json"
LLM_INPUT_PATH = EVAL_DIR / "llm_input.json"
OUT_MD_PATH = OUT_AGENT_DIR / "feedback.md"
RULE_OUT_MD_PATH = OUT_AGENT_DIR / "rule_feedback.md"
PROMPT_OUT_PATH = OUT_AGENT_DIR / "feedback_prompt.txt"
FINAL_CASE_TABLE_PATH = EVAL_DIR / "final_case_table.csv"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_LLM_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

LLM_INSTRUCTIONS = """
You are the feedback agent for a vision-based surgical skill evaluation system.
Write the final feedback in Simplified Chinese.

Use only the supplied JSON evidence. Do not invent case IDs, metrics, model
results, thresholds, or clinical claims. Explain that the route selection is
rule-based unless the data explicitly says otherwise.

Your output must be a concise Markdown report with these sections:
# LLM Agent Feedback
## Overall conclusion
## Policy route interpretation
## Representative case evidence
## Recommended next steps

Mention Kalman smoothing only when the input contains Kalman ablation or chosen
Kalman improvement metrics. Mention trajectory figures only when paths are
provided in the evidence pack. Avoid raw JSON dumps.
""".strip()


# ===============================================
# tools
# ================================================
def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def fmt_num(x, digits=3):
    v = safe_float(x, None)
    if v is None:
        return "None"
    return f"{v:.{digits}f}"


def clean_method_name(method):
    m = str(method).strip().lower()
    if m == "tri":
        return "TRI"
    if m == "monster":
        return "MONSTER"
    if m == "2d":
        return "2D"
    return str(method)


def load_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    import csv

    with open(csv_path, "r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def parse_flag(x):
    s = str(x).strip().lower()
    if s in ("1", "true", "yes"):
        return 1
    if s in ("0", "false", "no"):
        return 0
    return None


def mean_or_none(values):
    arr = [safe_float(v, None) for v in values]
    arr = [v for v in arr if v is not None]
    if len(arr) == 0:
        return None
    return sum(arr) / float(len(arr))


def build_overall_stats():
    # Read final case table, then make a small report summary
    rows = load_rows(FINAL_CASE_TABLE_PATH)
    if len(rows) == 0:
        return None

    total = len(rows)
    mode_count = {"tri": 0, "monster": 0, "2d": 0, "None": 0}
    tri_pass = 0
    monster_pass = 0
    fallback_2d = 0
    bad_cases = 0
    tri_gap_cases = 0

    tri_disp = []
    monster_disp = []
    chosen_kf_gain = []

    for r in rows:
        mode = str(r.get("policy_mode", "None"))
        if mode in mode_count:
            mode_count[mode] += 1

        ok2d = parse_flag(r.get("is_2d_ok", ""))
        oktri = parse_flag(r.get("is_tri_ok", ""))
        okmon = parse_flag(r.get("is_monster_ok", ""))

        if ok2d == 0 or mode == "None":
            bad_cases += 1
        if mode == "2d":
            fallback_2d += 1
        if mode == "tri" and oktri == 1:
            tri_pass += 1
        if mode == "monster" and okmon == 1:
            monster_pass += 1

        tri_disp_val = safe_float(r.get("tri_disp_p95"), None)
        mon_disp_val = safe_float(r.get("monster_disp_p95"), None)
        if mode == "tri" and tri_disp_val is not None:
            tri_disp.append(tri_disp_val)
        if mode == "monster" and mon_disp_val is not None:
            monster_disp.append(mon_disp_val)

        if tri_disp_val is not None and mon_disp_val is not None and mon_disp_val > 1e-6:
            if tri_disp_val > 1.3 * mon_disp_val:
                tri_gap_cases += 1

        gain = safe_float(r.get("chosen_kf_improve_disp_p95"), None)
        if gain is not None:
            chosen_kf_gain.append(gain)

    return {
        "total_cases": total,
        "mode_count": mode_count,
        "tri_pass": tri_pass,
        "monster_pass": monster_pass,
        "fallback_2d": fallback_2d,
        "bad_cases": bad_cases,
        "tri_gap_cases": tri_gap_cases,
        "avg_tri_disp": mean_or_none(tri_disp),
        "avg_monster_disp": mean_or_none(monster_disp),
        "avg_kf_gain": mean_or_none(chosen_kf_gain),
    }


def overall_tags(stats):
    if stats is None:
        return "overall-missing"

    tags = []
    tri_n = int(stats["mode_count"]["tri"])
    mon_n = int(stats["mode_count"]["monster"])
    bad_n = int(stats["bad_cases"])
    gap_n = int(stats["tri_gap_cases"])

    if tri_n >= max(1, mon_n * 3):
        tags.append("tri-dominant")
    if mon_n > 0:
        tags.append("monster-fallback-active")
    if bad_n > 0:
        tags.append("bad-cases-remain")
    if gap_n > 0:
        tags.append("tri-monster-gap-visible")

    if len(tags) == 0:
        tags.append("stable-overall")
    return ", ".join(tags)


def overall_note(stats):
    if stats is None:
        return "Overall summary is not available."

    tri_n = int(stats["mode_count"]["tri"])
    mon_n = int(stats["mode_count"]["monster"])
    two_n = int(stats["mode_count"]["2d"])
    none_n = int(stats["mode_count"]["None"])
    gap_n = int(stats["tri_gap_cases"])

    if gap_n > 0:
        return (
            f"TRI is still the main route ({tri_n} cases), but several TRI cases still show a clear gap "
            f"vs MONSTER. MONSTER is used in {mon_n} cases, 2D fallback remains in {two_n} cases, "
            f"and {none_n} bad cases still need attention."
        )

    return (
        f"TRI stays the main route ({tri_n} cases). MONSTER is used as fallback in {mon_n} cases, "
        f"2D fallback appears in {two_n} cases, and bad cases={none_n}."
    )


def quality_tags(case_entry):
    final_mode = str(case_entry.get("final_mode", ""))
    methods = case_entry.get("methods", {})
    tri = methods.get("tri", {})
    mon = methods.get("monster", {})
    two = methods.get("2d", {})

    tri_disp = safe_float((tri.get("key_metrics", {}) or {}).get("disp_p95"), None)
    mon_disp = safe_float((mon.get("key_metrics", {}) or {}).get("disp_p95"), None)
    tri_red = safe_float((tri.get("key_metrics", {}) or {}).get("red_total_s"), 0.0) or 0.0
    mon_red = safe_float((mon.get("key_metrics", {}) or {}).get("red_total_s"), 0.0) or 0.0
    two_red = safe_float((two.get("key_metrics", {}) or {}).get("red_total_s"), 0.0) or 0.0

    tags = []

    if final_mode == "tri":
        tags.append("tri-ready")
    elif final_mode == "monster":
        tags.append("monster-needed")
    elif final_mode == "2d":
        tags.append("fallback-2d")
    else:
        tags.append("bad-case")

    red_total = max(tri_red, mon_red, two_red)
    if red_total <= 0.0:
        tags.append("clean")
    elif red_total <= 1.5:
        tags.append("mild-red")
    else:
        tags.append("heavy-red")

    if tri_disp is not None and mon_disp is not None and mon_disp > 1e-6:
        ratio = tri_disp / mon_disp
        if 0.85 <= ratio <= 1.15:
            tags.append("close-to-monster")
        elif ratio > 1.5:
            tags.append("tri-weaker")
        elif ratio < 0.67:
            tags.append("tri-stronger")

    if final_mode == "None":
        tags.append("visual-fail")

    return ", ".join(tags)


def short_note(case_entry):
    final_mode = str(case_entry.get("final_mode", ""))
    methods = case_entry.get("methods", {})

    tri = methods.get("tri", {})
    mon = methods.get("monster", {})
    two = methods.get("2d", {})

    tri_disp = safe_float((tri.get("key_metrics", {}) or {}).get("disp_p95"), None)
    mon_disp = safe_float((mon.get("key_metrics", {}) or {}).get("disp_p95"), None)
    tri_evt = int(((tri.get("key_metrics", {}) or {}).get("event_count", 0)) or 0)
    mon_evt = int(((mon.get("key_metrics", {}) or {}).get("event_count", 0)) or 0)
    two_evt = int(((two.get("key_metrics", {}) or {}).get("event_count", 0)) or 0)

    if final_mode == "tri":
        if tri_disp is not None and mon_disp is not None and mon_disp > 1e-6:
            if tri_disp <= 1.2 * mon_disp:
                return "TRI looks usable on this side."
            return "TRI passes the gate, but the shape gap vs MONSTER still looks noticeable."
        return "TRI looks usable on this side."

    if final_mode == "monster":
        return "MONSTER is the safer 3D choice on this side."

    if final_mode == "2d":
        if tri_evt > 0 and mon_evt > 0:
            return "Both 3D routes stay unstable here, so 2D fallback is easier to justify."
        return "2D fallback is selected for this side."

    if two_evt > 0:
        return "Visual tracking is weak on this side."
    return "This side remains a bad case."


def build_method_line(method, pack):
    metrics = pack.get("key_metrics", {}) or {}
    return (
        f"- {clean_method_name(method)}: "
        f"frames={int(metrics.get('track_frames', 0) or 0)}, "
        f"valid={fmt_num(metrics.get('valid_ratio'), 3)}, "
        f"disp_p95={fmt_num(metrics.get('disp_p95'), 3)}, "
        f"red_s={fmt_num(metrics.get('red_total_s'), 2)}, "
        f"events={int(metrics.get('event_count', 0) or 0)}"
    )


def route_reason(case_entry):
    reason = str(case_entry.get("selection_reason", "")).strip()
    if reason != "":
        return reason

    gate = case_entry.get("gate_reasons", {}) or {}
    final_mode = str(case_entry.get("final_mode", ""))
    if final_mode == "monster":
        return f"TRI fails: {gate.get('reason_tri', '')}"
    if final_mode == "2d":
        return f"TRI fails: {gate.get('reason_tri', '')}; MONSTER fails: {gate.get('reason_monster', '')}"
    if final_mode == "None":
        return f"2D fails: {gate.get('reason_2d', '')}"
    return "TRI passes the current gate."


def build_text(data):
    # Rule-based fallback feedback, simple enough for report appendix
    lines = []
    lines.append("# Rule-Based Agent Feedback")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    overall = build_overall_stats()
    if overall is not None:
        lines.append("## Overall")
        lines.append("")
        lines.append(f"- Total cases: `{overall['total_cases']}`")
        lines.append(
            f"- Policy counts: `tri={overall['mode_count']['tri']}`, "
            f"`monster={overall['mode_count']['monster']}`, "
            f"`2d={overall['mode_count']['2d']}`, "
            f"`None={overall['mode_count']['None']}`"
        )
        lines.append(
            f"- Gate buckets: `tri pass={overall['tri_pass']}`, "
            f"`monster pass={overall['monster_pass']}`, "
            f"`fallback 2d={overall['fallback_2d']}`, "
            f"`bad cases={overall['bad_cases']}`"
        )
        lines.append(
            f"- Mean metrics: `tri disp_p95={fmt_num(overall['avg_tri_disp'], 3)}`, "
            f"`monster disp_p95={fmt_num(overall['avg_monster_disp'], 3)}`, "
            f"`chosen kf gain={fmt_num(overall['avg_kf_gain'], 3)}`"
        )
        lines.append(f"- Feedback: `{overall_tags(overall)}`")
        lines.append(f"- Note: {overall_note(overall)}")
        lines.append("")

    cases = data.get("cases", [])
    if len(cases) == 0:
        lines.append("## Selected Cases")
        lines.append("")
        lines.append("No selected case is available.")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Selected Cases")
    lines.append("")

    for case_entry in cases:
        case_id = str(case_entry.get("case_id", ""))
        side = str(case_entry.get("side", ""))
        final_mode = str(case_entry.get("final_mode", ""))
        chosen_method = str(case_entry.get("chosen_method", ""))
        anchor_kpt = str(case_entry.get("anchor_kpt", ""))
        scenario = str(case_entry.get("scenario", ""))
        methods = case_entry.get("methods", {})

        lines.append(f"## Case {case_id} | {side}")
        lines.append("")
        lines.append(f"- Final mode: `{final_mode}`")
        lines.append(f"- Chosen method: `{chosen_method}`")
        lines.append(f"- Scenario: `{scenario}`")
        lines.append(f"- Anchor kpt: `{anchor_kpt}`")
        lines.append(f"- Route reason: {route_reason(case_entry)}")

        if chosen_method in methods:
            lines.append(build_method_line(chosen_method, methods[chosen_method]))
            figs = methods[chosen_method].get("figures", [])
            traj_png = next((f for f in figs if str(f).endswith("_traj.png")), "")
            timeline_png = next((f for f in figs if str(f).endswith("_timeline.png")), "")
            if traj_png != "":
                lines.append(f"- Trajectory figure: `{traj_png}`")
            if timeline_png != "":
                lines.append(f"- Timeline figure: `{timeline_png}`")

        lines.append(f"- Feedback: `{quality_tags(case_entry)}`")
        lines.append(f"- Note: {short_note(case_entry)}")
        lines.append("")

    return "\n".join(lines)


# ===============================================
# llm feedback
# ================================================
def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_env(env_path=REPO_ROOT / ".env"):
    env_path = Path(env_path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s == "" or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def get_gemini_api_key():
    load_local_env()
    return (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY", "").strip()
    )


def build_llm_prompt(llm_input, evidence_pack, rule_feedback):
    llm_input_text = json.dumps(llm_input or {}, ensure_ascii=False, indent=2)
    evidence_text = json.dumps(evidence_pack or {}, ensure_ascii=False, indent=2)

    return "\n".join(
        [
            "Generate the final natural-language feedback for this project run.",
            "",
            "Context:",
            "- llm_input_json comes from output/evaluation/llm_input.json.",
            "- evidence_pack_json comes from output/agent/evidence_pack.json.",
            "- rule_feedback is a deterministic fallback summary; use it only as a guide.",
            "",
            "What to emphasize:",
            "- Overall quality of the visual pipeline.",
            "- Why policy selected TRI, MONSTER, 2D fallback, or None.",
            "- What the representative cases show.",
            "- Whether Kalman smoothing improved trajectory stability.",
            "- Concrete next steps for improving bad or fallback cases.",
            "",
            "llm_input_json:",
            "```json",
            llm_input_text,
            "```",
            "",
            "evidence_pack_json:",
            "```json",
            evidence_text,
            "```",
            "",
            "rule_feedback:",
            "```markdown",
            rule_feedback,
            "```",
        ]
    )


def extract_response_text(response_json):
    chunks = []
    for candidate in response_json.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            value = part.get("text", "")
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())

    return "\n\n".join(chunks).strip()


def call_gemini_generate_content(prompt, model=DEFAULT_LLM_MODEL, timeout_s=90):
    api_key = get_gemini_api_key()
    if api_key == "":
        raise RuntimeError("GEMINI_API_KEY is not set")

    body = {
        "system_instruction": {"parts": [{"text": LLM_INSTRUCTIONS}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 4096,
        },
    }
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        GEMINI_GENERATE_URL.format(model=model),
        data=data,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {e.code}: {detail}") from e

    payload = json.loads(raw)
    text = extract_response_text(payload)
    if text == "":
        raise RuntimeError(f"Gemini API returned an empty text response: {payload}")
    return text


def build_no_key_feedback(rule_out_md_path, prompt_out_path):
    return "\n".join(
        [
            "# LLM Agent Feedback",
            "",
            "LLM feedback was not generated because `GEMINI_API_KEY` is not set.",
            "",
            f"- Rule-based fallback saved to `{rule_out_md_path}`.",
            f"- LLM prompt saved to `{prompt_out_path}`.",
            "- Add `GEMINI_API_KEY` to `.env` or the shell environment, then rerun `python3 src/agent/run_agent_pipeline.py`.",
            "",
        ]
    )


def build_llm_error_feedback(error, rule_out_md_path, prompt_out_path):
    return "\n".join(
        [
            "# LLM Agent Feedback",
            "",
            "LLM feedback was not generated because the API call failed.",
            "",
            f"- Error: `{str(error)}`",
            f"- Rule-based fallback saved to `{rule_out_md_path}`.",
            f"- LLM prompt saved to `{prompt_out_path}`.",
            "",
        ]
    )


# ===============================================
# run
# ================================================
def run(
    in_json_path=IN_JSON_PATH,
    out_md_path=OUT_MD_PATH,
    llm_input_path=LLM_INPUT_PATH,
    rule_out_md_path=RULE_OUT_MD_PATH,
    prompt_out_path=PROMPT_OUT_PATH,
    use_llm=True,
):
    in_json_path = Path(in_json_path)
    out_md_path = Path(out_md_path)
    llm_input_path = Path(llm_input_path)
    rule_out_md_path = Path(rule_out_md_path)
    prompt_out_path = Path(prompt_out_path)

    print("\n", "-" * 5, "Step 1: Load evidence pack", "-" * 5)
    if not in_json_path.exists():
        print(f"[ERROR] missing: {in_json_path}")
        return None

    data = load_json(in_json_path)

    print("\n", "-" * 5, "Step 2: Build rule fallback and LLM prompt", "-" * 5)
    rule_text = build_text(data)
    rule_out_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rule_out_md_path, "w", encoding="utf-8") as f:
        f.write(rule_text)
    print(f"[DONE] saved: {rule_out_md_path}")

    llm_input = load_json(llm_input_path)
    if llm_input is None:
        print(f"[WARN] missing: {llm_input_path}")
        llm_input = {}

    prompt = build_llm_prompt(llm_input=llm_input, evidence_pack=data, rule_feedback=rule_text)
    prompt_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_out_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"[DONE] saved: {prompt_out_path}")

    print("\n", "-" * 5, "Step 3: Build LLM feedback", "-" * 5)
    if not use_llm:
        text = build_no_key_feedback(rule_out_md_path, prompt_out_path)
        print("[SKIP] LLM feedback disabled")
    elif get_gemini_api_key() == "":
        text = build_no_key_feedback(rule_out_md_path, prompt_out_path)
        print("[SKIP] GEMINI_API_KEY is not set")
    else:
        try:
            text = call_gemini_generate_content(prompt)
            print(f"[DONE] LLM feedback generated with model={DEFAULT_LLM_MODEL}")
        except Exception as e:
            text = build_llm_error_feedback(e, rule_out_md_path, prompt_out_path)
            print(f"[ERROR] LLM feedback failed: {e}")

    print("\n", "-" * 5, "Step 4: Save feedback", "-" * 5)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[DONE] saved: {out_md_path}")
    return str(out_md_path)


def main():
    run()


if __name__ == "__main__":
    main()
