"""
generate_dispatch_pairs.py — Synthetic fine-tuning dataset for the 0.5B dispatch model.

Uses Claude to paraphrase seed examples into training pairs for each dispatch task.
Output: hybrid/training/data/dispatch_pairs.jsonl in Qwen2.5 chat format.

Tasks covered:
  plotter       — "plot the flat profile"           → {"profile": "flat",   "mode": "healthy"}
  train_sum     — "show loss for v21_2k"            → {"run_id": "v21_2k"}
  uart_reader   — "read /tmp/gait_uart.log"         → {"log_path": "/tmp/gait_uart.log"}
  model_compare — "compare good v1 vs bad random"   → {"good_run_id": "v1", "bad_run_id": "random"}

Usage:
    python hybrid/training/generate_dispatch_pairs.py
    python hybrid/training/generate_dispatch_pairs.py --n_paraphrases 20 --tasks plotter train_sum
    python hybrid/training/generate_dispatch_pairs.py --dry_run  # show seeds only, no API call
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Load .env
_env = _REPO_ROOT / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

OUT_PATH = _REPO_ROOT / "hybrid" / "training" / "data" / "dispatch_pairs.jsonl"

# ── System prompts (must match dispatch agents exactly) ───────────────────────

SYSTEM_PROMPTS = {
    "plotter": (
        "You extract exactly two fields from a user request and return valid JSON only. "
        "No explanation. No markdown. Output: {\"profile\": \"<value>\", \"mode\": \"<value>\"}. "
        "Valid profiles: flat, bad_wear, stairs, slope. Valid modes: healthy, pathological. "
        "Default mode is healthy if not mentioned."
    ),
    "train_sum": (
        "You extract exactly one field from a user request and return valid JSON only. "
        "No explanation. No markdown. Output: {\"run_id\": \"<value>\"}. "
        "If the request says 'latest' or gives no specific run, output {\"run_id\": \"latest\"}. "
        "run_id format examples: v21_2k, v19, v15, v10. Extract the version identifier only."
    ),
    "uart_reader": (
        "You extract exactly one field from a user request and return valid JSON only. "
        "No explanation. No markdown. Output: {\"log_path\": \"<value>\"}. "
        "If the request names a specific file path, extract it exactly. "
        "If no path is mentioned or the user says 'latest', output {\"log_path\": \"latest\"}."
    ),
    "model_compare": (
        "You extract exactly two fields from a user request and return valid JSON only. "
        "No explanation. No markdown. "
        "Output: {\"good_run_id\": \"<value>\", \"bad_run_id\": \"<value>\"}. "
        "If the user says 'untrained', 'random', or 'random init', use 'random' for bad_run_id. "
        "If no good run_id is mentioned, default to 'v1'. "
        "If no bad run_id is mentioned, default to 'random'."
    ),
}

# ── Seed examples per task ────────────────────────────────────────────────────

SEEDS = {
    "plotter": [
        ("plot the flat profile",                        {"profile": "flat",     "mode": "healthy"}),
        ("show flat signal check",                       {"profile": "flat",     "mode": "healthy"}),
        ("generate flat healthy plot",                   {"profile": "flat",     "mode": "healthy"}),
        ("flat",                                         {"profile": "flat",     "mode": "healthy"}),
        ("plot stairs",                                  {"profile": "stairs",   "mode": "healthy"}),
        ("show me the stairs profile",                   {"profile": "stairs",   "mode": "healthy"}),
        ("stair walker signal check",                    {"profile": "stairs",   "mode": "healthy"}),
        ("plot slope profile",                           {"profile": "slope",    "mode": "healthy"}),
        ("slope signal",                                 {"profile": "slope",    "mode": "healthy"}),
        ("show slope walker",                            {"profile": "slope",    "mode": "healthy"}),
        ("plot bad wear",                                {"profile": "bad_wear", "mode": "healthy"}),
        ("bad_wear profile",                             {"profile": "bad_wear", "mode": "healthy"}),
        ("loose fit walker plot",                        {"profile": "bad_wear", "mode": "healthy"}),
        ("show pathological flat",                       {"profile": "flat",     "mode": "pathological"}),
        ("flat pathological mode",                       {"profile": "flat",     "mode": "pathological"}),
        ("plot flat with SI 25%",                        {"profile": "flat",     "mode": "pathological"}),
        ("stairs pathological",                          {"profile": "stairs",   "mode": "pathological"}),
        ("show pathological stairs signal",              {"profile": "stairs",   "mode": "pathological"}),
        ("slope pathological check",                     {"profile": "slope",    "mode": "pathological"}),
        ("bad_wear pathological mode",                   {"profile": "bad_wear", "mode": "pathological"}),
        ("signal diagnostic for flat walker",            {"profile": "flat",     "mode": "healthy"}),
        ("run amendment 11 check for stairs",            {"profile": "stairs",   "mode": "healthy"}),
        ("gy and az plot for slope",                     {"profile": "slope",    "mode": "healthy"}),
        ("imu signal for bad wear profile",              {"profile": "bad_wear", "mode": "healthy"}),
        ("gait check flat healthy",                      {"profile": "flat",     "mode": "healthy"}),
    ],
    "train_sum": [
        ("summarise training run v21_2k",                {"run_id": "v21_2k"}),
        ("show loss curves for v21_2k",                  {"run_id": "v21_2k"}),
        ("v21_2k training summary",                      {"run_id": "v21_2k"}),
        ("plot loss for v19",                            {"run_id": "v19"}),
        ("training summary v19",                         {"run_id": "v19"}),
        ("show v15 loss",                                {"run_id": "v15"}),
        ("loss curve v10",                               {"run_id": "v10"}),
        ("summarise v1",                                 {"run_id": "v1"}),
        ("show the latest training run",                 {"run_id": "latest"}),
        ("plot latest loss curve",                       {"run_id": "latest"}),
        ("most recent training run",                     {"run_id": "latest"}),
        ("latest pinn training summary",                 {"run_id": "latest"}),
        ("show me the training results",                 {"run_id": "latest"}),
        ("what happened in the last run",                {"run_id": "latest"}),
        ("v21 loss plot",                                {"run_id": "v21"}),
        ("training diagnostics for v21_2k",              {"run_id": "v21_2k"}),
        ("epoch log v18",                                {"run_id": "v18"}),
        ("physics loss components v17",                  {"run_id": "v17"}),
        ("show val loss for run v16",                    {"run_id": "v16"}),
        ("pinn training report v21_2k",                  {"run_id": "v21_2k"}),
    ],
    "uart_reader": [
        ("read /tmp/gait_uart.log",                      {"log_path": "/tmp/gait_uart.log"}),
        ("show uart output from /tmp/gait_uart.log",     {"log_path": "/tmp/gait_uart.log"}),
        ("parse /tmp/gait_uart.log",                     {"log_path": "/tmp/gait_uart.log"}),
        ("open /tmp/gait_uart.log",                      {"log_path": "/tmp/gait_uart.log"}),
        ("read the latest uart log",                     {"log_path": "latest"}),
        ("show latest simulation output",                {"log_path": "latest"}),
        ("print uart session",                           {"log_path": "latest"}),
        ("show step events from last sim",               {"log_path": "latest"}),
        ("latest gait uart",                             {"log_path": "latest"}),
        ("read uart",                                    {"log_path": "latest"}),
        ("show the simulation log",                      {"log_path": "latest"}),
        ("print step and snapshot lines",                {"log_path": "latest"}),
        ("read /tmp/gait.log",                           {"log_path": "/tmp/gait.log"}),
        ("parse simulator/logs/gait_uart.log",           {"log_path": "simulator/logs/gait_uart.log"}),
        ("show steps detected in /tmp/gait_uart.log",    {"log_path": "/tmp/gait_uart.log"}),
        ("uart output from last renode run",             {"log_path": "latest"}),
        ("firmware uart log",                            {"log_path": "latest"}),
        ("step count from simulation",                   {"log_path": "latest"}),
        ("read session end from /tmp/gait_uart.log",     {"log_path": "/tmp/gait_uart.log"}),
        ("show si values from uart",                     {"log_path": "latest"}),
    ],
    "model_compare": [
        ("compare good v1 vs bad random",                {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("show untrained model against v1",              {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("good v1 bad random",                           {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("compare v1 to random init",                    {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("v1 vs untrained",                              {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("show trained vs untrained pinn",               {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("pinn comparison v1 random",                    {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("compare good v21_2k vs bad random",            {"good_run_id": "v21_2k","bad_run_id": "random"}),
        ("v21_2k against untrained",                     {"good_run_id": "v21_2k","bad_run_id": "random"}),
        ("show v21_2k vs random init model",             {"good_run_id": "v21_2k","bad_run_id": "random"}),
        ("compare v19 to v1",                            {"good_run_id": "v19",  "bad_run_id": "v1"}),
        ("good v19 bad v1",                              {"good_run_id": "v19",  "bad_run_id": "v1"}),
        ("hybrid demo v1",                               {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("run model comparison",                         {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("show imu shape good vs bad",                   {"good_run_id": "v1",   "bad_run_id": "random"}),
        ("step shape comparison v21_2k random",          {"good_run_id": "v21_2k","bad_run_id": "random"}),
        ("compare pinn v15 against untrained",           {"good_run_id": "v15",  "bad_run_id": "random"}),
        ("good model v1 bad model v3",                   {"good_run_id": "v1",   "bad_run_id": "v3"}),
        ("diff between v21_2k and random model",         {"good_run_id": "v21_2k","bad_run_id": "random"}),
        ("trained v1 vs random initialisation",          {"good_run_id": "v1",   "bad_run_id": "random"}),
    ],
}

# ── Paraphrase prompt ─────────────────────────────────────────────────────────

def _paraphrase_prompt(task: str, request: str, output: dict, n: int) -> str:
    return (
        f"You are generating training data for a small engineering language model.\n\n"
        f"Task: {task} dispatch — parse a natural language request into structured JSON.\n"
        f"Example request: \"{request}\"\n"
        f"Correct JSON output: {json.dumps(output)}\n\n"
        f"Generate {n} different ways an engineer might phrase the same request. "
        f"Use engineering shorthand, abbreviations, varied phrasing, different word orders. "
        f"Include some terse forms (1-3 words) and some verbose forms. "
        f"All paraphrases must map to the SAME JSON output.\n\n"
        f"Return a JSON array of strings only. No explanation. No markdown.\n"
        f"Example: [\"phrase 1\", \"phrase 2\", ...]"
    )


def _to_chat_example(task: str, request: str, output: dict) -> dict:
    """Format as Qwen2.5 chat fine-tuning example."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPTS[task]},
            {"role": "user",      "content": request},
            {"role": "assistant", "content": json.dumps(output)},
        ],
        "_task": task,
        "_source": "seed" if True else "generated",
    }


def generate(n_paraphrases: int = 15, tasks: list = None, dry_run: bool = False) -> list:
    """Generate dispatch training pairs.

    Args:
        n_paraphrases: number of Claude-generated paraphrases per seed example
        tasks:         list of task names to generate for (default: all 4)
        dry_run:       if True, return seeds only without calling Claude API

    Returns:
        list of chat-format training examples
    """
    if tasks is None:
        tasks = list(SEEDS.keys())

    examples = []

    # ── Step 1: add all seeds directly ───────────────────────────────────────
    seed_count = 0
    for task in tasks:
        for request, output in SEEDS[task]:
            examples.append(_to_chat_example(task, request, output))
            seed_count += 1

    print(f"[generate] Seeds: {seed_count} examples across {len(tasks)} tasks")

    if dry_run:
        print("[generate] Dry run — skipping Claude API calls")
        return examples

    # ── Step 2: paraphrase with Claude ───────────────────────────────────────
    import anthropic
    client = anthropic.Anthropic()

    generated_count = 0
    error_count     = 0

    for task in tasks:
        seeds = SEEDS[task]
        print(f"\n[generate] Task: {task}  ({len(seeds)} seeds × {n_paraphrases} paraphrases)")

        for i, (request, output) in enumerate(seeds):
            prompt = _paraphrase_prompt(task, request, output, n_paraphrases)

            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",   # cheapest — pure text generation
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()

                # Strip markdown fences if present
                if raw.startswith("```"):
                    raw = "\n".join(raw.split("\n")[1:])
                if raw.endswith("```"):
                    raw = "\n".join(raw.split("\n")[:-1])

                paraphrases = json.loads(raw.strip())

                for p in paraphrases:
                    if isinstance(p, str) and p.strip():
                        examples.append(_to_chat_example(task, p.strip(), output))
                        generated_count += 1

                print(f"  [{i+1:02d}/{len(seeds)}] '{request[:40]}' → {len(paraphrases)} paraphrases")

                # Rate limit courtesy pause
                time.sleep(0.3)

            except Exception as e:
                error_count += 1
                print(f"  [{i+1:02d}/{len(seeds)}] ERROR: {e}")
                continue

    print(f"\n[generate] Done: {seed_count} seeds + {generated_count} generated = {len(examples)} total")
    if error_count:
        print(f"[generate] Errors: {error_count} seed(s) failed paraphrase — seeds still included")

    return examples


def save(examples: list, path: Path = OUT_PATH):
    """Write examples to JSONL. Shuffles before saving."""
    random.seed(42)
    random.shuffle(examples)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            # Strip internal metadata before saving
            out = {k: v for k, v in ex.items() if not k.startswith("_")}
            f.write(json.dumps(out) + "\n")

    print(f"\n[save] {len(examples)} examples → {path}")
    _print_summary(examples, path)


def _print_summary(examples: list, path: Path):
    from collections import Counter
    task_counts = Counter(ex.get("_task", "?") for ex in examples)

    print()
    print("═" * 52)
    print("DISPATCH TRAINING DATASET SUMMARY")
    print("═" * 52)
    print(f"Output: {path}")
    print(f"Total:  {len(examples)} examples")
    print()
    print(f"{'Task':<16}  {'Count':>6}  {'Example input'}")
    print("─" * 52)
    shown = set()
    for ex in examples:
        task = ex.get("_task", "?")
        if task not in shown:
            user_msg = ex["messages"][1]["content"]
            print(f"{task:<16}  {task_counts[task]:>6}  \"{user_msg[:28]}...\"")
            shown.add(task)
    print("═" * 52)
    print()
    print("Format: Qwen2.5 chat JSONL — ready for LLaMA-Factory / Axolotl fine-tuning")
    print("System prompt per task matches dispatch agent prompts exactly.")
    print("Next: fine-tune Qwen2.5-0.5B with LoRA, export GGUF, replace Ollama model.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dispatch fine-tuning pairs for the 0.5B engineering model"
    )
    parser.add_argument("--n_paraphrases", type=int, default=15,
                        help="Claude paraphrases per seed example (default: 15)")
    parser.add_argument("--tasks", nargs="+",
                        choices=list(SEEDS.keys()),
                        default=None,
                        help="Tasks to generate (default: all 4)")
    parser.add_argument("--out",  default=str(OUT_PATH),
                        help=f"Output JSONL path (default: {OUT_PATH})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print seeds only, no Claude API call")
    args = parser.parse_args()

    examples = generate(
        n_paraphrases=args.n_paraphrases,
        tasks=args.tasks,
        dry_run=args.dry_run,
    )
    save(examples, Path(args.out))
