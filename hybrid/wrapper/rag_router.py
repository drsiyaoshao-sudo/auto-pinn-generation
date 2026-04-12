"""
rag_router.py — Corpus tier filter for the GaitSense hybrid RAG system.

Every agent carries a `contract.retrieves` block declaring which corpus tiers
it is permitted to access. This module enforces that boundary: given an agent
name and a query, it returns only the corpus entries the agent is contractually
allowed to retrieve from.

Usage:
    from hybrid.wrapper.rag_router import get_permitted_chunks

    chunks = get_permitted_chunks("plotter", "walker model step detection")
    # → [{"path": "simulator/walker_model.py", "tier": "PRIVATE"}, ...]

    chunks = get_permitted_chunks("pinn-executor", "walker model step detection")
    # → []   ← pinn-executor cannot retrieve PRIVATE; walker_model.py is filtered out

CLI smoke test:
    python hybrid/wrapper/rag_router.py --agent plotter --query "walker model"
    python hybrid/wrapper/rag_router.py --agent pinn-executor --query "walker model"
    python hybrid/wrapper/rag_router.py --agent loss-setter --query "lambda derivation"

Constitutional grounding:
    corpus_classification.md — tier definitions and tier truth table
    skill_contract_spec.md   — contract format spec (contract.retrieves field)
    Article II               — no agent self-selects access beyond its contract
"""

import os
import re
import json
import fnmatch
import argparse
from typing import Optional

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

CORPUS_INDEX_PATH = os.path.join(_REPO_ROOT, "docs", "gaitsense_code", "corpus_index.json")
AGENTS_DIR        = os.path.join(_REPO_ROOT, ".claude", "agents")

TIER_ORDER = {"PUBLIC": 0, "DERIVED-OK": 1, "PRIVATE": 2}


# ── Index loading ─────────────────────────────────────────────────────────────

def load_corpus_index() -> list:
    """Return the full corpus index as a list of {path, tier} dicts."""
    with open(CORPUS_INDEX_PATH) as f:
        return json.load(f)


# ── Contract parsing ──────────────────────────────────────────────────────────

def _parse_yaml_retrieves(agent_md_path: str) -> list:
    """Extract the list of permitted tiers from the contract.retrieves block(s).

    Uses a line-by-line state machine so it never accidentally picks up tier
    values from `receives:`, `produces:`, or `must_not_forward:` sections.
    Works for both simple contracts and split contracts (step_1 / step_2).
    """
    with open(agent_md_path) as f:
        lines = f.readlines()

    # Find frontmatter (content between the first --- and the second ---)
    fm_lines = []
    dashes_seen = 0
    for line in lines:
        if line.strip() == "---":
            dashes_seen += 1
            if dashes_seen == 2:
                break
            continue
        if dashes_seen == 1:
            fm_lines.append(line)

    # State-machine: collect tiers only from retrieves: sections
    retrieves_tiers = set()
    in_retrieves = False
    retrieves_indent = None

    for line in fm_lines:
        stripped = line.lstrip()
        indent   = len(line) - len(stripped)

        # Entering a retrieves: block
        if stripped.startswith("retrieves:"):
            in_retrieves    = True
            retrieves_indent = indent
            continue

        if in_retrieves:
            # Exit the retrieves block when we hit a sibling key at the same indent
            # (a line that starts a new key, not a sub-item)
            if stripped and not stripped.startswith("-") and not stripped.startswith("#"):
                if indent <= retrieves_indent and ":" in stripped.split("#")[0]:
                    in_retrieves = False
                    continue

            # Extract tier values within the block
            tier_match = re.search(r"tier:\s*(PUBLIC|DERIVED-OK|PRIVATE)", line)
            if tier_match:
                retrieves_tiers.add(tier_match.group(1))

    return list(retrieves_tiers)


def load_contract_tiers(agent_name: str) -> list:
    """Return the list of corpus tiers the named agent is permitted to retrieve.

    Looks for <agent_name>.md in .claude/agents/.
    Returns e.g. ["PRIVATE", "PUBLIC"] for plotter.
    Raises FileNotFoundError if agent definition does not exist.
    Raises ValueError if the agent has no contract block.
    """
    # Normalise name (accept 'attorney-A' or 'Attorney-A')
    candidates = [
        os.path.join(AGENTS_DIR, f"{agent_name}.md"),
        os.path.join(AGENTS_DIR, f"{agent_name.lower()}.md"),
    ]
    agent_path = None
    for c in candidates:
        if os.path.exists(c):
            agent_path = c
            break

    if agent_path is None:
        raise FileNotFoundError(
            f"Agent definition not found for '{agent_name}' in {AGENTS_DIR}"
        )

    tiers = _parse_yaml_retrieves(agent_path)
    if not tiers:
        raise ValueError(
            f"Agent '{agent_name}' has no contract.retrieves block in {agent_path}"
        )
    return tiers


# ── Chunk filtering ───────────────────────────────────────────────────────────

def _path_matches(entry_path: str, query: str) -> bool:
    """Simple keyword match: query words appear in the path or filename."""
    words = query.lower().split()
    target = entry_path.lower()
    return all(w in target for w in words)


def _glob_match(entry_path: str, pattern: str) -> bool:
    """Check if a corpus entry path matches a glob pattern."""
    # Match against the basename and the full path
    return (fnmatch.fnmatch(os.path.basename(entry_path), pattern) or
            fnmatch.fnmatch(entry_path, pattern))


def get_permitted_chunks(agent_name: str, query: str,
                         require_match: bool = True) -> list:
    """Return corpus entries the agent may retrieve that match the query.

    Args:
        agent_name:    Name of the agent (must match .claude/agents/<name>.md)
        query:         Keyword query string — matched against entry paths
        require_match: If True (default), only return entries that match the query.
                       If False, return all entries the agent can access (full permitted set).

    Returns:
        List of {path, tier} dicts, sorted by tier (PUBLIC → DERIVED-OK → PRIVATE).

    Example:
        get_permitted_chunks("plotter", "walker model")
        → [{"path": "simulator/walker_model.py", "tier": "PRIVATE"}]

        get_permitted_chunks("pinn-executor", "walker model")
        → []   ← PRIVATE filtered out for pinn-executor
    """
    permitted_tiers = set(load_contract_tiers(agent_name))
    index           = load_corpus_index()

    # Filter by tier
    tier_filtered = [e for e in index if e["tier"] in permitted_tiers]

    # Filter by query match
    if require_match and query:
        tier_filtered = [e for e in tier_filtered if _path_matches(e["path"], query)]

    # Sort by tier order (PUBLIC first, PRIVATE last)
    return sorted(tier_filtered, key=lambda e: TIER_ORDER.get(e["tier"], 99))


def get_all_permitted(agent_name: str) -> list:
    """Return all corpus entries the agent may access (no query filter)."""
    return get_permitted_chunks(agent_name, query="", require_match=False)


# ── CLI smoke test ────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="GaitSense RAG tier filter — test corpus access for an agent"
    )
    parser.add_argument("--agent", required=True, help="Agent name (e.g. plotter)")
    parser.add_argument("--query", default="",    help="Keyword query (e.g. 'walker model')")
    parser.add_argument("--all",   action="store_true",
                        help="Show full permitted corpus (no query filter)")
    args = parser.parse_args()

    print(f"\nRAG Router — agent={args.agent}  query='{args.query}'")
    try:
        permitted_tiers = load_contract_tiers(args.agent)
        print(f"Permitted tiers: {permitted_tiers}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return

    chunks = get_all_permitted(args.agent) if args.all else \
             get_permitted_chunks(args.agent, args.query)

    if not chunks:
        print("No matching chunks — query filtered out or tier boundary enforced.\n")
        return

    print(f"\n{'Path':<70}  {'Tier':<12}")
    print("─" * 85)
    for c in chunks:
        print(f"{c['path']:<70}  {c['tier']:<12}")
    print(f"\n{len(chunks)} chunk(s) returned.\n")


if __name__ == "__main__":
    _cli()
