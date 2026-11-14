"""Bind an Agent's explicit per-result decisions to frozen evaluation artifacts.

This tool does not grade answers or call a model. Every question and every link
must already have an explicit decision in notes authored after reading sources.
"""
import argparse
import json
from pathlib import Path

from evaluation.review import make_template, summarize_agent


def bind_agent_notes(root, notes):
    annotation = make_template(root, "agent")
    decisions = {}
    for item in notes["reviews"]:
        key = item["mode"], item["question_id"]
        if key in decisions:
            raise ValueError("Duplicate Agent decisions")
        decisions[key] = item
    expected = {(entry["mode"], entry["question_id"]) for entry in annotation["reviews"]}
    if set(decisions) != expected:
        raise ValueError("Agent decisions must explicitly cover all modes/questions, including failures")
    for entry in annotation["reviews"]:
        item = decisions[entry["mode"], entry["question_id"]]
        if item.get("result_sha256") != entry["result_sha256"]:
            raise ValueError("Agent notes/result SHA changed; reread the result")
        positives, negatives = item["true_links"], item["false_links"]
        indices = positives + negatives
        if (len(set(indices)) != len(indices) or any(type(i) is not int for i in indices)
                or set(indices) != set(range(len(entry["links"])))):
            raise ValueError("Every published citation link needs exactly one explicit decision")
        entry.update(agent_review=True, reviewer=notes["reviewer"], covered_required_keys=item["covered"],
                     action_correct=item["action"], conditions_correct=item["conditions"],
                     forbidden_conclusions_present=item["forbidden"], notes=item["notes"])
        for number, link in enumerate(entry["links"]):
            link.update(supports_claim=(number in positives) if link["relation"] == "supports" else None,
                        relation_correct=number in positives)
    for mode in sorted({key[0] for key in expected}):
        summarize_agent(root, mode, annotation)
    return annotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("notes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotation = bind_agent_notes(args.root, json.loads(args.notes.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(annotation, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
