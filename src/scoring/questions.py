"""Turn claims into d3 questions and d3 answers into claim scores."""

from __future__ import annotations

from scoring.claims import FORMATS, MEDIA

STATE = (
    "The images are chronological keyframes sampled from one 30-second segment of a video. "
    "Judge only what is visible in these keyframes."
)
VERDICT = {
    "supported": "Yes, the keyframes show this",
    "contradicted": "No, the keyframes show this is wrong or absent",
    "cannot_tell": "The keyframes do not show enough to decide",
}
CENTRALITY = ["Not shown in the segment", "Shown but incidental", "A main subject of the segment"]


def _verdict(instructions):
    return {"type": "choice", "instructions": instructions, "criteria": dict(VERDICT)}


def question(claim: dict) -> dict | None:
    """The d3 question of one claim; None when the claim scores without a question."""
    kind, value = claim["kind"], claim["value"]
    if kind in ("medium", "format"):
        options = MEDIA if kind == "medium" else FORMATS
        if value not in options:
            return None  # a label outside the prompt's list is wrong by definition
        text = ("What is the medium of this segment?" if kind == "medium"
                else "What is the presentation format of this segment?")
        return {"type": "choice", "instructions": text, "criteria": dict(options)}
    if kind == "topic":
        return {
            "type": "score",
            "instructions": f"How central is this topic to the segment: \"{value}\"?",
            "criteria": list(CENTRALITY),
        }
    if kind == "entity":
        return _verdict(f"Is this visible in at least one keyframe: {value}?")
    if kind == "attribute":
        return _verdict(
            f"Look at the {claim['entity']} in the keyframes. "
            f"Is this description of it correct: \"{value}\"?"
        )
    if kind == "action":
        subject = f"the {claim['actor']}" if claim.get("actor") else "someone or something"
        return _verdict(f"Do the keyframes show {subject} doing this: \"{value}\"?")
    subject = f"the {claim['actor']}" if claim.get("actor") else "someone or something"
    asked = {
        "target": "what is acted on",
        "tool": "what tool or means is used to do it",
        "place": "where does it take place",
    }[kind]
    return {
        "type": "choice",
        "instructions": f"In the action \"{claim['action']}\" done by {subject}, {asked}?",
        "criteria": {**claim["options"], "none": "Nothing, or none of the listed items"},
    }


def claim_score(claim: dict, answer: dict | None) -> dict:
    """Probability that the claim holds, and the judge's abstention where it has one."""
    if answer is None:
        return {"p": 0.0, "note": "label outside the allowed list"}
    if "error" in answer:
        return {"p": None, "note": f"{answer['error']}: {answer.get('message', '')}".strip()}
    if answer["type"] == "score":
        return {"p": answer["score"] / (len(CENTRALITY) - 1)}
    probabilities = answer["probabilities"]
    if "cannot_tell" in probabilities and "supported" in probabilities:
        return {"p": probabilities["supported"], "abstain": probabilities["cannot_tell"]}
    return {"p": probabilities.get(claim["value"], 0.0), "judge_choice": answer["choice"]}


def unit_scores(claims: list[dict]) -> dict:
    """Mean effective score per unit; a claim counts for no more than its parent."""
    by_id = {claim["id"]: claim for claim in claims}

    def effective(claim):
        if "effective" not in claim:
            parent = by_id.get(claim.get("parent"))
            base = claim["p"]
            claim["effective"] = (
                None if base is None
                else base if parent is None
                else None if effective(parent) is None else base * effective(parent)
            )
        return claim["effective"]

    totals: dict[str, list[float]] = {}
    for claim in claims:
        value = effective(claim)
        if value is not None:
            totals.setdefault(claim["unit"], []).append(value)
    return {unit: {"score": sum(values) / len(values), "claims": len(values)}
            for unit, values in totals.items()}
