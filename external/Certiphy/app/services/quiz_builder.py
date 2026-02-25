import hashlib
import math
import random
from typing import Dict, List, Optional, Tuple
from .sonar import sonar_generate_json

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "topic": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                        },
                        "required": ["A", "B", "C", "D"],
                        "additionalProperties": False,
                    },
                    "correct_key": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "explanation": {"type": "string"},
                },
                "required": ["domain", "topic", "question", "options", "correct_key", "explanation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

SCENARIO_TEMPLATES = [
    "A company is planning to {goal}. Which {vendor} service is the BEST choice?",
    "A team needs to {requirement} with minimal operational overhead. Which option should they choose in {vendor}?",
    "An architect must design a solution to {scenario}. Which approach aligns BEST with {vendor} best practices?",
    "A developer observes {symptom}. Which action should they take FIRST in {vendor}?",
    "A security engineer must ensure {security_need}. Which {vendor} feature helps MOST?",
]

DIRECT_TEMPLATES = [
    "Which statement BEST describes {concept} in {vendor}?",
    "What is the PRIMARY purpose of {service_or_feature} in {vendor}?",
    "Which option is a key benefit of {concept} in {vendor}?",
]

def _qid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

def _forbidden_terms(vendor: str) -> List[str]:
    v = (vendor or "").strip().lower()
    if v == "aws":
        return ["azure", "microsoft", "az-900", "dp-900", "ai-900", "gcp", "google cloud", "entra", "resource group"]
    if v == "microsoft":
        return ["aws", "amazon", "ec2", "s3", "cloud practitioner", "gcp", "google cloud", "iam (aws)", "vpc"]
    if v == "google cloud":
        return ["aws", "amazon", "azure", "microsoft", "az-900", "ec2", "s3", "vpc (aws)"]
    return []

def _contains_forbidden(text: str, vendor: str) -> bool:
    lower = (text or "").lower()
    return any(bad in lower for bad in _forbidden_terms(vendor))

def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    lower = (text or "").lower()
    return any(k.lower() in lower for k in (keywords or []))

def _normalize_distribution(domains: Dict[str, float]) -> Dict[str, float]:
    if not domains:
        return {}
    total = sum(max(0.0, float(v)) for v in domains.values())
    if total <= 0:
        # fallback uniform
        n = len(domains)
        return {k: 1.0 / n for k in domains}
    return {k: float(v) / total for k, v in domains.items()}

def _allocate_counts(total_questions: int, distribution: Dict[str, float]) -> Dict[str, int]:
    """
    Allocate integer counts per domain, preserving sum=total_questions.
    """
    dist = _normalize_distribution(distribution)
    if not dist:
        return {"General": total_questions}

    raw = {d: dist[d] * total_questions for d in dist}
    counts = {d: int(math.floor(raw[d])) for d in raw}
    remainder = total_questions - sum(counts.values())

    # distribute remainder by largest fractional part
    fracs = sorted(((d, raw[d] - counts[d]) for d in raw), key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        counts[fracs[i % len(fracs)][0]] += 1

    # remove zeros
    return {d: c for d, c in counts.items() if c > 0}

def _section_pool(sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # Shuffle for diversity
    secs = [s for s in sections if (s.get("title") or "").strip() and (s.get("body") or "").strip()]
    random.shuffle(secs)
    return secs

PROMPT = """
You are generating ORIGINAL practice questions STRICTLY for:

Certification: {exam_name}
Vendor: {vendor}
Domain focus: {domain}

NON-NEGOTIABLE RULES:
- Generate questions ONLY about {vendor}. Do NOT mention other cloud providers or other certifications.
- Do NOT reproduce or paraphrase known exam dump questions.
- If you are uncertain, create a NEW scenario-based question aligned to the objectives.
- If the provided content is insufficient for {domain}, return an empty questions list.

Style requirements:
- Exam mode: {mode}
- Scenario ratio target: {scenario_ratio}
- Use {vendor}-specific services/terminology.
- Make distractors realistic (common misconceptions).

Blueprint content (official guide extract):
Section Title: {title}
Section Content: {body}

Generate exactly {n} questions for this domain (or fewer ONLY if content is insufficient).
Return ONLY JSON matching the schema.
""".strip()

async def generate_domain_questions(
    *,
    exam_name: str,
    vendor: str,
    domain: str,
    mode: str,
    scenario_ratio: float,
    difficulty: str,
    keywords: List[str],
    sections: List[Dict[str, str]],
    target_n: int,
    include_explanations: bool,
    max_retries: int,
) -> List[Dict]:
    results: List[Dict] = []
    pool = _section_pool(sections)

    attempt = 0
    idx = 0

    while len(results) < target_n and attempt <= max_retries and idx < len(pool):
        sec = pool[idx]
        idx += 1

        title = sec["title"].strip()
        body = sec["body"].strip()

        # Hard pre-filter: if section already has forbidden vendor terms, skip
        if _contains_forbidden(f"{title}\n{body}", vendor):
            continue

        prompt = PROMPT.format(
            exam_name=exam_name,
            vendor=vendor,
            domain=domain,
            mode=mode,
            scenario_ratio=scenario_ratio,
            title=title,
            body=body,
            n=min(5, target_n - len(results)),  # generate in small batches
        )

        obj = await sonar_generate_json(prompt, schema=QUIZ_SCHEMA)
        questions = obj.get("questions", []) or []

        for q in questions:
            qt = (q.get("question") or "").strip()
            if not qt:
                continue
            # Post-filters (strict)
            if _contains_forbidden(qt, vendor):
                continue
            if not _contains_any_keyword(qt, keywords):
                # allow some generic conceptual ones, but keep exam relevance high
                # For beginner exams, still require at least a keyword in most cases
                continue

            opts = q.get("options") or {}
            correct = (q.get("correct_key") or "").strip()

            if correct not in ("A", "B", "C", "D"):
                continue
            if any(k not in opts or not str(opts.get(k, "")).strip() for k in ("A", "B", "C", "D")):
                continue

            item = {
                "id": _qid(qt),
                "domain": (q.get("domain") or domain).strip(),
                "topic": (q.get("topic") or title).strip(),
                "question": qt,
                "options": [
                    {"key": "A", "text": str(opts["A"]).strip()},
                    {"key": "B", "text": str(opts["B"]).strip()},
                    {"key": "C", "text": str(opts["C"]).strip()},
                    {"key": "D", "text": str(opts["D"]).strip()},
                ],
                "correct_key": correct,
                "explanation": (q.get("explanation") if include_explanations else None),
            }
            results.append(item)

            if len(results) >= target_n:
                break

        attempt += 1

    return results

async def build_exam_like_quiz(
    *,
    exam_name: str,
    vendor: str,
    sections: List[Dict[str, str]],
    domains: Dict[str, float],
    keywords: List[str],
    mode: str,
    total_questions: int,
    difficulty: str,
    include_explanations: bool,
    scenario_ratio: float,
    min_questions: int,
    max_retries: int,
    domain_distribution_override: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Builds a full quiz with domain distribution + strict vendor relevance.
    """
    distribution = domain_distribution_override or domains or {"General": 1.0}
    allocation = _allocate_counts(total_questions, distribution)

    all_qs: List[Dict] = []
    for domain, count in allocation.items():
        qs = await generate_domain_questions(
            exam_name=exam_name,
            vendor=vendor,
            domain=domain,
            mode=mode,
            scenario_ratio=scenario_ratio,
            difficulty=difficulty,
            keywords=keywords,
            sections=sections,
            target_n=count,
            include_explanations=include_explanations,
            max_retries=max_retries,
        )
        all_qs.extend(qs)

    # Final: ensure minimum
    if len(all_qs) < min_questions:
        # Try a second pass with relaxed keyword requirement by reusing sections but still block forbidden terms
        # (We keep it simple: just return what we have; main can throw a 502 if desired.)
        return all_qs

    # Shuffle for exam feel
    random.shuffle(all_qs)
    return all_qs