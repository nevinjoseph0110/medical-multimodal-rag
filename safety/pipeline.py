"""
safety/pipeline.py

Demonstrates the intended future safety pipeline flow for MedGuide AI:

    input -> response generation -> safety check -> triage -> patient-facing output

For this prototype, "response generation" is mock logic (see
frontend/app.py: generate_response). This module simply wires together
the safety check and triage steps so the overall flow is visible and can
be extended once the real multimodal/RAG/LLM backend is ready.
"""

from safety.triage import classify_urgency
from safety.hallucination import check_response_grounding


def run_safety_pipeline(symptom_text: str, response_dict: dict) -> dict:
    """
    Run the mock response through the safety pipeline.

    Steps:
    1. Structural grounding check (placeholder)
    2. Urgency classification (rule-based demonstration)
    3. Attach results to the response for patient-facing output

    Returns the response_dict with two extra keys:
    - "grounding_check": result of check_response_grounding
    - "urgency": final urgency label (may confirm or override the mock value)
    """
    grounding_check = check_response_grounding(response_dict)

    # In this prototype, urgency is primarily determined by the mock
    # response generator, but we re-run the triage classifier on the raw
    # symptom text as a demonstration of an independent safety check.
    triage_result = classify_urgency(symptom_text)

    response_dict["grounding_check"] = grounding_check
    # Prefer the mock response's own urgency if it already set one;
    # otherwise fall back to the triage classifier.
    response_dict.setdefault("urgency", triage_result)

    return response_dict
