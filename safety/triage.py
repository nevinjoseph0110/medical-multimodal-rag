"""
safety/triage.py

Demonstration triage module for the MedGuide AI prototype.

IMPORTANT:
This is NOT a real clinical triage system. It uses simple keyword matching
to demonstrate where a safety/urgency classification step will sit in the
final pipeline. The real version will eventually use the outputs of the
multimodal encoder, RAG retrieval, and grounded response generation stages.
"""

# Urgency levels used throughout the app
SELF_CARE = "SELF-CARE"
CONSULT_DOCTOR = "CONSULT DOCTOR"
EMERGENCY = "EMERGENCY"

# Keyword lists used ONLY for this demonstration prototype.
_EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "can't breathe", "cannot breathe",
    "shortness of breath", "severe bleeding", "heavy bleeding",
    "loss of consciousness", "lost consciousness", "unconscious", "fainted", "stroke",
    "slurred speech", "seizure", "severe allergic reaction",
    "anaphylaxis", "suicidal", "overdose", "not breathing",
    "blue lips", "crushing pain",
]

_CONSULT_DOCTOR_KEYWORDS = [
    "fever", "cough", "vomiting", "diarrhea", "rash", "infection",
    "persistent pain", "high temperature", "swelling", "dizziness",
    "shortness of breath on exertion", "blood in stool", "blood in urine",
]

_SELF_CARE_KEYWORDS = [
    "headache", "mild headache", "stress", "dehydration", "tired",
    "fatigue", "minor ache", "sore muscle", "mild pain", "tension",
]


def classify_urgency(symptom_text: str) -> str:
    """
    Classify the urgency of a symptom description into one of:
    SELF-CARE, CONSULT DOCTOR, or EMERGENCY.

    This is a placeholder, rule-based classifier for demonstration
    purposes only. It does NOT represent real clinical triage logic.
    The production system will replace this with a properly validated
    safety layer informed by the actual model pipeline.
    """
    if not symptom_text:
        return SELF_CARE

    text = symptom_text.lower()

    for keyword in _EMERGENCY_KEYWORDS:
        if keyword in text:
            return EMERGENCY

    for keyword in _CONSULT_DOCTOR_KEYWORDS:
        if keyword in text:
            return CONSULT_DOCTOR

    for keyword in _SELF_CARE_KEYWORDS:
        if keyword in text:
            return SELF_CARE

    # Default: when nothing is recognized, err toward suggesting the user
    # provide more detail rather than guessing at urgency.
    return CONSULT_DOCTOR
