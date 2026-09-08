"""
evaluation/evaluate.py

Basic evaluation script for the MedGuide AI prototype's mock response
logic and triage classification.

This is a lightweight demonstration harness for the frontend prototype.
It is NOT a clinical validation suite. Once the real backend (multimodal
encoder + RAG + LLM) is integrated, this file should be expanded with
proper evaluation metrics (e.g. retrieval accuracy, groundedness,
triage sensitivity/specificity).

Run with:
    python evaluation/evaluate.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety.triage import classify_urgency, SELF_CARE, CONSULT_DOCTOR, EMERGENCY


# Simple demonstration test cases: (symptom text, expected urgency)
TEST_CASES = [
    ("I have a mild headache and feel stressed", SELF_CARE),
    ("I have a fever and cough for 3 days", CONSULT_DOCTOR),
    ("I have severe chest pain and difficulty breathing", EMERGENCY),
    ("I feel a bit tired today", SELF_CARE),
    ("I lost consciousness briefly", EMERGENCY),
]


def run_evaluation():
    print("Running MedGuide AI triage evaluation (demonstration only)\n")
    correct = 0

    for text, expected in TEST_CASES:
        predicted = classify_urgency(text)
        result = "PASS" if predicted == expected else "FAIL"
        if predicted == expected:
            correct += 1

        print(f"[{result}] Input: '{text}'")
        print(f"       Expected: {expected} | Predicted: {predicted}\n")

    accuracy = correct / len(TEST_CASES) * 100
    print(f"Demonstration accuracy: {accuracy:.1f}% ({correct}/{len(TEST_CASES)})")
    print(
        "\nNote: This evaluation only checks the placeholder keyword-based "
        "triage logic. It does not evaluate real clinical accuracy."
    )


if __name__ == "__main__":
    run_evaluation()
