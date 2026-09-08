"""
safety/hallucination.py

Placeholder grounding / hallucination-check module for the MedGuide AI
prototype.

IMPORTANT:
This is NOT a real hallucination detector. Real grounding validation
requires comparing generated text against retrieved evidence from the
RAG stage, which is not implemented yet in this frontend-only prototype.

For now, this module performs very basic structural checks on the mock
response so the pipeline has a clear place for this step later.
"""

REQUIRED_FIELDS = ["response", "urgency", "recommended_action"]


def check_response_grounding(response_dict: dict) -> dict:
    """
    Perform a basic structural check on a mock response.

    This placeholder only verifies that expected fields are present and
    non-empty. It does not verify factual accuracy or medical correctness.
    A real implementation will validate the generated response against
    retrieved sources from the RAG module.
    """
    issues = []

    for field in REQUIRED_FIELDS:
        if not response_dict.get(field):
            issues.append(f"Missing or empty field: {field}")

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "issues": issues,
        "note": (
            "This is a placeholder structural check only. It does not "
            "perform real hallucination detection or clinical fact "
            "verification."
        ),
    }
