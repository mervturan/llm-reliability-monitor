from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag_monitoring.metrics import exact_match, token_f1

def test_exact_match_normalization():
    assert exact_match("The Eiffel Tower.", ["Eiffel Tower"]) == 1.0

def test_token_f1_positive_overlap():
    assert token_f1("Paris France", ["Paris"]) > 0.0
