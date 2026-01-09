
import sys
import os
import json
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm import generate_portfolio_llm_analysis

def test_fallback_behavior():
    print("🧪 Testing LLM Fallback Behavior...")

    products_data = [
        {
            "product_name": "Critical Product",
            "stock_analysis": {"risk_level": "HIGH", "days_until_stockout": 2},
            "recommendation": {"action": "RESTOCK", "quantity_range": {"min": 10, "max": 20}, "reason": "Low stock"},
            "business_priority": {"priority_score": 90, "priority_tier": "CRITICAL"},
        }
    ]
    
    # Mock OpenAI client
    mock_client = MagicMock()
    # We don't even need to mock expection if prompt generation fails first
    # But let's mock it anyway
    mock_client.chat.completions.create.side_effect = Exception("API Connection Failed")

    # Run function
    result, tokens = generate_portfolio_llm_analysis(products_data, client=mock_client)
    
    print("\nResult JSON:", json.dumps(result, indent=2))
    
    # Verify keys exist (Fallback should produce them)
    assert result is not None, "Result should not be None"
    assert "pattern_trend_summary" in result, "Missing pattern_trend_summary"
    assert "priority_actions" in result, "Missing priority_actions"
    
    print("\n✅ Verification Successful: Fallback handled error gracefully.")

if __name__ == "__main__":
    test_fallback_behavior()
