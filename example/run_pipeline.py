"""
Complete pipeline example with LLM integration.
"""

import json
from dotenv import load_dotenv
from core.pipeline import run_forecast_pipeline
from core.api_schema import ForecastRequest

# Load environment variables
load_dotenv()

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

print("📦 Step 1: Loading data...")
try:
    with open("/Users/rasya2121/Documents/code/projects/dinacom/test_doc.json") as f:
        data = json.load(f)
except FileNotFoundError:
    # Fallback for relative path if absolute fails on different machine
    with open("test_doc.json") as f:
        data = json.load(f)

# Construct ForecastRequest objects
products_data = []
for p in data['products']:
    products_data.append({
        "product_id": p["product_id"],
        "product_name": p["product_name"],
        "unit": p["unit"],
        "stock": {
            "current_stock_on_hand": p["stock"]["current_stock_on_hand"]
        },
        "daily_sales": [
            {"date": d["date"], "qty": d["qty"]} for d in p["daily_sales"]
        ]
    })

request = ForecastRequest(
    products=products_data,
    lebaran_date=data.get("lebaran_date")
)

# ============================================================================
# 2. Run Pipeline
# ============================================================================

print("🚀 Running forecast pipeline...")
complete_json = run_forecast_pipeline(request)

# ============================================================================
# 3. Save Results
# ============================================================================

output_file = "complete_forecast_with_llm.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(complete_json, f, indent=2, ensure_ascii=False)

print(f"\n" + "="*60)
print(f"✅ JSON saved to: {output_file}")
print(f"="*60)

# ============================================================================
# 4. Display Summary
# ============================================================================

print(f"\n📊 FORECAST SUMMARY")
print("="*60)
print(f"📦 Total Products: {complete_json['total_products']}")
print(f"🤖 LLM Enabled: {complete_json['llm_enabled']}")
if complete_json['llm_enabled']:
    print(f"   - Success: {complete_json['llm_success_count']}")
    print(f"   - Fallback: {complete_json['fallback_count']}")
    print(f"   - Tokens Used: {complete_json['total_tokens_used']}")

print(f"\n📈 PORTFOLIO INSIGHTS")
print("="*60)
insights = complete_json['portfolio_insights']
print(f"🚨 High Risk: {insights['summary']['high_risk_count']} products")
print(f"🟡 Medium Risk: {insights['summary']['medium_risk_count']} products")
print(f"📦 Need Restock: {insights['summary']['restock_needed_count']} products")

print(f"\n🎯 TOP 5 PRIORITY PRODUCTS:")
print("="*60)
for idx, p in enumerate(insights['priority_actions'], start=1):
    print(f"{idx}. {p['product_name']}")
    print(f"   Score: {p['priority_score']}/100 | Risk: {p['risk_level']}")

print(f"\n{'='*60}")
print("✅ Pipeline completed successfully!")
print("="*60)
