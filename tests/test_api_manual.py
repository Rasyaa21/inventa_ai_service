"""
Test script to verify FastAPI endpoint manually.
"""
import requests
import json
import os

def test_api():
    url = "http://127.0.0.1:8000/forecast"
    
    try:
        with open("test_doc.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        with open("/Users/rasya2121/Documents/code/projects/dinacom/test_doc.json") as f:
            data = json.load(f)

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
                {"date": d["date"], "qty": int(d["qty"])} for d in p["daily_sales"]
            ]
        })

    payload = {
        "products": products_data,
        "lebaran_date": data.get("lebaran_date")
    }
    
    print(f"Sending request to {url} with {len(products_data)} products...")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("✅ Success!")
            result = response.json()
            print(f"Total Products: {result.get('total_products')}")
            print(f"LLM Enabled: {result.get('llm_enabled')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")

if __name__ == "__main__":
    test_api()
