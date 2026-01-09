"""
LLM analysis module for inventory recommendations.
"""
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from groq import Groq
import os


def get_llm_client(api_key: Optional[str] = None) -> Groq:
    """Get Groq client with API key from environment or parameter."""
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable or pass api_key parameter.")
    
    return Groq(api_key=api_key)


def generate_batch_llm_analysis(
    products_data: List[Dict[str, Any]], 
    model: str = "deepseek-r1-distill-llama-70b", 
    batch_size: int = 3,
    client: Optional[Groq] = None
) -> tuple[List[Dict[str, Any]], int]:
    """
    Generate LLM analysis untuk multiple products sekaligus (batch).
    Lebih efisien dan hemat API calls.
    
    Args:
        products_data: List of product data dictionaries
        model: OpenAI model to use
        batch_size: Number of products per batch
        client: Optional OpenAI client instance
        
    Returns:
        Tuple of (results list, total_tokens)
    """
    if client is None:
        client = get_llm_client()
    
    results = []
    total_tokens = 0
    
    # Sort by priority
    sorted_products = sorted(
        products_data, 
        key=lambda x: (
            x['stock_analysis']['risk_level'] == 'HIGH',
            x['business_priority']['priority_score']
        ),
        reverse=True
    )
    
    for i in range(0, len(sorted_products), batch_size):
        batch = sorted_products[i:i+batch_size]
        
        batch_prompts = []
        for idx, p in enumerate(batch, start=1):
            prompt = f"""
**PRODUK {idx}: {p['product_name']}**
- Stok: {p['current_stock']} {p['unit']}
- Permintaan 14 hari: {p['forecast']['total_demand']} {p['unit']} (rata-rata {p['forecast']['average_per_day']}/hari)
- Risiko: {p['stock_analysis']['risk_level']} (habis dalam {p['stock_analysis']['days_until_stockout'] or '>14'} hari)
- Rekomendasi: {p['recommendation']['action']} ({p['recommendation']['quantity_range']['min']}-{p['recommendation']['quantity_range']['max']} {p['unit']})
- Pattern: Trend {p['business_insights']['sales_patterns']['trend']}, Volatilitas {p['business_insights']['sales_patterns']['volatility']}
- Priority Score: {p['business_priority']['priority_score']}/100 (Tier: {p['business_priority']['priority_tier']})
"""
            batch_prompts.append(prompt)
        
        combined_prompt = f"""
Anda adalah ahli manajemen inventori untuk bisnis retail. Analisis {len(batch)} produk berikut dan berikan insight singkat untuk MASING-MASING produk dalam bahasa Indonesia.

{chr(10).join(batch_prompts)}

Untuk SETIAP produk, berikan analisis dalam format:

**[NAMA PRODUK]**
**Kesimpulan:** (1 kalimat - kondisi kritis atau aman?)
**Tindakan:** (1 kalimat - apa yang harus dilakukan sekarang?)
**Impact:** (1 kalimat - konsekuensi jika tidak bertindak?)

Gunakan bahasa praktis dan fokus pada keputusan bisnis. Prioritaskan produk risiko tinggi.
"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Anda adalah expert inventory manager yang memberikan rekomendasi praktis untuk pemilik usaha retail."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            llm_output = response.choices[0].message.content
            total_tokens += response.usage.total_tokens
            
            # Parse output
            lines = llm_output.split('\n')
            current_product = None
            current_text = []
            
            for line in lines:
                if line.strip().startswith('**[') or line.strip().startswith('**PRODUK'):
                    if current_product is not None:
                        results.append({
                            "product_id": batch[current_product]['product_id'],
                            "reasoning": '\n'.join(current_text).strip(),
                            "model_used": model,
                            "tokens_used": response.usage.total_tokens // len(batch)
                        })
                    current_product = len(results) if len(results) < len(batch) else 0
                    current_text = [line]
                else:
                    current_text.append(line)
            
            # Add last product
            if current_product is not None and current_text:
                results.append({
                    "product_id": batch[current_product]['product_id'],
                    "reasoning": '\n'.join(current_text).strip(),
                    "model_used": model,
                    "tokens_used": response.usage.total_tokens // len(batch)
                })
            
            print(f"  ✅ Batch {i//batch_size + 1} selesai ({len(batch)} produk)")
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"  ⚠️ Batch {i//batch_size + 1} gagal: {type(e).__name__}: {e}")
            # Fallback to rule-based
            for p in batch:
                results.append({
                    "product_id": p['product_id'],
                    "reasoning": generate_rule_based_fallback(p),
                    "model_used": "rule_based_fallback",
                    "tokens_used": 0
                })
    
    return results, total_tokens


def generate_rule_based_fallback(product: Dict[str, Any]) -> str:
    """Generate rule-based reasoning when LLM fails."""
    risk = product['stock_analysis']['risk_level']
    action = product['recommendation']['action']
    days = product['stock_analysis']['days_until_stockout']
    qty_range = product['recommendation']['quantity_range']
    
    if risk == "HIGH":
        return f"""**{product['product_name']}**
**Kesimpulan:** URGENT - Stok akan habis dalam {days} hari!
**Tindakan:** Segera restock {qty_range['min']}-{qty_range['max']} {product['unit']} untuk menghindari kekosongan stok.
**Impact:** Kehilangan penjualan dan pelanggan jika tidak segera direstock."""
    
    elif risk == "MEDIUM":
        return f"""**{product['product_name']}**
**Kesimpulan:** Stok mencukupi untuk {days} hari, tapi perlu monitoring.
**Tindakan:** Persiapkan restock {qty_range['min']}-{qty_range['max']} {product['unit']} dalam beberapa hari.
**Impact:** Risiko kekosongan jika permintaan tiba-tiba meningkat."""
    
    else:
        if action == "RESTOCK":
            return f"""**{product['product_name']}**
**Kesimpulan:** Stok saat ini aman untuk jangka pendek.
**Tindakan:** Restock {qty_range['min']}-{qty_range['max']} {product['unit']} untuk mempertahankan buffer stok.
**Impact:** Risiko minimal, lebih ke optimasi inventory."""
        else:
            return f"""**{product['product_name']}**
**Kesimpulan:** Stok sangat mencukupi untuk periode forecast.
**Tindakan:** Monitor saja, tidak perlu restock saat ini.
**Impact:** Tidak ada risiko jangka pendek."""


def generate_complete_forecast_with_batch_llm(
    df_ts,
    results,
    scores_df,
    use_llm: bool = True,
    llm_model: str = "deepseek-r1-distill-llama-70b",
    batch_size: int = 3,
    client: Optional[Groq] = None
) -> Dict[str, Any]:
    """
    Enhanced version dengan:
    - Batch LLM processing
    - Pattern analysis
    - Business priority scoring
    - Portfolio insights
    """
    from .forecast import (
        hybrid_forecast,
        estimate_days_until_stockout,
        calculate_risk_and_urgency,
        decide_restock_action,
        calculate_business_priority,
        generate_portfolio_insights
    )
    from .features import analyze_sales_patterns
    import math
    import numpy as np
    
    print("📊 Step 1: Generating forecasts dan analisis pattern...")
    output_products = []
    
    for pid in df_ts['product_id'].unique():
        g = df_ts[df_ts["product_id"] == pid].copy()
        current_stock = g["current_stock"].iloc[0] if len(g) > 0 else 0
        product_name = g["product_name"].iloc[0] if len(g) > 0 else pid
        unit = g["unit"].iloc[0] if len(g) > 0 else "unit"
        
        # Forecast
        forecast_result = hybrid_forecast(
            df_ts=df_ts,
            results=results,
            scores_df=scores_df,
            product_id=pid,
            horizon=14,
            use_ewma_fallback=True
        )
        
        forecast_14 = [math.ceil(x) for x in forecast_result['forecast'][:14]]
        total_14 = sum(forecast_14)
        avg_14 = round(np.mean(forecast_14), 2)
        
        # WAPE
        wape_val = np.nan
        if isinstance(scores_df, pd.DataFrame) and "product_id" in scores_df.columns:
            tmp = scores_df[scores_df["product_id"].astype(str).str.strip() == str(pid).strip()]
            if not tmp.empty:
                wape_val = tmp.iloc[0].get("wape", np.nan)
        
        # Stock analysis
        days_until_stockout = estimate_days_until_stockout(current_stock, forecast_14)
        risk_info = calculate_risk_and_urgency(
            days_until_stockout=days_until_stockout,
            forecast_mean=avg_14,
            wape=wape_val,
            current_stock=current_stock
        )
        
        # Restock decision
        restock_decision = decide_restock_action(
            current_stock=current_stock,
            forecast_total_14=total_14,
            forecast_mean=avg_14,
            risk_level=risk_info["risk_level"],
            days_until_stockout=days_until_stockout,
            safety_stock_multiplier=1.5
        )
        
        # Pattern Analysis
        sales_patterns = analyze_sales_patterns(df_ts, pid)
        
        product_obj = {
            "product_id": str(pid),
            "product_name": product_name,
            "unit": unit,
            "current_stock": int(current_stock),
            
            "forecast": {
                "horizon_days": 14,
                "daily": forecast_14,
                "total_demand": total_14,
                "average_per_day": avg_14,
                "method": forecast_result["method"],
                "confidence": forecast_result["confidence"]
            },
            
            "stock_analysis": {
                "days_until_stockout": days_until_stockout,
                "risk_level": risk_info["risk_level"],
                "urgency_score": risk_info["urgency_score"],
                "forecast_reliability": "HIGH" if pd.notna(wape_val) and wape_val < 0.15 else "MEDIUM" if pd.notna(wape_val) and wape_val < 0.30 else "LOW"
            },
            
            "recommendation": {
                "action": restock_decision["action"],
                "quantity_range": {
                    "min": restock_decision["qty_min"],
                    "max": restock_decision["qty_max"]
                },
                "reason": restock_decision["reason"]
            },
            
            "business_insights": {
                "sales_patterns": sales_patterns
            }
        }
        
        # Calculate Priority
        priority_score = calculate_business_priority(product_obj, sales_patterns)
        
        if priority_score >= 80:
            priority_tier = "CRITICAL"
        elif priority_score >= 60:
            priority_tier = "HIGH"
        elif priority_score >= 40:
            priority_tier = "MEDIUM"
        else:
            priority_tier = "LOW"
        
        product_obj["business_priority"] = {
            "priority_score": priority_score,
            "priority_tier": priority_tier
        }
        
        output_products.append(product_obj)
    
    print(f"✅ {len(output_products)} produk dianalisis")
    
    if use_llm:
        print(f"\n🤖 Step 2: Running batch LLM analysis (batch size={batch_size})...")
        try:
            # 1. Product-level analysis
            llm_results, total_tokens = generate_batch_llm_analysis(
                output_products, 
                model=llm_model,
                batch_size=batch_size,
                client=client
            )
            
            # Map results back to products
            for product in output_products:
                matching_result = next(
                    (r for r in llm_results if r['product_id'] == product['product_id']), 
                    None
                )
                if matching_result:
                    product["ai_insights"] = {
                        "reasoning": matching_result['reasoning'],
                        "model": matching_result['model_used'],
                        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            
            llm_success = sum(1 for r in llm_results if r['model_used'] != 'rule_based_fallback')
            fallback_count = len(llm_results) - llm_success

            # 2. Portfolio-level analysis (NEW)
            print(f"\n🤖 Step 2b: Running portfolio LLM analysis...")
            portfolio_ai_summary, summary_tokens = generate_portfolio_llm_analysis(
                output_products,
                model=llm_model,
                client=client
            )
            total_tokens += summary_tokens

        except Exception as e:
            print(f"⚠️ LLM analysis failed: {e}")
            total_tokens = 0
            llm_success = 0
            fallback_count = len(output_products)
            portfolio_ai_summary = None
    else:
        total_tokens = 0
        llm_success = 0
        fallback_count = 0
        portfolio_ai_summary = None
    
    # Step 3: Generate Portfolio Insights
    print("\n📈 Step 3: Generating portfolio insights...")
    portfolio_insights = generate_portfolio_insights(output_products)
    
    # Add AI summary to portfolio insights
    if portfolio_ai_summary:
        portfolio_insights["ai_summary"] = portfolio_ai_summary

    # Final JSON
    final_json = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "hybrid_sarimax_ewma_v3.0_batch_priority",
        "total_products": len(output_products),
        "llm_enabled": use_llm,
        "llm_success_count": llm_success,
        "fallback_count": fallback_count,
        "total_tokens_used": total_tokens,
        
        "portfolio_insights": portfolio_insights,
        
        "products": sorted(
            output_products, 
            key=lambda x: x['business_priority']['priority_score'], 
            reverse=True
        )
    }
    
    return final_json


def generate_portfolio_llm_analysis(
    products_data: List[Dict[str, Any]], 
    model: str = "deepseek-r1-distill-llama-70b", 
    client: Optional[Groq] = None
) -> tuple[Dict[str, Any], int]:
    """
    Generate high-level portfolio insights summary.
    """
    if client is None:
        client = get_llm_client()

    try:
        # Filter critical/high priority items for the summary to save tokens and focus attention
        priority_items = [
            p for p in products_data 
            if p['business_priority']['priority_tier'] in ['CRITICAL', 'HIGH']
        ]
        
        # If too few high priority, add some medium
        if len(priority_items) < 3:
            medium_items = [
                p for p in products_data 
                if p['business_priority']['priority_tier'] == 'MEDIUM'
            ][:3]
            priority_items.extend(medium_items)
        
        # Prepare summary data for prompt
        items_summary = []
        for p in priority_items:
            # Safe access to nested keys
            trend = p.get('business_insights', {}).get('sales_patterns', {}).get('trend', 'unknown')
            items_summary.append(
                f"- {p['product_name']}: Risk {p['stock_analysis']['risk_level']}, "
                f"Trend {trend}, "
                f"Action {p['recommendation']['action']} ({p['recommendation']['reason']})"
            )
        
        overall_stats = {
            "total": len(products_data),
            "high_risk": sum(1 for p in products_data if p['stock_analysis']['risk_level'] == 'HIGH'),
            "restock_needed": sum(1 for p in products_data if p['recommendation']['action'] == 'RESTOCK')
        }

        prompt = f"""
Anda adalah asisten AI untuk manajemen inventori. Berikan "AI Insight Summary" level portofolio singkat berdasarkan data berikut:

STATISTIK:
Total Produk: {overall_stats['total']}
High Risk: {overall_stats['high_risk']}
Perlu Restock: {overall_stats['restock_needed']}

DETAIL PRODUK PRIORITAS TINGGI:
{chr(10).join(items_summary)}

TUGAS ANDA:
Buat ringkasan eksekutif dalam JSON format dengan struktur persis seperti ini:
{{
  "pattern_trend_summary": "Satu kalimat ringkas menggabungkan observasi pola (Pattern) dan tren (Trend) pasar secara umum.",
  "priority_actions": {{
      "urgent": "Satu kalimat tindakan untuk tier Urgent/Critical (sebutkan nama produk jika perlu).",
      "medium": "Satu kalimat tindakan untuk tier Medium (sebutkan nama produk jika perlu).",
      "low": "Satu kalimat tindakan untuk tier Low atau monitoring."
  }}
}}

CONTOH OUTPUT YANG DIHARAPKAN (isi konten sesuaikan data):
{{
  "pattern_trend_summary": "Pattern: Stable demand with weekend peaks • Trend: 12% increase in dairy sales",
  "priority_actions": {{
      "urgent": "Urgent: Restock Dark Chocolate Bar and Greek Yogurt (critical stock)",
      "medium": "Medium: Consider restocking dairy alternatives before weekend",
      "low": "Low: Monitor organic produce inventory for seasonal changes"
  }}
}}

PENTING:
- Gunakan Bahasa Bahasa Indonesia
- Gabungkan "Pattern: [Pola]" dan "Trend: [Trend]" dengan separator '•'.
- Jangan bertele-tele.
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful inventory AI assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        import json
        result_json = json.loads(content)
        
        return result_json, response.usage.total_tokens

    except Exception as e:
        print(f"⚠️ Portfolio summary generation failed: {e}")
        return generate_rule_based_portfolio_summary(products_data), 0


def generate_rule_based_portfolio_summary(products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate rule-based summary when LLM fails."""
    
    # 1. Calculate stats
    high_risk_count = sum(1 for p in products_data if p['stock_analysis']['risk_level'] == 'HIGH')
    restock_count = sum(1 for p in products_data if p['recommendation']['action'] == 'RESTOCK')
    
    # 2. Get top priority items per tier
    sorted_products = sorted(
        products_data, 
        key=lambda x: x['business_priority']['priority_score'], 
        reverse=True
    )
    
    urgent_items = [p for p in sorted_products if p['business_priority']['priority_tier'] in ['CRITICAL', 'HIGH']]
    medium_items = [p for p in sorted_products if p['business_priority']['priority_tier'] == 'MEDIUM']
    low_items = [p for p in sorted_products if p['business_priority']['priority_tier'] == 'LOW']
    
    # 3. Construct Summary Strings
    pattern_summary = "Pattern: Stable demand with localized peaks"
    trend_summary = f"Trend: {restock_count} products need restocking"
    
    # 4. Construct Actions
    actions = {}
    
    if urgent_items:
        p = urgent_items[0]
        actions["urgent"] = f"Urgent: Restock {p['product_name']} ({p['stock_analysis']['days_until_stockout']} days left)"
    else:
        actions["urgent"] = "Urgent: No critical items at the moment"
        
    if medium_items:
        p = medium_items[0]
        actions["medium"] = f"Medium: Monitor {p['product_name']} for {p['recommendation']['action'].lower()}"
    else:
        actions["medium"] = "Medium: Review safety stock levels"
        
    if low_items:
        actions["low"] = "Low: Routine inventory check"
    else:
        actions["low"] = "Low: All items are high priority"

    return {
        "pattern_trend_summary": f"{pattern_summary} • {trend_summary}",
        "priority_actions": actions
    }

