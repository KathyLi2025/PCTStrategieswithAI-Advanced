import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ProjectConfig import configClass


def load_agent(model_path):
    """
    Load the trained PCT Decision Agent.
    """
    print(f"Loading agent from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval() # Set to evaluation mode
    
    return model, tokenizer, device

def predict_decision(model, tokenizer, device, patent_summary, market_trends):
    """
    Generate a strategic decision based on the patent and market trends.
    """
    # 1. Construct the Prompt (Must match training format EXACTLY)
    prompt = (
        f"[Decision Context]\n"
        f"Date: 2025-12-04\n" # Current date for inference
        f"Patent Summary: {patent_summary}\n"
        f"Market Trends Snapshot:\n"
        f"- Growth Rate (3y): {market_trends.get('growth_rate', 'N/A')}\n"
        f"- Competitor Activity: {market_trends.get('competitor_activity', 'N/A')}\n"
        f"- Lifecycle Stage: {market_trends.get('lifecycle_stage', 'N/A')}\n"
        f"- Key Focus/Risk: {market_trends.get('key_focus_risk', 'N/A')}\n\n"
        f"Task: Based on the historical ROI analysis, provide a strategic PCT application decision.\n"
        f"Analysis:"
    )
    
    # 2. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 3. Generate
    # We use greedy decoding or low temperature to get the most "confident" strategic decision
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,      # Allow enough space for reasoning
            temperature=0.1,         # Low temp for deterministic, risk-averse output
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # 4. Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the newly generated part (Analysis)
    analysis_part = full_response[len(prompt):].strip()
    
    return analysis_part

def main():
    # Path to your trained model
    # Use configClass to get the correct path
    model_path = configClass.PCT_AGENT_OUTPUT_AGENT_V1

    try:
        model, tokenizer, device = load_agent(model_path)
    except OSError:
        print(f"Error: Could not find model at {model_path}. Please run the training script first.")
        return

    # Path to the input JSON file
    input_file_path = os.path.join(os.path.dirname(__file__), 'DataCenter', 'PCTReview', 'Patent_extract_for_roi_predict.json')

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    print(f"Reading data from {input_file_path}...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} patents for prediction.")
    results = []

    for item in data:
        meta_id = item.get('meta_id', 'Unknown')
        input_context = item.get('input_context', {})
        patent_summary = input_context.get('patent_summary', '')
        market_snapshot = input_context.get('market_trend_snapshot', {})

        # Map JSON fields to function arguments
        trends = {
            "growth_rate": market_snapshot.get('pct_category_growth_rate_3y', 'N/A'),
            "competitor_activity": market_snapshot.get('competitor_activity_index', 'N/A'),
            "lifecycle_stage": market_snapshot.get('tech_lifecycle_stage', 'N/A'),
            "key_focus_risk": market_snapshot.get('key_pain_point', 'N/A')
        }

        print(f"\nProcessing {meta_id}...")
        decision = predict_decision(model, tokenizer, device, patent_summary, trends)
        print(f"Decision: {decision[:100]}...") # Print start of decision

        results.append({
            "meta_id": meta_id,
            "predicted_decision": decision
        })

    # Save results
    output_file_path = os.path.join(os.path.dirname(__file__), 'DataCenter', 'PCTReview', 'predicted_roi_results.json')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nPredictions saved to {output_file_path}")

if __name__ == "__main__":
    main()
