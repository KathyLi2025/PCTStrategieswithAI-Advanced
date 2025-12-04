import os
import sys
import json
import datetime
import pdfplumber
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ProjectConfig import configClass
from COTPctTrends.era_cot_trainer import ERACoTProcessor, PatentBertaLLMClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def analyze_samples():
    sample_dir = configClass.SAMPLE_DIR
    output_file = os.path.join(configClass.PATENT_SUMARY_DATA_PATH)
    
    # Initialize Model
    model_path = configClass.MODEL_NAME_BASE_GPT2
    logger.info(f"Loading model from {model_path}...")
    client = PatentBertaLLMClient(model_path)
    processor = ERACoTProcessor(client)
    
    results = []
    
    # Iterate over PDFs
    for filename in os.listdir(sample_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(sample_dir, filename)
            logger.info(f"Processing {filename}...")
            
            text = extract_text_from_pdf(pdf_path)
            if not text:
                continue
            
            # Truncate text for the model if too long (simple truncation)
            # The model has a limit, and we want to capture the essence.
            # Usually the abstract/summary is at the beginning.
            truncated_text = text[:800] 
            
            # Run ERA-CoT Pipeline
            analysis_result = processor.run_pipeline(truncated_text)
            
            # Construct Output Object
            # We map the ERA-CoT output to the requested JSON structure
            
            # Infer status from decision
            decision_text = analysis_result['decision'].lower()
            status = "Projected Profit" if "apply" in decision_text else "Projected Loss"
            
            roi_data = {
                "meta_id": filename, # Use filename as ID
                "timestamp": datetime.date.today().isoformat(),
                "input_context": {
                    "patent_summary": text[:500] + "...", # First 500 chars as summary
                    "market_trend_snapshot": {
                        "pct_category_growth_rate_3y": "N/A (Requires Market Data)",
                        "competitor_activity_index": "N/A",
                        "tech_lifecycle_stage": "N/A",
                        "key_pain_point": "Extracted from text: " + analysis_result['entities'][:50] # Mocking extraction
                    }
                },
                "financial_outcome": {
                    "status": status,
                    "actual_roi": "Predicted",
                    "note": analysis_result['decision']
                },
                "preference_pair": {
                    "chosen": {
                        "rationale": analysis_result['decision'],
                        "score": 0.8 # Placeholder
                    },
                    "rejected": {
                        "rationale": "Alternative decision rejected based on reasoning.",
                        "score": 0.2 # Placeholder
                    }
                },
                "era_cot_details": {
                    "entities": analysis_result['entities'],
                    "explicit_relations": analysis_result['explicit_relations'],
                    "implicit_relations": analysis_result['implicit_relations'],
                    "scored_relations": analysis_result['scored_relations']
                }
            }
            
            results.append(roi_data)
            
    # Save Results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    analyze_samples()
