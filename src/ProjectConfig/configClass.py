import os
import pathlib

# Configuration
OFFLINE_MODEL = 1  # Set to 1 for offline mode (use local models only)
OFFLINE = bool(OFFLINE_MODEL)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW_DIR = os.path.join(BASE_DIR, 'DataCenter', 'PCTReview')
SAMPLE_DIR = os.path.join(BASE_DIR, 'DataCenter', 'Sample')
PCT_REFERENCE_DIR = os.path.join(BASE_DIR, 'DataCenter', 'Reference')
 
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#COT
USE_ADVANCED_COT = True # Enable the new CoT + RoFormer engine

#DPO Training
if OFFLINE:
    print("Offline mode is enabled: Models will be loaded from local paths only.")
    MODEL_NAME_BASE_GPT2 = str(pathlib.Path.home() / "models" / "distilgpt2")
else:
    MODEL_NAME_BASE_GPT2 = "distilgpt2"


ROI_DATA_PATH = os.path.join(PCT_REFERENCE_DIR, "roi_training_data.json")
PATENT_SUMARY_DATA_PATH = os.path.join(REVIEW_DIR, "Patent_extract_for_roi_predict.json")

PCT_AGENT_OUTPUT_AGENT_V1 = os.path.join(BASE_DIR, 'FineTuningModels',"pct_decision_agent_v1")
