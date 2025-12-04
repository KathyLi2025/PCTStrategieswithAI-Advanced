import argparse
import sys
import os
import logging

# Add src to path so imports work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.DPOFineTuning import train_pct_decision_agent
from src import analyze_patent_samples_eracot
from src import predict_pct_roi

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="PCT Strategies with AI - Advanced Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train the PCT Decision Agent using DPO")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run the full inference pipeline (ERA-CoT + Decision Agent)")

    args = parser.parse_args()

    if args.command == "train":
        logger.info("Starting DPO Fine-tuning...")
        train_pct_decision_agent.main()
        logger.info("Training completed.")

    elif args.command == "run":
        logger.info("Starting Inference Pipeline...")
        
        # Step 1: ERA-CoT Summarization
        logger.info("Step 1: Running ERA-CoT Analysis on Patent Samples...")
        analyze_patent_samples_eracot.analyze_samples()
        
        # Step 2: Decision Inference
        logger.info("Step 2: Predicting PCT Decisions based on ROI Analysis...")
        predict_pct_roi.main()
        
        logger.info("Inference Pipeline completed.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()