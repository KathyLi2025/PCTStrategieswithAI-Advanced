# PCTStrategieswithAI-Advanced - Patent PCT Feasibility Analysis Tool

## Overview

PCTStrategieswithAI is a Python-based tool that analyzes patent documents to determine their feasibility for Patent Cooperation Treaty (PCT) applications. It leverages a two-stage pipeline:
1.  **ERA-CoT (Entity-Relationship Analysis & Chain-of-Thought)**: Extracts insights and summarizes patent data.
2.  **DPO Decision Agent**: A fine-tuned model that predicts the Return on Investment (ROI) and makes PCT filing decisions based on the analyzed data.

### Key Features
- **Dual-Mode Operation**: Train the decision agent or run the full inference pipeline.
- **ERA-CoT Analysis**: Uses advanced prompting strategies to analyze patent trends and samples.
- **DPO Fine-Tuning**: Includes a Direct Preference Optimization (DPO) training loop to refine the decision agent's judgment.
- **Offline Support**: Configurable to run with local models.

## Requirements

- Python 3.8+
- Dependencies: `torch`, `transformers`, `pandas`, `numpy`, `scipy`, `pdfplumber`
- Optional: NVIDIA GPU with CUDA for faster processing

## Installation

1. **Clone or Download the Project**:
   - Place the `PCTStrategieswithAI-Advanced` folder in your desired location.

2. **Install Dependencies**:
   ```bash
   cd PCTStrategieswithAI-Advanced
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place  DPO fine-tuning reference in  `./DataCenter/Reference/`.
   - Place COT trends sumaries in `./DataCenter/PCTReview/`.
   - Place  sample patent PDFs in `./DataCenter/Sample/`.

## Usage

The project is controlled via the `main.py` script with two primary commands:

### 1. Train the Decision Agent
Fine-tunes the decision model using Direct Preference Optimization (DPO).
**Output**:
-  `DataCenter/FineTuningModels/pct_decision_agent_v1`:  Fine-tuning model using DPO

```bash
python main.py train
```


### 2. Run Inference Pipeline
Executes the full analysis workflow:
1.  **ERA-CoT Analysis**: Summarizes patent samples and extracts key features.
2.  **ROI Prediction**: Uses the fine-tuned decision agent to predict PCT feasibility.
**Output**:
- `DataCenter/Output/predicted_roi_results.json`: Final predictions and ROI analysis.


```bash
python main.py run
```

## Configuration

Configuration settings (paths, model parameters, etc.) are managed in:
- `ProjectConfig/configClass.py`

## Module Structure

- `main.py`: Central CLI entry point.
- `src/`:
    - `analyze_patent_samples_eracot.py`: Handles the ERA-CoT analysis phase.
    - `predict_pct_roi.py`: Runs the decision agent for ROI prediction.
    - `DPOFineTuning/`: Contains logic for training the decision agent.
    - `COTPctTrends/`: Core logic for Chain-of-Thought trend analysis.
- `DataCenter/`: Stores input data (PDFs) and output results.

## License

This project is for educational/research purposes. Ensure compliance with model licenses (Hugging Face, etc.).
