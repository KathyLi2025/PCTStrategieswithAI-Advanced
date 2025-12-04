import json
import torch
from torch.utils.data import Dataset

class ROIPreferenceDataset(Dataset):
    """
    Dataset for ROI-based DPO training.
    Parses the complex JSON schema into Prompt + Chosen + Rejected format.
    """
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.data = self._process_data(raw_data)

    def _process_data(self, raw_data):
        processed = []
        for item in raw_data:
            # 1. Construct the Prompt (x)
            # We combine the Patent Summary and the Market Trend Snapshot into a structured context.
            context = item["input_context"]
            trends = context["market_trend_snapshot"]
            
            prompt = (
                f"[Decision Context]\n"
                f"Date: {item['timestamp']}\n"
                f"Patent Summary: {context['patent_summary']}\n"
                f"Market Trends Snapshot:\n"
                f"- Growth Rate (3y): {trends.get('pct_category_growth_rate_3y', 'N/A')}\n"
                f"- Competitor Activity: {trends.get('competitor_activity_index', 'N/A')}\n"
                f"- Lifecycle Stage: {trends.get('tech_lifecycle_stage', 'N/A')}\n"
                f"- Key Focus/Risk: {trends.get('dominant_players_focus') or trends.get('key_pain_point') or trends.get('litigation_risk') or 'N/A'}\n\n"
                f"Task: Based on the historical ROI analysis, provide a strategic PCT application decision.\n"
                f"Analysis:"
            )
            
            # 2. Extract Chosen (yw) and Rejected (yl)
            # Note: The 'reasoning' in the JSON is the core value we want to learn.
            chosen_text = f" {item['preference_pair']['chosen']['reasoning']}"
            rejected_text = f" {item['preference_pair']['rejected']['reasoning']}"
            
            processed.append({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text
            })
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenization logic (Standard DPO formatting)
        def tokenize_pair(p, r):
            full_text = p + r
            tokenized = self.tokenizer(
                full_text, 
                truncation=True, 
                max_length=self.max_length, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            prompt_tokenized = self.tokenizer(
                p, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            prompt_len = prompt_tokenized.input_ids.shape[1]
            
            labels = tokenized.input_ids.clone()
            # Mask prompt
            if prompt_len < labels.shape[1]:
                labels[:, :prompt_len] = -100
            
            # Mask padding
            labels[tokenized.attention_mask == 0] = -100
            
            return tokenized.input_ids.squeeze(0), tokenized.attention_mask.squeeze(0), labels.squeeze(0)

        chosen_input_ids, chosen_mask, chosen_labels = tokenize_pair(prompt, chosen)
        rejected_input_ids, rejected_mask, rejected_labels = tokenize_pair(prompt, rejected)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_labels
        }
