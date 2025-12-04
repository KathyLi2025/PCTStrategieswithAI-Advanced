import sys
import os
import torch
from torch.utils.data import DataLoader


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ProjectConfig import configClass
from DPOFineTuning.pct_decision_trainer import PCTDecisionTrainer
from DPOFineTuning.data_loader import ROIPreferenceDataset




def main():
    print("==================================================")
    print("   PCT Decision Agent - ROI Alignment Training    ")
    print("==================================================")
    
    # 1. Configuration
    # Using 'distilgpt2' as a lightweight proxy for the base model.
    # In production, this would be your SFT-tuned Llama-3 or Mistral.
     
    model_name = configClass.MODEL_NAME_BASE_GPT2
    data_path = configClass.ROI_DATA_PATH
    output_dir = configClass.PCT_AGENT_OUTPUT_AGENT_V1
    
    # 2. Initialize Trainer
    # beta=0.1 is standard. 
    # If we want the model to be very risk-averse (stick to ROI data), we might lower beta to 0.05.
    trainer = PCTDecisionTrainer(model_name_or_path=model_name, beta=0.1, lr=5e-6)
    
    # 3. Load Data
    print(f"Loading ROI training data from {data_path}...")
    dataset = ROIPreferenceDataset(data_path, trainer.tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # Batch size 1 for demo stability
    
    print(f"Loaded {len(dataset)} historical decision cases.")
    
    # 4. Training Loop
    print("\nStarting DPO Training (Value Alignment)...")
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        total_reward_gap = 0
        
        for step, batch in enumerate(dataloader):
            loss, reward_chosen, reward_rejected = trainer.train_step(batch)
            total_loss += loss
            total_reward_gap += (reward_chosen - reward_rejected)
            
            print(f"Epoch {epoch+1} | Step {step+1} | Loss: {loss:.4f} | "
                  f"Reward Gap: {reward_chosen - reward_rejected:.4f} "
                  f"(Chosen: {reward_chosen:.3f}, Rejected: {reward_rejected:.3f})")
        
        avg_loss = total_loss / len(dataloader)
        avg_gap = total_reward_gap / len(dataloader)
        print(f">>> Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}, Avg Reward Gap: {avg_gap:.4f}")
        print("-" * 50)
        
    # 5. Save the Aligned Agent
    print(f"Saving ROI-Aligned Agent to {output_dir}...")
    trainer.save_model(output_dir)
    print("Training Complete. The agent has now internalized the historical ROI logic.")

if __name__ == "__main__":
    main()
