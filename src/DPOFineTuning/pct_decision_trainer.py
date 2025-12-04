import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

class PCTDecisionTrainer:
    """
    A specialized DPO Trainer for the PCT Decision Agent.
    It optimizes the policy to align with historical ROI outcomes.
    """
    def __init__(self, model_name_or_path, beta=0.1, lr=1e-5, device=None):
        self.beta = beta
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Policy Model: {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Policy Model (The Agent we are training)
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.policy_model.train()
        
        # Reference Model (The Baseline / Risk-Neutral Observer)
        # In a real scenario, this would be the SFT model before DPO.
        print(f"Loading Reference Model (Frozen)...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)

    def get_batch_logps(self, model, input_ids, attention_mask, labels):
        """
        Compute log probabilities of the response tokens.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        loss_mask = (shift_labels != -100)
        safe_labels = shift_labels.clone()
        safe_labels[~loss_mask] = 0
        
        per_token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        
        sum_logps = (per_token_logps * loss_mask).sum(dim=1)
        return sum_logps

    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # Move batch to device
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)
        
        # 1. Policy LogPs
        policy_chosen_logps = self.get_batch_logps(self.policy_model, chosen_input_ids, chosen_mask, chosen_labels)
        policy_rejected_logps = self.get_batch_logps(self.policy_model, rejected_input_ids, rejected_mask, rejected_labels)
        
        # 2. Reference LogPs (No Grad)
        with torch.no_grad():
            ref_chosen_logps = self.get_batch_logps(self.ref_model, chosen_input_ids, chosen_mask, chosen_labels)
            ref_rejected_logps = self.get_batch_logps(self.ref_model, rejected_input_ids, rejected_mask, rejected_labels)
            
        # 3. DPO Loss Calculation
        # Reward_chosen = beta * (policy_chosen - ref_chosen)
        # Reward_rejected = beta * (policy_rejected - ref_rejected)
        # Loss = -log(sigmoid(Reward_chosen - Reward_rejected))
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = pi_logratios - ref_logratios
        
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()
        
        # 4. Backprop
        loss.backward()
        self.optimizer.step()
        
        # Return metrics for monitoring
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
        
        return loss.item(), chosen_rewards.mean().item(), rejected_rewards.mean().item()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.policy_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
