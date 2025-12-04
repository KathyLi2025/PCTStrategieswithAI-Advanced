import re
import sys
import os
from typing import List, Dict, Any

# Add parent directory to sys.path to allow importing ProjectConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import pipeline
from ProjectConfig import configClass

class ERACoTPrompts:
    """
    Prompts derived from the ERA-CoT paper (Appendix E).
    """
    
    ENTITIES_EXTRACTION = """
Entities Extraction
Given a sentence (or text), possible entities may include: [individuals, organizations, locations, technical terms, components, metrics, percentages]. Find all entities based on the provided text.

Text: {text}
Entities:
"""

    EXPLICIT_RELATION_EXTRACTION = """
Entities Relation Extraction
Given a text, and all entities within the text. Extract all relationships between entities which directly stated in the text.
Every relationship stated as a triple: (Entity A, Entity B, Relation)

Text: {text}
Entities: {entities}
Relationships:
"""

    IMPLICIT_RELATION_INFERENCE = """
Entities Relation Inference
Given a text, all entities, and all explicit relationships within the text. Infer all possible implicit relationships between entities.
For each pair of entities, infer up to 3 implicit relationships.
Every relationship stated as a triple: (Entity A, Entity B, Relation)

Text: {text}
Entities: {entities}
Explicit Relationships: {explicit_relations}
Implicit Relationships:
"""

    RELATIONSHIP_DISCRIMINATION = """
Relationship Discrimination
Given a text, and all uncertain relationships (implicit ones) within the text. Score the confidence level of each relationship.
The confidence score ranges from 0 to 10, where a higher score indicates a higher likelihood of the relationship being correct.
Filter out relationships with a score below 6.

Every relationship stated as a triple: (Entity A, Entity B, Relation)

Text: {text}
Entities: {entities}
Uncertain Relationships: {implicit_relations}
Scores:
"""

    DECISION_MAKING = """
Decision Making (Question Answering)
Given the text, all entities, and all validated relationships (explicit + high confidence implicit). Make a strategic decision.

Text: {text}
Entities: {entities}
Relationships: {all_relations}

Question: {question}
Answer (Decision and Reasoning):
"""

class ERACoTProcessor:
    def __init__(self, llm_client):
        """
        Initialize with an LLM client.
        The llm_client should have a method `generate(prompt: str) -> str`.
        """
        self.llm = llm_client

    def _truncate(self, text: str, max_chars: int = 400) -> str:
        """Helper to truncate text to avoid context window overflow."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...(truncated)"

    def extract_entities(self, text: str) -> str:
        prompt = ERACoTPrompts.ENTITIES_EXTRACTION.format(text=text)
        return self.llm.generate(prompt)

    def extract_explicit_relations(self, text: str, entities: str) -> str:
        prompt = ERACoTPrompts.EXPLICIT_RELATION_EXTRACTION.format(
            text=text, 
            entities=self._truncate(entities)
        )
        return self.llm.generate(prompt)

    def infer_implicit_relations(self, text: str, entities: str, explicit_relations: str) -> str:
        prompt = ERACoTPrompts.IMPLICIT_RELATION_INFERENCE.format(
            text=text, 
            entities=self._truncate(entities), 
            explicit_relations=self._truncate(explicit_relations)
        )
        return self.llm.generate(prompt)

    def discriminate_relationships(self, text: str, entities: str, implicit_relations: str) -> str:
        prompt = ERACoTPrompts.RELATIONSHIP_DISCRIMINATION.format(
            text=text, 
            entities=self._truncate(entities), 
            implicit_relations=self._truncate(implicit_relations)
        )
        return self.llm.generate(prompt)

    def make_decision(self, text: str, entities: str, explicit_relations: str, validated_implicit_relations: str, question: str) -> str:
        # Combine relations
        all_relations = f"Explicit: {self._truncate(explicit_relations)}\nImplicit (High Confidence): {self._truncate(validated_implicit_relations)}"
        
        prompt = ERACoTPrompts.DECISION_MAKING.format(
            text=text,
            entities=self._truncate(entities),
            all_relations=all_relations,
            question=question
        )
        return self.llm.generate(prompt)

    def run_pipeline(self, text: str, question: str = "Should we apply for a PCT patent for this invention? Provide a recommendation (APPLY/REJECT) and reasoning.") -> Dict[str, Any]:
        """
        Executes the full ERA-CoT pipeline.
        """
        print("--- Step 1: Extracting Entities ---")
        entities = self.extract_entities(text)
        print(f"Entities: {entities}\n")

        print("--- Step 2: Extracting Explicit Relations ---")
        explicit_relations = self.extract_explicit_relations(text, entities)
        print(f"Explicit Relations: {explicit_relations}\n")

        print("--- Step 3: Inferring Implicit Relations ---")
        implicit_relations = self.infer_implicit_relations(text, entities, explicit_relations)
        print(f"Implicit Relations (Raw): {implicit_relations}\n")

        print("--- Step 4: Discriminating Relationships ---")
        scored_relations = self.discriminate_relationships(text, entities, implicit_relations)
        # In a real implementation, we would parse the scores and filter. 
        # For this template, we assume the LLM output includes the filtering or we pass the raw output.
        print(f"Relationship Scores: {scored_relations}\n")

        print("--- Step 5: Making Decision ---")
        decision = self.make_decision(text, entities, explicit_relations, scored_relations, question)
        print(f"Decision: {decision}\n")

        return {
            "entities": entities,
            "explicit_relations": explicit_relations,
            "implicit_relations": implicit_relations,
            "scored_relations": scored_relations,
            "decision": decision
        }

class PatentBertaLLMClient:
    def __init__(self, model_path):
        model_path= model_path
        
        self.generator = pipeline("text-generation", model=model_path, max_new_tokens=256)

    def generate(self, prompt: str) -> str:
        # Generate response and strip the input prompt from the output
        # Add repetition penalty and ngram blocking to prevent loops
        # Ensure the prompt is not too long for the model (distilgpt2 has 1024 limit)
        # We'll rely on the pipeline's truncation, but also add generation constraints
        output = self.generator(
            prompt, 
            num_return_sequences=1, 
            truncation=True, 
            max_length=1024,
            max_new_tokens=128,
            pad_token_id=50256, # EOS token for GPT2
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            do_sample=True,
            temperature=0.7
        )
        return output[0]['generated_text'][len(prompt):].strip()

if __name__ == "__main__":
    # Example text from the sample PDF
    sample_text = """
    The present invention provides an automated storage and retrieval system comprising: a track system...
    wherein the container handling vehicle comprises an auxiliary battery to allow movement of the container handling vehicle when void of the replaceable power supply.
    """
    
    # Initialize the processor with the mock client
    # In production, replace MockLLMClient() with a real client wrapper.
    # Implementation of a real LLM client using Hugging Face Transformers

    model_path = configClass.MODEL_NAME_BASE_GPT2
    client = PatentBertaLLMClient(model_path)
    processor = ERACoTProcessor(client)
    
    # Run the pipeline
    result = processor.run_pipeline(sample_text)
