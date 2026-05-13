"""
Passkey Retrieval Benchmark.

Tests the model's ability to retrieve a hidden "passkey" from
a long context with filler text.

Standard task: Hide a random number in a long document, ask model to retrieve it.
"""

import torch
import random
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PasskeyExample:
    """Single passkey example."""
    context: str
    passkey: str
    query: str
    answer: str
    context_length: int
    passkey_position: int  # Position in tokens (0-1 normalized)


class PasskeyBenchmark:
    """
    Passkey retrieval benchmark.
    
    Generates synthetic long-context examples with hidden passkeys
to test retrieval capability under different retention policies.
    """
    
    # Filler text (repeated to create long context)
    FILLER_TEXT = """
The quick brown fox jumps over the lazy dog. This is filler text used to create
long context lengths for testing. The grass is green and the sky is blue.
Mountains rise in the distance and rivers flow through valleys.
""".strip()
    
    def __init__(
        self,
        tokenizer,
        context_lengths: List[int] = [1024, 4096, 16384, 32768, 65536],
        passkey_positions: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        num_samples_per_config: int = 10,
    ):
        """
        Initialize passkey benchmark.
        
        Args:
            tokenizer: Tokenizer for the model
            context_lengths: Context lengths to test
            passkey_positions: Where to hide passkey (0=start, 1=end)
            num_samples_per_config: Samples per (length, position) config
        """
        self.tokenizer = tokenizer
        self.context_lengths = context_lengths
        self.passkey_positions = passkey_positions
        self.num_samples_per_config = num_samples_per_config
    
    def generate_passkey(self) -> str:
        """Generate random passkey (5-digit number)."""
        return "".join([str(random.randint(0, 9)) for _ in range(5)])
    
    def create_example(
        self,
        target_length: int,
        passkey_position: float,
    ) -> PasskeyExample:
        """
        Create a single passkey example.
        
        Args:
            target_length: Target context length in tokens
            passkey_position: Where to hide passkey (0-1)
            
        Returns:
            PasskeyExample
        """
        passkey = self.generate_passkey()
        
        # Create passkey sentence
        passkey_sentence = f"The pass key is {passkey}. Remember it."
        
        # Calculate how much filler we need
        passkey_tokens = len(self.tokenizer.encode(passkey_sentence))
        filler_tokens_needed = target_length - passkey_tokens - 50  # Buffer
        
        # Tokenize filler and repeat
        filler_tokens = self.tokenizer.encode(self.FILLER_TEXT)
        repeats_needed = filler_tokens_needed // len(filler_tokens) + 1
        
        # Insert passkey at specified position
        all_filler = (self.FILLER_TEXT + " ") * repeats_needed
        filler_before = all_filler[:int(len(all_filler) * passkey_position)]
        filler_after = all_filler[int(len(all_filler) * passkey_position):]
        
        context = filler_before + " " + passkey_sentence + " " + filler_after
        
        # Trim to target length
        context_tokens = self.tokenizer.encode(context)
        if len(context_tokens) > target_length:
            context_tokens = context_tokens[:target_length]
            context = self.tokenizer.decode(context_tokens)
        
        query = "What is the pass key? The pass key is"
        
        return PasskeyExample(
            context=context,
            passkey=passkey,
            query=query,
            answer=passkey,
            context_length=len(context_tokens),
            passkey_position=passkey_position,
        )
    
    def generate_dataset(self) -> List[PasskeyExample]:
        """Generate full passkey dataset."""
        examples = []
        
        for length in self.context_lengths:
            for position in self.passkey_positions:
                for _ in range(self.num_samples_per_config):
                    example = self.create_example(length, position)
                    examples.append(example)
        
        logger.info(f"Generated {len(examples)} passkey examples")
        return examples
    
    def evaluate(
        self,
        runner,
        examples: List[PasskeyExample],
    ) -> Dict:
        """
        Evaluate a runner on passkey retrieval.
        
        Args:
            runner: Runner with .run() method
            examples: Passkey examples
            
        Returns:
            Results dict
        """
        correct = 0
        total = 0
        results_by_length = {length: {"correct": 0, "total": 0} 
                            for length in self.context_lengths}
        results_by_position = {pos: {"correct": 0, "total": 0} 
                              for pos in self.passkey_positions}
        
        for example in examples:
            # Run generation
            context_ids = self.tokenizer.encode(example.context, return_tensors="pt")
            query_ids = self.tokenizer.encode(example.query, return_tensors="pt")
            
            generated_ids, _ = runner.run(context_ids, query_ids, max_new_tokens=10)
            generated_text = self.tokenizer.decode(generated_ids)
            
            # Check if passkey is in output
            is_correct = example.passkey in generated_text
            
            correct += int(is_correct)
            total += 1
            
            results_by_length[example.context_length]["correct"] += int(is_correct)
            results_by_length[example.context_length]["total"] += 1
            results_by_position[example.passkey_position]["correct"] += int(is_correct)
            results_by_position[example.passkey_position]["total"] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "by_length": {
                k: v["correct"] / v["total"] if v["total"] > 0 else 0
                for k, v in results_by_length.items()
            },
            "by_position": {
                k: v["correct"] / v["total"] if v["total"] > 0 else 0
                for k, v in results_by_position.items()
            },
        }
