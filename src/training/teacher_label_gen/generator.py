"""
Teacher Label Generator for ProSE-X 2.0.

Generates teacher utility targets from full-KV traces.
This is an OFFLINE-ONLY process.

Teacher targets combine:
1. Gold evidence indicator (if available)
2. Future attention gain proxy
3. Answer logit / correctness delta (if measurable)

The generated labels are saved to disk for inspection and used to train ODUS.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import numpy as np

from src.training.trace_collection.collector import StepTrace, TraceReader

logger = logging.getLogger(__name__)


@dataclass
class TeacherLabel:
    """
    Teacher label for a single chunk at a single step.
    
    Contains the utility target and component breakdown.
    """
    request_id: str
    step: int
    chunk_id: str
    
    # Target utility score
    utility_target: float
    
    # Component breakdown (for interpretability)
    gold_indicator: float  # 1.0 if gold, 0.0 otherwise
    future_attention_proxy: float  # Normalized future attention
    correctness_delta: float  # Impact on answer correctness
    
    # Metadata
    generation_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "step": self.step,
            "chunk_id": self.chunk_id,
            "utility_target": self.utility_target,
            "gold_indicator": self.gold_indicator,
            "future_attention_proxy": self.future_attention_proxy,
            "correctness_delta": self.correctness_delta,
            "generation_timestamp": self.generation_timestamp.isoformat(),
        }


class TeacherLabelGenerator:
    """
    Generator for teacher utility labels.
    
    Combines multiple signals into a teacher utility target:
    - Gold evidence presence
    - Future attention patterns
    - Correctness impact
    
    Formula (configurable):
        utility = w1 * gold + w2 * attention + w3 * correctness
    """
    
    def __init__(
        self,
        gold_weight: float = 0.5,
        attention_weight: float = 0.3,
        correctness_weight: float = 0.2,
        future_window: int = 10,
    ):
        self.gold_weight = gold_weight
        self.attention_weight = attention_weight
        self.correctness_weight = correctness_weight
        self.future_window = future_window
        
        # Normalization
        total = gold_weight + attention_weight + correctness_weight
        self.gold_weight /= total
        self.attention_weight /= total
        self.correctness_weight /= total
        
        logger.info(
            f"TeacherLabelGenerator initialized: "
            f"gold={self.gold_weight:.2f}, "
            f"attention={self.attention_weight:.2f}, "
            f"correctness={self.correctness_weight:.2f}"
        )
    
    def generate_from_trace(
        self,
        step_trace: StepTrace,
        future_traces: Optional[List[StepTrace]] = None,
    ) -> Dict[str, TeacherLabel]:
        """
        Generate teacher labels for all chunks in a step.
        
        Args:
            step_trace: Current step trace
            future_traces: Next N steps (for future attention)
            
        Returns:
            Dictionary mapping chunk_id to TeacherLabel
        """
        labels = {}
        
        # Precompute gold set
        gold_ids = set(step_trace.gold_chunk_ids or [])
        
        # Precompute future attention (if future traces available)
        future_attention = self._compute_future_attention(
            step_trace.chunk_id, future_traces
        )
        
        for chunk in step_trace.all_chunks:
            chunk_id = chunk.chunk_id
            
            # Component 1: Gold indicator
            gold_indicator = 1.0 if chunk_id in gold_ids else 0.0
            
            # Component 2: Future attention proxy
            attention_proxy = future_attention.get(chunk_id, 0.0)
            
            # Component 3: Correctness delta (placeholder)
            # In real implementation, would measure impact on answer generation
            correctness = 0.0
            if gold_indicator > 0:
                correctness = 1.0  # Placeholder
            
            # Combine into utility target
            utility = (
                self.gold_weight * gold_indicator +
                self.attention_weight * attention_proxy +
                self.correctness_weight * correctness
            )
            
            labels[chunk_id] = TeacherLabel(
                request_id=step_trace.request_id,
                step=step_trace.step,
                chunk_id=chunk_id,
                utility_target=utility,
                gold_indicator=gold_indicator,
                future_attention_proxy=attention_proxy,
                correctness_delta=correctness,
            )
        
        return labels
    
    def _compute_future_attention(
        self,
        chunk_id: str,
        future_traces: Optional[List[StepTrace]],
    ) -> Dict[str, float]:
        """
        Compute future attention proxy.
        
        Returns:
            Dictionary mapping chunk_id to normalized future attention
        """
        if not future_traces:
            return {}
        
        # Aggregate attention over future window
        attention_sum: Dict[str, float] = {}
        
        for trace in future_traces:
            if trace.attention_weights:
                for cid, weight in trace.attention_weights.items():
                    attention_sum[cid] = attention_sum.get(cid, 0.0) + weight
        
        # Normalize
        if attention_sum:
            max_val = max(attention_sum.values())
            if max_val > 0:
                attention_sum = {k: v / max_val for k, v in attention_sum.items()}
        
        return attention_sum
    
    def generate_from_traces(
        self,
        traces: List[StepTrace],
    ) -> List[TeacherLabel]:
        """
        Generate labels from a list of step traces.
        
        Args:
            traces: List of step traces (assumed sorted by request_id, step)
            
        Returns:
            List of TeacherLabels
        """
        all_labels = []
        
        # Group by request
        traces_by_request: Dict[str, List[StepTrace]] = {}
        for trace in traces:
            if trace.request_id not in traces_by_request:
                traces_by_request[trace.request_id] = []
            traces_by_request[trace.request_id].append(trace)
        
        # Generate labels for each request
        for request_id, request_traces in traces_by_request.items():
            # Sort by step
            request_traces.sort(key=lambda t: t.step)
            
            for i, trace in enumerate(request_traces):
                # Get future window
                future = request_traces[i+1:i+1+self.future_window]
                
                labels = self.generate_from_trace(trace, future)
                all_labels.extend(labels.values())
        
        return all_labels


class LabelWriter:
    """Writer for teacher labels."""
    
    def __init__(
        self,
        output_dir: str,
        labels_per_file: int = 10000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.labels_per_file = labels_per_file
        
        self._buffer: List[TeacherLabel] = []
        self._file_counter = 0
    
    def write_label(self, label: TeacherLabel) -> None:
        """Write a single label."""
        self._buffer.append(label)
        
        if len(self._buffer) >= self.labels_per_file:
            self._flush()
    
    def write_labels(self, labels: List[TeacherLabel]) -> None:
        """Write multiple labels."""
        for label in labels:
            self.write_label(label)
    
    def _flush(self) -> None:
        """Flush buffer to file."""
        if not self._buffer:
            return
        
        output_path = self.output_dir / f"labels_{self._file_counter:05d}.jsonl"
        
        with open(output_path, 'w') as f:
            for label in self._buffer:
                f.write(json.dumps(label.to_dict()) + '\n')
        
        logger.debug(f"Wrote {len(self._buffer)} labels to {output_path}")
        
        self._buffer = []
        self._file_counter += 1
    
    def close(self) -> None:
        """Close writer and flush remaining labels."""
        self._flush()


class BatchLabelGenerator:
    """Batch processor for generating labels from trace directory."""
    
    def __init__(
        self,
        trace_dir: str,
        output_dir: str,
        generator: Optional[TeacherLabelGenerator] = None,
    ):
        self.trace_reader = TraceReader(trace_dir)
        self.label_writer = LabelWriter(output_dir)
        self.generator = generator or TeacherLabelGenerator()
    
    def process_all(self) -> int:
        """
        Process all traces in the directory.
        
        Returns:
            Number of labels generated
        """
        total_labels = 0
        
        # Process step traces
        for step_data in self.trace_reader.iterate_step_traces():
            # Reconstruct StepTrace (simplified)
            step_trace = self._dict_to_step_trace(step_data)
            
            labels = self.generator.generate_from_trace(step_trace)
            self.label_writer.write_labels(list(labels.values()))
            
            total_labels += len(labels)
        
        self.label_writer.close()
        
        logger.info(f"Generated {total_labels} teacher labels")
        return total_labels
    
    def _dict_to_step_trace(self, data: Dict[str, Any]) -> StepTrace:
        """Convert dict back to StepTrace (simplified)."""
        return StepTrace(
            request_id=data["request_id"],
            step=data["step"],
            query_tokens=data.get("query_tokens", []),
            query_text=data.get("query_text"),
            all_chunks=[],  # Would need full reconstruction
            attention_weights=data.get("attention_weights"),
            gold_chunk_ids=data.get("gold_chunk_ids"),
            generated_token=data.get("generated_token"),
        )
