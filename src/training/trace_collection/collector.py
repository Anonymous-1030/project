"""
Offline Trace Collection for ProSE-X 2.0.

Collects full-KV traces for offline training.
These traces capture the complete state at each step and are used
for generating teacher labels.

IMPORTANT: This module is ONLY for offline data collection.
It uses oracle/full-KV information that would not be available at runtime.
The collected traces are used for training, not inference.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from src.core_types import (
    ChunkMetadata, QueryContext, ChunkTier
)

logger = logging.getLogger(__name__)


@dataclass
class StepTrace:
    """
    Trace of a single decode step.
    
    Contains full information for offline analysis.
    """
    request_id: str
    step: int
    
    # Query state
    query_tokens: List[int]
    query_text: Optional[str]
    
    # Chunk states (full)
    all_chunks: List[ChunkMetadata]
    
    # Attention weights (if available)
    attention_weights: Optional[Dict[str, float]] = None  # chunk_id -> weight
    
    # Gold information (for training)
    gold_chunk_ids: Optional[List[str]] = None
    gold_token_positions: Optional[List[int]] = None
    
    # Generated output
    generated_token: Optional[int] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "step": self.step,
            "query_tokens": self.query_tokens,
            "query_text": self.query_text,
            "all_chunks": [c.to_dict() for c in self.all_chunks],
            "attention_weights": self.attention_weights,
            "gold_chunk_ids": self.gold_chunk_ids,
            "gold_token_positions": self.gold_token_positions,
            "generated_token": self.generated_token,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RequestTrace:
    """Complete trace for a request."""
    request_id: str
    steps: List[StepTrace] = field(default_factory=list)
    
    # Request metadata
    prompt: Optional[str] = None
    expected_answer: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "steps": [s.to_dict() for s in self.steps],
        }


class TraceCollector:
    """
    Collector for offline traces.
    
    Captures full-KV state at each step for later analysis and training.
    This is an OFFLINE-ONLY tool - it has access to oracle information
    that would not be available at runtime.
    """
    
    def __init__(
        self,
        output_dir: str,
        steps_per_file: int = 1000,
        save_attention: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps_per_file = steps_per_file
        self.save_attention = save_attention
        
        # Active traces
        self._active_traces: Dict[str, RequestTrace] = {}
        
        # Buffer for writing
        self._step_buffer: List[StepTrace] = []
        self._file_counter = 0
        
        logger.info(f"TraceCollector initialized: output_dir={output_dir}")
    
    def start_request(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        expected_answer: Optional[str] = None,
    ) -> None:
        """Start collecting trace for a new request."""
        self._active_traces[request_id] = RequestTrace(
            request_id=request_id,
            prompt=prompt,
            expected_answer=expected_answer,
        )
        logger.debug(f"Started trace collection for {request_id}")
    
    def record_step(
        self,
        request_id: str,
        step: int,
        query_context: QueryContext,
        all_chunks: List[ChunkMetadata],
        attention_weights: Optional[Dict[str, float]] = None,
        gold_chunk_ids: Optional[List[str]] = None,
        generated_token: Optional[int] = None,
    ) -> None:
        """
        Record a single step.
        
        Args:
            request_id: Request identifier
            step: Decode step number
            query_context: Query context
            all_chunks: All chunks (full state)
            attention_weights: Attention weights per chunk (oracle)
            gold_chunk_ids: Gold evidence chunks (oracle)
            generated_token: Generated token ID
        """
        if request_id not in self._active_traces:
            logger.warning(f"No active trace for {request_id}, starting new")
            self.start_request(request_id)
        
        trace = StepTrace(
            request_id=request_id,
            step=step,
            query_tokens=query_context.query_tokens or [],
            query_text=query_context.query_text,
            all_chunks=all_chunks,
            attention_weights=attention_weights if self.save_attention else None,
            gold_chunk_ids=gold_chunk_ids,
            generated_token=generated_token,
        )
        
        self._active_traces[request_id].steps.append(trace)
        self._step_buffer.append(trace)
        
        # Flush if buffer is full
        if len(self._step_buffer) >= self.steps_per_file:
            self._flush_buffer()
    
    def end_request(self, request_id: str) -> Optional[Path]:
        """
        End trace collection for a request and save.
        
        Returns:
            Path to saved file
        """
        if request_id not in self._active_traces:
            logger.warning(f"No active trace for {request_id}")
            return None
        
        request_trace = self._active_traces.pop(request_id)
        
        # Save individual request trace
        output_path = self.output_dir / f"request_{request_id}.json"
        with open(output_path, 'w') as f:
            json.dump(request_trace.to_dict(), f, indent=2)
        
        logger.debug(f"Saved trace for {request_id} to {output_path}")
        return output_path
    
    def _flush_buffer(self) -> None:
        """Flush step buffer to file."""
        if not self._step_buffer:
            return
        
        output_path = self.output_dir / f"steps_{self._file_counter:05d}.jsonl"
        
        with open(output_path, 'w') as f:
            for step in self._step_buffer:
                f.write(json.dumps(step.to_dict()) + '\n')
        
        logger.debug(f"Flushed {len(self._step_buffer)} steps to {output_path}")
        
        self._step_buffer = []
        self._file_counter += 1
    
    def close(self) -> None:
        """Close collector and flush remaining traces."""
        # Flush buffer
        self._flush_buffer()
        
        # Save any remaining active traces
        for request_id in list(self._active_traces.keys()):
            self.end_request(request_id)
        
        logger.info("TraceCollector closed")


class TraceReader:
    """Reader for collected traces."""
    
    def __init__(self, trace_dir: str):
        self.trace_dir = Path(trace_dir)
    
    def read_request_trace(self, request_id: str) -> Optional[RequestTrace]:
        """Read a complete request trace."""
        path = self.trace_dir / f"request_{request_id}.json"
        if not path.exists():
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct (simplified - full reconstruction would need ChunkMetadata parsing)
        return RequestTrace(
            request_id=data["request_id"],
            prompt=data.get("prompt"),
            expected_answer=data.get("expected_answer"),
        )
    
    def iterate_step_traces(self):
        """Iterate over all step traces."""
        for path in sorted(self.trace_dir.glob("steps_*.jsonl")):
            with open(path, 'r') as f:
                for line in f:
                    yield json.loads(line)
    
    def get_all_request_ids(self) -> List[str]:
        """Get all request IDs in the trace directory."""
        request_ids = []
        for path in self.trace_dir.glob("request_*.json"):
            request_id = path.stem.replace("request_", "")
            request_ids.append(request_id)
        return request_ids
