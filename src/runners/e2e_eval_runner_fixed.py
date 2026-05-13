    def store_prefill_kv(
        self, past_key_values,
    ) -> None:
        """Handle different KV cache formats including DynamicCache."""
        if past_key_values is None:
            return
        
        # Try different formats in order of likelihood
        
        # 1. Try iterating (works for DynamicCache and list of tuples)
        try:
            kv_list = list(past_key_values)
            if len(kv_list) > 0:
                first = kv_list[0]
                # Check if it's a list of (k, v) tuples
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    self.full_k = [kv[0].detach() for kv in kv_list]
                    self.full_v = [kv[1].detach() for kv in kv_list]
                # Check if it's a list of tensors (flattened)
                elif isinstance(first, torch.Tensor):
                    self.full_k = [kv_list[i].detach() for i in range(0, len(kv_list), 2)]
                    self.full_v = [kv_list[i].detach() for i in range(1, len(kv_list), 2)]
                else:
                    raise ValueError(f"Unknown item type in iterable: {type(first)}")
                
                # Successfully parsed
                self.seq_len = self.full_k[0].shape[2]
                self.chunk_boundaries = []
                for start in range(0, self.seq_len, self.chunk_size):
                    end = min(start + self.chunk_size, self.seq_len)
                    self.chunk_boundaries.append((start, end))
                num_chunks = len(self.chunk_boundaries)
                num_anchors = max(1, int(num_chunks * self.anchor_ratio))
                first_a = list(range(min(num_anchors // 2, num_chunks)))
                last_a = list(range(max(0, num_chunks - num_anchors // 2), num_chunks))
                self.anchor_ids = sorted(set(first_a + last_a))
                self.tail_ids = [i for i in range(num_chunks) if i not in self.anchor_ids]
                self.promoted_ids = []
                return
        except Exception as e:
            pass
        
        # 2. Try key_cache/value_cache attributes
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            self.full_k = [k.detach() for k in past_key_values.key_cache]
            self.full_v = [v.detach() for v in past_key_values.value_cache]
        elif hasattr(past_key_values, '_key_cache') and hasattr(past_key_values, '_value_cache'):
            self.full_k = [k.detach() for k in past_key_values._key_cache]
            self.full_v = [v.detach() for v in past_key_values._value_cache]
        else:
            raise ValueError(f"Cannot parse past_key_values of type: {type(past_key_values)}")
        
        self.seq_len = self.full_k[0].shape[2]
        self.chunk_boundaries = []
        for start in range(0, self.seq_len, self.chunk_size):
            end = min(start + self.chunk_size, self.seq_len)
            self.chunk_boundaries.append((start, end))
        num_chunks = len(self.chunk_boundaries)
        num_anchors = max(1, int(num_chunks * self.anchor_ratio))
        first_a = list(range(min(num_anchors // 2, num_chunks)))
        last_a = list(range(max(0, num_chunks - num_anchors // 2), num_chunks))
        self.anchor_ids = sorted(set(first_a + last_a))
        self.tail_ids = [i for i in range(num_chunks) if i not in self.anchor_ids]
        self.promoted_ids = []
