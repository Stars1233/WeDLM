# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WeDLM Decoder module for sliding window decoding.

This module handles all WeDLM-specific decoding logic including:
- WeDLM state initialization and lifecycle management
- Window token management (refilling, pruning)
- Input preparation for decode phase
- Output processing and token extraction

The actual model forward pass is handled by ModelRunner,
with this class providing the inputs and processing the outputs.
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

from wedlm.engine.sequence import Sequence, WeDLMState
from wedlm.engine.sampler import Sampler


@dataclass
class DecodeContext:
    """Context data needed for decode phase model execution.
    
    This dataclass bundles all the tensor data needed to set up
    the attention context for WeDLM decoding.
    """
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    per_seq_wedlm_sizes: torch.Tensor
    max_seqlen_q: int


@dataclass
class PreparedDecodeInputs:
    """Prepared inputs for the decode phase.
    
    Contains all data needed to run a decode step:
    - Model inputs (input_ids, positions)
    - Context for attention
    - Metadata for output processing
    """
    active_seqs: List[Sequence]
    active_states: List[WeDLMState]
    active_indices: List[int]
    input_ids: torch.Tensor
    positions: torch.Tensor
    per_seq_num_non_mask: List[int]
    context: DecodeContext


class WeDLMDecoder:
    """Handles WeDLM sliding window decoding algorithm.
    
    This class manages all WeDLM-specific logic, separating it from the
    model execution infrastructure in ModelRunner. The decoding flow is:
    
    1. initialize_states() - Create WeDLMState for new sequences
    2. prepare_decode_inputs() - Prepare tensors for model forward pass
    3. [ModelRunner runs the model]
    4. process_decode_outputs() - Process logits and update window states
    
    Attributes:
        mask_token_id: Token ID used for mask positions
        block_size: KV cache block size
        wedlm_window_size: Size of the sliding window
        sampler: Sampler instance for token sampling
    """

    def __init__(
        self,
        mask_token_id: int,
        block_size: int,
        wedlm_window_size: int,
        sampler: Sampler
    ):
        """Initialize the WeDLM decoder.
        
        Args:
            mask_token_id: Token ID used for mask positions
            block_size: KV cache block size
            wedlm_window_size: Maximum size of the sliding window
            sampler: Sampler instance for token sampling and position selection
        """
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.wedlm_window_size = wedlm_window_size
        self.sampler = sampler

    def init_wedlm_state(self, seq: Sequence) -> WeDLMState:
        """Initialize WeDLM state for a sequence.
        
        Creates a new WeDLMState with mask tokens filling the initial window.
        The window size is limited by both wedlm_window_size and kv_budget.
        
        Args:
            seq: Sequence to initialize state for
            
        Returns:
            Initialized WeDLMState
        """
        initial_window_size = min(self.wedlm_window_size, seq.kv_budget)
        
        return WeDLMState(
            window_tokens=[self.mask_token_id] * initial_window_size,
            window_mask_flags=[True] * initial_window_size,
            current_seq_len=len(seq),
            is_finished=False,
            is_initialized=True,
        )

    def initialize_states(self, seqs: List[Sequence]) -> None:
        """Initialize WeDLM states for sequences that don't have one.
        
        Args:
            seqs: List of sequences to check and initialize
        """
        for seq in seqs:
            if seq.wedlm_state is None:
                seq.wedlm_state = self.init_wedlm_state(seq)

    def _compute_slot_mapping(
        self,
        seq: Sequence,
        state: WeDLMState
    ) -> torch.Tensor:
        """Compute slot mapping for the sliding window.
        
        Maps logical positions in the window to physical KV cache slots.
        
        Args:
            seq: Sequence being processed
            state: Current WeDLM state
            
        Returns:
            Tensor of physical slot indices
        """
        window_size = len(state.window_tokens)
        slots = []

        for k in range(window_size):
            logical_idx = state.current_seq_len + k
            block_idx = logical_idx // self.block_size
            block_offset = logical_idx % self.block_size
            physical_block = seq.block_table[block_idx]
            physical_slot = physical_block * self.block_size + block_offset
            slots.append(physical_slot)

        return torch.tensor(slots, dtype=torch.int32, device="cuda")

    def _refill_window_masks(self, seq: Sequence, state: WeDLMState) -> None:
        """Refill mask tokens at the end of window when prefix tokens are decoded.
        
        When tokens at the front of the window have been filled (non-mask),
        we can extend the window by adding more mask tokens at the end,
        up to the kv_budget limit.
        
        Args:
            seq: Sequence being processed
            state: Current WeDLM state (modified in place)
        """
        mask_indices = [i for i, flag in enumerate(state.window_mask_flags) if flag]
        prefix_len = mask_indices[0] if mask_indices else len(state.window_tokens)

        if prefix_len > 0:
            refill_count = min(prefix_len, seq.kv_budget)
            if refill_count > 0:
                state.window_tokens = (
                    state.window_tokens + [self.mask_token_id] * refill_count
                )
                state.window_mask_flags = (
                    state.window_mask_flags + [True] * refill_count
                )

    def _prepare_block_tables(self, seqs: List[Sequence]) -> torch.Tensor:
        """Prepare block tables for batch processing.
        
        Pads block tables to the same length and converts to tensor.
        
        Args:
            seqs: List of sequences
            
        Returns:
            Padded block tables tensor
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

    def _prepare_window_inputs(
        self,
        active_states: List[WeDLMState]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Prepare input tensors from window tokens.
        
        Non-mask tokens are placed first, followed by mask tokens.
        This ordering is important for correct processing of the model output.
        
        Args:
            active_states: List of WeDLMState for active sequences
            
        Returns:
            Tuple of (input_ids, positions, per_seq_num_non_mask)
        """
        device = torch.device("cuda")
        input_ids_list = []
        positions_list = []
        per_seq_num_non_mask = []

        for state in active_states:
            window_tokens = state.window_tokens
            window_mask_flags = state.window_mask_flags

            # Separate mask and non-mask positions, non-mask first
            non_mask_idx = [i for i, flag in enumerate(window_mask_flags) if not flag]
            mask_idx = [i for i, flag in enumerate(window_mask_flags) if flag]
            order = non_mask_idx + mask_idx

            ordered_tokens = [window_tokens[i] for i in order]
            ordered_positions = [state.current_seq_len + i for i in order]

            per_seq_num_non_mask.append(len(non_mask_idx))
            input_ids_list.append(torch.tensor(ordered_tokens, dtype=torch.long))
            positions_list.append(torch.tensor(ordered_positions, dtype=torch.long))

        input_ids = torch.cat(input_ids_list, dim=0).to(device)
        positions = torch.cat(positions_list, dim=0).to(device)

        return input_ids, positions, per_seq_num_non_mask

    def prepare_decode_inputs(
        self,
        seqs: List[Sequence]
    ) -> Optional[PreparedDecodeInputs]:
        """Prepare inputs for decode phase.
        
        This method:
        1. Refills window masks for sequences with confirmed prefix tokens
        2. Filters to active sequences (not finished, has window tokens)
        3. Prepares all tensors needed for model forward pass
        
        Args:
            seqs: List of all sequences
            
        Returns:
            PreparedDecodeInputs if there are active sequences, None otherwise
        """
        # Refill window masks for sequences with confirmed prefix tokens
        for seq in seqs:
            state = seq.wedlm_state
            if state is None or state.is_finished or len(state.window_tokens) == 0:
                continue
            self._refill_window_masks(seq, state)

        # Collect active sequences
        active_indices = []
        active_seqs = []
        active_states = []

        for i, seq in enumerate(seqs):
            state = seq.wedlm_state
            if state is None:
                continue
            if len(state.window_tokens) > 0 and not state.is_finished:
                active_indices.append(i)
                active_seqs.append(seq)
                active_states.append(state)

        if not active_seqs:
            return None

        # Prepare context tensors
        block_tables = self._prepare_block_tables(active_seqs)
        slot_map_list = []
        ctx_lens = []
        per_seq_wedlm_sizes = []

        for seq, state in zip(active_seqs, active_states):
            window_size = len(state.window_tokens)
            per_seq_wedlm_sizes.append(window_size)
            ctx_lens.append(state.current_seq_len)
            slots = self._compute_slot_mapping(seq, state)
            slot_map_list.append(slots)

        slot_mapping = torch.cat(slot_map_list, dim=0)
        context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
        per_seq_wedlm_sizes_tensor = torch.tensor(
            per_seq_wedlm_sizes, dtype=torch.int32, device="cuda"
        )

        context = DecodeContext(
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            per_seq_wedlm_sizes=per_seq_wedlm_sizes_tensor,
            max_seqlen_q=max(per_seq_wedlm_sizes),
        )

        # Prepare model inputs
        input_ids, positions, per_seq_num_non_mask = self._prepare_window_inputs(
            active_states
        )

        return PreparedDecodeInputs(
            active_seqs=active_seqs,
            active_states=active_states,
            active_indices=active_indices,
            input_ids=input_ids,
            positions=positions,
            per_seq_num_non_mask=per_seq_num_non_mask,
            context=context,
        )

    def _process_pruned_tokens(
        self,
        seq: Sequence,
        state: WeDLMState,
        prune_count: int
    ) -> List[int]:
        """Process and prune confirmed tokens from window.
        
        Removes the first prune_count tokens from the window and updates
        the WeDLM state accordingly. Checks for stop conditions.
        
        Args:
            seq: The sequence being processed
            state: Current WeDLM state (modified in place)
            prune_count: Number of tokens to prune from the front of the window
            
        Returns:
            List of pruned token IDs
        """
        if prune_count == 0:
            return []

        pruned_tokens = state.window_tokens[:prune_count]

        for t in pruned_tokens:
            is_stop_token = t in seq.stop_token_ids

            state.current_seq_len += 1

            if is_stop_token:
                state.is_finished = True
                break

            # Check max generation length
            total_generated = state.current_seq_len - seq.num_prompt_tokens
            if total_generated >= seq.max_tokens:
                state.is_finished = True
                break

        if state.is_finished:
            # Return only tokens up to stop position
            stop_idx = len(pruned_tokens)
            for idx, t in enumerate(pruned_tokens):
                if t in seq.stop_token_ids:
                    stop_idx = idx + 1
                    break
            pruned_tokens = pruned_tokens[:stop_idx]
            state.window_tokens = []
            state.window_mask_flags = []
        else:
            # Remove pruned tokens from window
            state.window_tokens = state.window_tokens[prune_count:]
            state.window_mask_flags = state.window_mask_flags[prune_count:]

        return pruned_tokens

    def process_decode_outputs(
        self,
        seqs: List[Sequence],
        prepared: PreparedDecodeInputs,
        logits: torch.Tensor,
    ) -> List[Optional[List[int]]]:
        """Process model outputs and update window states.
        
        For each active sequence:
        1. Extract the relevant logits
        2. Determine prefix tokens to prune (consecutive non-mask from start)
        3. Process pruned tokens and check stop conditions
        4. For remaining masks, use sampler to select positions and sample tokens
        5. Update window state with newly filled positions
        
        Args:
            seqs: All sequences (for result indexing)
            prepared: Prepared decode inputs containing active sequence info
            logits: Model output logits
            
        Returns:
            List of generated tokens for each sequence (None if no tokens generated)
        """
        # Initialize results - one entry per input sequence
        step_results: List[Optional[List[int]]] = [None for _ in seqs]

        active_seqs = prepared.active_seqs
        active_states = prepared.active_states
        active_indices = prepared.active_indices
        per_seq_num_non_mask = prepared.per_seq_num_non_mask

        # Calculate row offsets for each sequence's logits
        row_offsets = []
        total_rows = 0
        for state in active_states:
            row_offsets.append(total_rows)
            total_rows += len(state.window_tokens)

        for j, (seq, state, orig_idx) in enumerate(
            zip(active_seqs, active_states, active_indices)
        ):
            window_size = len(state.window_tokens)
            num_non_mask = per_seq_num_non_mask[j]
            row_start = row_offsets[j]

            seq_logits = logits[row_start : row_start + window_size]

            # Find prefix tokens to prune (consecutive non-mask from the start)
            mask_indices = [
                i for i, flag in enumerate(state.window_mask_flags) if flag
            ]
            prune_count = mask_indices[0] if mask_indices else window_size

            # Process pruned tokens
            pruned_tokens = self._process_pruned_tokens(seq, state, prune_count)

            # Process mask positions if sequence is not finished
            remaining_mask_indices = [
                i for i, flag in enumerate(state.window_mask_flags) if flag
            ]

            if remaining_mask_indices and not state.is_finished:
                # Get logits for mask positions only (they come after non-mask)
                mask_logits = seq_logits[num_non_mask:]

                if mask_logits.size(0) > 0:
                    # Use the Sampler to handle all sampling logic
                    fill_indices, token_ids = self.sampler.process_mask_positions(
                        mask_logits=mask_logits,
                        remaining_mask_indices=remaining_mask_indices,
                        temperature=seq.temperature,
                        entropy_threshold=seq.wedlm_entropy_threshold,
                        pos_penalty_factor=seq.wedlm_pos_penalty_factor,
                        top_p=seq.top_p,
                        top_k=seq.top_k,
                    )

                    # Fill selected positions with sampled tokens
                    for k, token_id in zip(fill_indices, token_ids):
                        if k < len(remaining_mask_indices):
                            target_pos = remaining_mask_indices[k]
                            if target_pos < len(state.window_tokens):
                                state.window_tokens[target_pos] = token_id
                                state.window_mask_flags[target_pos] = False

            # Store results - only the pruned tokens that were confirmed
            step_results[orig_idx] = pruned_tokens if pruned_tokens else None

        return step_results