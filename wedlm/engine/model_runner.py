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
Model runner for WeDLM decoding.

This module handles model execution, KV cache management, and multi-GPU
coordination. The WeDLM decoding algorithm logic has been moved to
engine/wedlm_decoder.py for better separation of concerns.

Responsibilities:
- Model initialization and lifecycle
- KV cache allocation and management
- Multi-GPU process communication
- Prefill input preparation
- Model forward pass execution (eager and CUDA graph modes)
- Coordination with WeDLMDecoder for decode phase
"""

import pickle
import logging
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional

from wedlm.config import Config
from wedlm.engine.sequence import Sequence
from wedlm.engine.sampler import Sampler
from wedlm.engine.wedlm_decoder import WeDLMDecoder
from wedlm.models.wedlm import WeDLMForDiffusionLM
from wedlm.utils.context import set_context, get_context, reset_context
from wedlm.utils.loader import load_model


logger = logging.getLogger(__name__)


class ModelRunner:
    """Model runner for WeDLM decoding.
    
    Handles model execution, KV cache management, and multi-GPU coordination.
    Delegates WeDLM-specific decoding logic to WeDLMDecoder.
    
    The class is organized into several logical sections:
    - Initialization: Model loading, distributed setup, KV cache allocation
    - Process Management: Multi-GPU communication via shared memory
    - Input Preparation: Building tensors for prefill phase
    - Model Execution: Forward pass with eager or CUDA graph modes
    - Main Entry Point: The run() method that coordinates everything
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.wedlm_window_size = config.wedlm_window_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        Sequence.block_size = self.block_size

        # Setup HF config
        hf_config = config.hf_config
        hf_config.wedlm_window_size = self.wedlm_window_size
        hf_config.max_model_len = config.max_model_len

        # Initialize components
        self._init_distributed(config)
        self._init_model(hf_config)
        self._init_mask_token(config, hf_config)
        self._init_wedlm_decoder()

        # Multi-GPU communication setup
        if self.world_size > 1:
            self._init_shared_memory()

    # ========== Initialization ==========

    def _init_distributed(self, config: Config):
        """Initialize distributed training environment."""
        init_method = f"tcp://localhost:{config.nccl_port}"
        dist.init_process_group(
            "nccl", init_method, world_size=self.world_size, rank=self.rank
        )
        torch.cuda.set_device(self.rank)

    def _init_model(self, hf_config):
        """Initialize and load model along with the sampler."""
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        self.model = WeDLMForDiffusionLM(hf_config)
        load_model(self.model, self.config.model)
        
        # Initialize the sampler for token sampling
        self.sampler = Sampler()
        
        self.warmup_model()
        self.allocate_kv_cache()
        
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def _init_mask_token(self, config: Config, hf_config):
        """Initialize mask token ID."""
        if config.mask_token_id is not None:
            self.mask_token_id = config.mask_token_id
        else:
            self.mask_token_id = getattr(hf_config, "mask_token_id", None)
            if self.mask_token_id is None:
                self.mask_token_id = 151665

    def _init_wedlm_decoder(self):
        """Initialize the WeDLM decoder for sliding window decoding."""
        self.wedlm_decoder = WeDLMDecoder(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            wedlm_window_size=self.wedlm_window_size,
            sampler=self.sampler,
        )

    def _init_shared_memory(self):
        """Initialize shared memory for multi-GPU communication."""
        if self.rank == 0:
            self.shm = SharedMemory(name="wedlm", create=True, size=2**20)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name="wedlm")
            self.loop()

    # ========== Process Management ==========

    def exit(self):
        """Clean up resources and terminate."""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """Worker loop for non-master ranks."""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """Read method call from shared memory."""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """Write method call to shared memory."""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """Call method, broadcasting to workers if needed."""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ========== Model Setup ==========

    def warmup_model(self):
        """Warmup model with dummy input to trigger JIT compilation."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """Allocate KV cache based on available GPU memory."""
        config = self.config
        hf_config = config.hf_config
        
        # Calculate available memory
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # Calculate block size in bytes
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        
        # Allocate cache
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        if config.num_kvcache_blocks <= 0:
            logger.warning(
                f"num_kvcache_blocks ({config.num_kvcache_blocks}) is <= 0. "
                "Setting it to 1."
            )
        config.num_kvcache_blocks = max(config.num_kvcache_blocks, 1)
        
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        
        # Assign cache to attention layers
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # ========== Input Preparation ==========

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """Prepare block tables for batch processing.
        
        Pads block tables to the same length and converts to tensor.
        
        Args:
            seqs: List of sequences
            
        Returns:
            Padded block tables tensor on GPU
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare inputs for prefill phase.
        
        Builds input tensors and sets up attention context for prefill.
        
        Args:
            seqs: List of sequences to prefill
            
        Returns:
            Tuple of (input_ids, positions) tensors
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    # ========== Model Execution ==========

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ) -> torch.Tensor:
        """Run model forward pass.
        
        Chooses between eager execution and CUDA graph based on configuration
        and input size.
        
        Args:
            input_ids: Input token IDs
            positions: Position indices
            is_prefill: Whether this is a prefill (vs decode) step
            
        Returns:
            Model logits
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions)
            logits = self.model.compute_logits(hidden_states)
        else:
            logits = self._run_with_cudagraph(input_ids, positions)

        if self.rank == 0 and is_prefill:
            context = get_context()
            last_indices = context.cu_seqlens_q[1:] - 1
            logits = logits[last_indices]

        return logits

    def _run_with_cudagraph(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Run model using CUDA graph for better performance.
        
        Args:
            input_ids: Input token IDs
            positions: Position indices
            
        Returns:
            Model logits
        """
        context = get_context()
        real_bs = context.per_seq_wedlm_sizes.size(0)

        try:
            graph_seq_capacity = next(x for x in self.graph_bs if x >= real_bs)
        except StopIteration:
            hidden_states = self.model(input_ids, positions)
            return self.model.compute_logits(hidden_states)

        graph = self.graphs[graph_seq_capacity]
        graph_vars = self.graph_vars[graph_seq_capacity]
        num_tokens = input_ids.size(0)

        if num_tokens > graph_vars["input_ids"].size(0):
            hidden_states = self.model(input_ids, positions)
            return self.model.compute_logits(hidden_states)

        # Copy inputs to graph buffers
        graph_vars["input_ids"][:num_tokens].copy_(input_ids)
        graph_vars["positions"][:num_tokens].copy_(positions)
        graph_vars["per_seq_wedlm_sizes"][:real_bs].copy_(context.per_seq_wedlm_sizes)
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:num_tokens].copy_(context.slot_mapping)
        graph_vars["context_lens"][:real_bs].copy_(context.context_lens)

        if context.block_tables is not None:
            valid_rows = min(real_bs, context.block_tables.size(0))
            valid_cols = min(
                graph_vars["block_tables"].size(1), context.block_tables.size(1)
            )
            graph_vars["block_tables"][:valid_rows, :valid_cols].copy_(
                context.block_tables[:valid_rows, :valid_cols]
            )

        graph.replay()

        hidden_states = graph_vars["outputs"][:num_tokens]
        return self.model.compute_logits(hidden_states)

    # ========== WeDLM Decode Step ==========

    @torch.inference_mode()
    def _wedlm_decode_one_step(
        self, seqs: List[Sequence]
    ) -> List[Optional[List[int]]]:
        """Execute one step of WeDLM decoding (single forward pass).
        
        This method coordinates with WeDLMDecoder:
        1. WeDLMDecoder initializes states
        2. WeDLMDecoder prepares inputs
        3. ModelRunner runs the model
        4. WeDLMDecoder processes outputs
        
        Args:
            seqs: List of sequences to process
            
        Returns:
            List of generated tokens for each sequence (None if no tokens)
        """
        # Initialize WeDLM states for sequences that don't have one
        self.wedlm_decoder.initialize_states(seqs)

        # Initialize results - one entry per input sequence
        step_results: List[Optional[List[int]]] = [None for _ in seqs]

        # Prepare decode inputs (returns None if no active sequences)
        prepared = self.wedlm_decoder.prepare_decode_inputs(seqs)
        
        if prepared is None:
            return step_results

        # Set context for attention
        context = prepared.context
        set_context(
            False,
            slot_mapping=context.slot_mapping,
            context_lens=context.context_lens,
            block_tables=context.block_tables,
            per_seq_wedlm_sizes=context.per_seq_wedlm_sizes,
            max_seqlen_q=context.max_seqlen_q,
        )

        # Run model forward pass
        logits = self.run_model(prepared.input_ids, prepared.positions, is_prefill=False)

        # Process outputs through WeDLMDecoder
        step_results = self.wedlm_decoder.process_decode_outputs(seqs, prepared, logits)

        reset_context()
        return step_results

    # ========== Main Entry Point ==========

    def run(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> List[Optional[List[int]]]:
        """Execute one inference step.
        
        This method is responsible only for token generation.
        The outer layer (LLMEngine) can determine the number of active sequences
        by examining the input seqs list.
        
        Args:
            seqs: List of sequences to process
            is_prefill: True for prefill phase, False for decode phase
            
        Returns:
            List of generated tokens for each sequence (None for prefill or
            if no tokens generated)
        """
        if is_prefill:
            # Prefill phase: process prompts and fill KV cache
            input_ids, positions = self.prepare_prefill(seqs)
            _ = self.run_model(input_ids, positions, is_prefill=True)
            reset_context()
            return [None for _ in seqs]

        # Decode phase: generate tokens using WeDLM decoding
        return self._wedlm_decode_one_step(seqs)

    # ========== CUDA Graph Capture ==========

    @torch.inference_mode()
    def capture_cudagraph(self):
        """Capture CUDA graphs for efficient decode execution.
        
        Creates CUDA graphs for various batch sizes to accelerate
        decode-phase model execution.
        """
        config = self.config
        hf_config = config.hf_config
        max_seqs = config.max_num_seqs

        # Determine batch sizes to capture (powers of 2 up to max_seqs)
        self.graph_bs = []
        current_bs = 1
        while current_bs <= max_seqs:
            self.graph_bs.append(current_bs)
            current_bs *= 2

        base_step_size = (
            self.wedlm_window_size if self.wedlm_window_size is not None else 1
        )
        capture_step_size = 2 * max(1, int(base_step_size))
        max_num_blocks = (
            (config.max_model_len + self.block_size - 1) // self.block_size
        )

        self.graphs = {}
        self.graph_vars = {}
        self.graph_pool = None

        for num_seqs in self.graph_bs:
            max_tokens_in_bucket = num_seqs * capture_step_size

            # Allocate buffers for graph capture
            input_ids = torch.zeros(
                max_tokens_in_bucket, dtype=torch.int64, device="cuda"
            )
            positions = torch.zeros(
                max_tokens_in_bucket, dtype=torch.int64, device="cuda"
            )
            slot_mapping = torch.full(
                (max_tokens_in_bucket,), -1, dtype=torch.int32, device="cuda"
            )
            context_lens = torch.zeros(num_seqs, dtype=torch.int32, device="cuda")
            block_tables = torch.zeros(
                num_seqs, max_num_blocks, dtype=torch.int32, device="cuda"
            )
            per_seq_wedlm_sizes = torch.full(
                (num_seqs,), capture_step_size, dtype=torch.int32, device="cuda"
            )
            outputs = torch.zeros(
                max_tokens_in_bucket, hf_config.hidden_size, device="cuda"
            )

            graph = torch.cuda.CUDAGraph()

            set_context(
                False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                per_seq_wedlm_sizes=per_seq_wedlm_sizes,
                max_seqlen_q=capture_step_size,
            )

            # Warmup run before capture
            outputs[:] = self.model(input_ids, positions)

            # Capture the graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:] = self.model(input_ids, positions)

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[num_seqs] = graph
            self.graph_vars[num_seqs] = dict(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                per_seq_wedlm_sizes=per_seq_wedlm_sizes,
                outputs=outputs,
            )

            reset_context()

        torch.cuda.synchronize()