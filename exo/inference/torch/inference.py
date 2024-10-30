# inference.py

import asyncio
import os
import json
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union, List, Dict

import numpy as np
import torch
import logging

from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.torch.model.hf import ShardedHuggingFaceModel
from exo.inference.tokenizers import resolve_tokenizer
from exo.helpers import DEBUG
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.download.hf.hf_helpers import get_weight_map

# Model sampling options
TOP_K = 20
TEMP = 0.6
TOP_P = 0.9

# Number of model instances per model ID in the pool
NUM_MODELS_IN_POOL = 2  # Adjust based on your resource capacity
MAX_CONCURRENT_INFERENCES = 4  # Global concurrency limit


class TorchDynamicShardInferenceEngine(InferenceEngine):
    """
    Torch Dynamic Shard Inference Engine for performing model inference with sharded PyTorch/HF based models.
    """

    def __init__(self, shard_downloader: HFShardDownloader):
        """
        Initialize the inference engine.

        Args:
            shard_downloader: Model and weights sharding download
        """
        self.shard_downloader = shard_downloader

        # Setup device
        if os.environ.get("TORCH_DEVICE"):
            self.device = torch.device(os.environ["TORCH_DEVICE"])
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        torch.set_default_device(self.device)

        # Setup dtype
        self.dtype = torch.get_default_dtype()

        # Setup device_map
        if os.environ.get("TORCH_DEVICE_MAP"):
            self.device_map = os.environ["TORCH_DEVICE_MAP"]
        else:
            self.device_map = str(self.device)

        # Model pools per model ID
        self.model_pools: Dict[str, asyncio.Queue] = {}
        self.model_pool_locks: Dict[str, asyncio.Lock] = {}

        # Tokenizers per model ID
        self.tokenizers: Dict[str, any] = {}

        # Global semaphore to limit concurrent inferences
        self.global_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

        # Logger for debugging
        self.logger = self.get_logger()

    def get_logger(self) -> logging.Logger:
        """
        Initialize and return a logger.

        Returns:
            Configured logger.
        """
        logger = logging.getLogger("TorchDynamicShardInferenceEngine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if DEBUG >= 1 else logging.INFO)
        return logger

    async def get_model_pool(self, model_id: str, shard: Shard) -> asyncio.Queue:
        """
        Get or create a model pool for the specified model ID.

        Args:
            model_id: The model identifier.
            shard: The model shard information.

        Returns:
            An asyncio.Queue representing the model pool for the given model ID.
        """
        if model_id not in self.model_pools:
            # Initialize lock for this model_id
            self.model_pool_locks[model_id] = asyncio.Lock()

            async with self.model_pool_locks[model_id]:
                if model_id not in self.model_pools:
                    self.logger.debug(f"Creating model pool for model_id: {model_id}")
                    # Create a new model pool
                    pool = asyncio.Queue(maxsize=NUM_MODELS_IN_POOL)
                    for _ in range(NUM_MODELS_IN_POOL):
                        model = await self.create_model_instance(shard)
                        await pool.put(model)
                    self.model_pools[model_id] = pool

                    # Load and cache the tokenizer
                    self.tokenizers[model_id] = await resolve_tokenizer(model_id)
                    self.logger.debug(
                        f"Model pool and tokenizer created for model_id: {model_id}"
                    )
        return self.model_pools[model_id]

    async def create_model_instance(self, shard: Shard) -> ShardedHuggingFaceModel:
        """
        Create a new model instance for the given shard.

        Args:
            shard: The model shard information.

        Returns:
            A ShardedHuggingFaceModel instance.
        """
        self.logger.debug(f"Creating model instance for shard: {shard}")

        model_path = await self.shard_downloader.ensure_shard(shard)
        model_wm = await get_weight_map(repo_id=shard.model_id)

        stateful_sharded_model = ShardedHuggingFaceModel(
            shard=shard,
            local_model_path=model_path,
            weight_map=model_wm,
            device=self.device,
            dtype=self.dtype,
            device_map=self.device_map,
            top_k=TOP_K,
            temp=TEMP,
            top_p=TOP_P,
        )

        self.logger.debug(f"Model instance created successfully for shard: {shard}")

        return stateful_sharded_model

    async def infer_prompt(
        self,
        request_id: str,
        shard: Shard,
        prompt: str,
        image_str: Optional[str] = None,
        inference_state: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, bool]:
        """
        Asynchronously processes a prompt using the specified shard and returns the inference result.

        Args:
            request_id: The unique identifier for the request.
            shard: The model shard used for inference.
            prompt: The text prompt to be processed by the model.
            image_str: A base64 encoded image string to be optionally used in the inference.
            inference_state: The cached inference state for resuming or continuing inference.

        Returns:
            A tuple containing:
            - input_ids (np.ndarray): The processed token IDs as a NumPy array if logits were generated.
            - cache_json (str): A JSON string containing the cached input IDs for further inference steps.
            - is_finished (bool): A boolean indicating whether the model has reached the end-of-sequence (EOS) token.
        """
        model_id = shard.model_id
        pool = await self.get_model_pool(model_id, shard)
        stateful_sharded_model = await pool.get()
        tokenizer = self.tokenizers[model_id]

        # Use a per-request variable for past_input_ids
        past_input_ids: Optional[torch.Tensor] = None

        try:
            # Limit concurrent inferences using the global semaphore
            async with self.global_semaphore:
                # Tokenize the prompt
                inputs = tokenizer([prompt], return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)
                input_attention_mask = inputs.attention_mask.to(self.device)

                # Get cache from inference_state
                past_iids, cached_iids = self.infer_caching(inference_state)

                # Decide whether to use past_iids or the current input_ids
                if past_iids is not None:
                    past_input_ids = past_iids
                else:
                    past_input_ids = input_ids

                if DEBUG >= 4:
                    self.logger.debug(f"past_input_ids: {past_input_ids}\n")

                # Perform the forward pass
                shard_hidden_states, shard_past_kvs, shard_logits = (
                    await self.async_forward(
                        stateful_sharded_model=stateful_sharded_model,
                        input_ids=past_input_ids,
                        attention_mask=input_attention_mask,
                    )
                )

                if DEBUG >= 4:
                    self.logger.debug(f"shard_hidden_states: {shard_hidden_states}\n")
                    self.logger.debug(f"shard_past_kvs: {shard_past_kvs}\n")
                    self.logger.debug(f"shard_logits: {shard_logits}")

                next_token = None
                if shard_logits is not None:
                    next_token = await self.async_logit_sample(
                        stateful_sharded_model, shard_logits
                    )
                    # Update past_input_ids with the new token
                    past_input_ids = torch.cat(
                        [past_input_ids, next_token[:, None].squeeze(-1)], dim=-1
                    )
                    input_ids = next_token

                    if DEBUG >= 4:
                        self.logger.debug(f"next_token: {next_token}")

                # Update cached_iids for the next inference step
                if past_input_ids is not None:
                    cached_iids = {"input_ids": past_input_ids.tolist()}

                # Check if the model has reached the end-of-sequence token
                is_finished = False
                if next_token is not None:
                    is_finished = next_token.item() == tokenizer.eos_token_id

                # Prepare the return values
                return_values = (
                    (
                        input_ids.cpu().numpy()
                        if shard_logits is not None
                        else shard_hidden_states.cpu().numpy()
                    ),
                    json.dumps({"cached_iids": cached_iids}),
                    is_finished,
                )

                if DEBUG >= 4:
                    self.logger.debug(f"return_values: {return_values}")

                return return_values
        except Exception as e:
            self.logger.exception(f"Error during inference: {e}")
            raise e
        finally:
            # Ensure model is returned to the pool
            await pool.put(stateful_sharded_model)
            self.logger.debug(
                f"Model instance returned to pool for model_id: {model_id}"
            )

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, bool]:
        """
        Asynchronously processes input tensor data using the specified shard and returns the inference result.

        Args:
            request_id: The unique identifier for the request.
            shard: The model shard used for inference.
            input_data: The input data in NumPy array format to be processed by the model.
            inference_state: The cached inference state for resuming or continuing inference.

        Returns:
            A tuple containing:
            - input_ids (np.ndarray): The processed token IDs as a NumPy array if logits were generated.
            - cache_json (str): A JSON string containing the cached input IDs for further inference steps.
            - is_finished (bool): A boolean indicating whether the model has reached the end-of-sequence (EOS) token.
        """
        model_id = shard.model_id
        pool = await self.get_model_pool(model_id, shard)
        stateful_sharded_model = await pool.get()
        tokenizer = self.tokenizers[model_id]

        # Use a per-request variable for past_input_ids
        past_input_ids: Optional[torch.Tensor] = None

        try:
            # Limit concurrent inferences using the global semaphore
            async with self.global_semaphore:
                if DEBUG >= 4:
                    self.logger.debug("infer_tensor called")
                    self.logger.debug(f"input_data: {input_data}")
                    self.logger.debug(f"shard: {shard}")
                    self.logger.debug(f"inference_state: {inference_state}")

                input_ids = torch.tensor(input_data).to(self.device)

                # Get cache from inference_state
                past_iids, cached_iids = self.infer_caching(inference_state)

                # Detect if hidden_states or not
                hidden_states = None
                if input_ids.size()[-1] > 1:
                    hidden_states = input_ids
                    past_input_ids = past_iids
                else:
                    if past_iids is not None:
                        past_input_ids = past_iids
                    else:
                        past_input_ids = input_ids

                if DEBUG >= 4:
                    self.logger.debug(f"past_input_ids: {past_input_ids}")
                    self.logger.debug(f"hidden_states: {hidden_states}")
                    self.logger.debug(f"inference_state: {inference_state}")

                # Perform the forward pass
                shard_hidden_states, shard_past_kvs, shard_logits = (
                    await self.async_forward(
                        stateful_sharded_model=stateful_sharded_model,
                        input_ids=past_input_ids,
                        hidden_states=hidden_states,
                    )
                )

                next_token = None
                if shard_logits is not None:
                    next_token = await self.async_logit_sample(
                        stateful_sharded_model, shard_logits
                    )
                    input_ids = next_token

                # Update cached_iids for the next inference step
                next_cached_logits = None
                if next_token is not None:
                    if past_input_ids is not None:
                        next_cached_logits = torch.cat(
                            [past_input_ids, next_token], dim=-1
                        ).to(self.device)
                    elif past_iids is not None:
                        next_cached_logits = torch.cat(
                            [past_iids, next_token], dim=-1
                        ).to(self.device)

                    cached_iids = {
                        "input_ids": (
                            next_cached_logits.tolist()
                            if next_cached_logits is not None
                            else []
                        )
                    }

                is_finished = False
                if next_token is not None:
                    is_finished = next_token.item() == tokenizer.eos_token_id

                if is_finished:
                    # Clear cache
                    cached_iids = {"input_ids": []}

                if DEBUG >= 4:
                    self.logger.debug(f"input_ids: {input_ids}")
                    self.logger.debug(f"shard_hidden_states: {shard_hidden_states}\n")
                    self.logger.debug(f"shard_past_kvs: {shard_past_kvs}\n")
                    self.logger.debug(f"shard_logits: {shard_logits}")

                return_values = (
                    (
                        input_ids.cpu().numpy()
                        if shard_logits is not None
                        else shard_hidden_states.cpu().numpy()
                    ),
                    json.dumps({"cached_iids": cached_iids}),
                    is_finished,
                )

                if DEBUG >= 4:
                    self.logger.debug(f"return_values: {return_values}")

                return return_values
        except Exception as e:
            self.logger.exception(f"Error during inference: {e}")
            raise e
        finally:
            # Ensure model is returned to the pool
            await pool.put(stateful_sharded_model)
            self.logger.debug(
                f"Model instance returned to pool for model_id: {model_id}"
            )

    def infer_caching(
        self, inference_state: Optional[str] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
        """
        Inference caching from inference_state JSON.

        Args:
            inference_state: The cached inference state.

        Returns:
            A tuple containing:
            - past_iids (Optional[torch.Tensor]): The cached input IDs as a torch.Tensor.
            - cached_iids (Optional[dict]): The cached input IDs in dictionary form.
        """
        # Setup cache and cached input_ids
        past_iids = None
        cached_iids = None
        if inference_state is not None:
            try:
                infer_state = json.loads(inference_state)
            except ValueError:
                infer_state = None

            if infer_state is not None:
                cached_iids = infer_state.get("cached_iids", {})
                if cached_iids and "input_ids" in cached_iids:
                    past_iids = torch.tensor(cached_iids["input_ids"]).to(self.device)

            if DEBUG >= 4:
                self.logger.debug(
                    f"cached_iids len: {len(cached_iids) if cached_iids else 0}"
                )
                self.logger.debug(f"cached_iids: {cached_iids}")

        return past_iids, cached_iids

    async def async_forward(
        self,
        stateful_sharded_model: ShardedHuggingFaceModel,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[Union[Dict[str, torch.Tensor], List[torch.FloatTensor]]],
        Optional[torch.Tensor],
    ]:
        """
        Asynchronously performs the forward pass using a stateful sharded model.

        Args:
            stateful_sharded_model: The model instance to use for inference.
            input_ids: Input token IDs for the model.
            hidden_states: Precomputed hidden states to be used instead of `input_ids`.
            attention_mask: Mask to prevent attention on padding token indices.

        Returns:
            A tuple containing:
            - shard_hidden_states: Hidden states resulting from the forward pass.
            - shard_past_kvs: List of past key-value tensors (cache) used in the model.
            - shard_logits: The logits computed during the forward pass.
        """
        loop = asyncio.get_running_loop()

        # Offload the blocking forward pass to a thread pool
        try:
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    stateful_sharded_model.forward,
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                ),
            )
        except Exception as e:
            self.logger.exception(f"Exception during async_forward: {e}")
            raise e

        # Assuming the model returns (hidden_states, past_kvs, logits)
        shard_hidden_states = (
            result[0].to(self.device) if result[0] is not None else None
        )
        shard_past_kvs = result[1]  # Assuming these are CPU tensors or non-tensors
        shard_logits = result[2].to(self.device) if result[2] is not None else None

        if DEBUG >= 4:
            self.logger.debug("async_forward result processed")

        return shard_hidden_states, shard_past_kvs, shard_logits

    async def async_logit_sample(
        self, stateful_sharded_model: ShardedHuggingFaceModel, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Asynchronously samples logits using the model's logit sampling method.

        Args:
            stateful_sharded_model: The model instance to use for sampling.
            logits: The logits produced by the model for sampling.

        Returns:
            next_logit: The next logit sampled from given logits.
        """
        loop = asyncio.get_running_loop()

        # Offload the blocking logits_sample to a thread pool
        try:
            next_token = await loop.run_in_executor(
                None,
                functools.partial(stateful_sharded_model.logits_sample, logits=logits),
            )
        except Exception as e:
            self.logger.exception(f"Exception during async_logit_sample: {e}")
            raise e

        if isinstance(next_token, torch.Tensor):
            next_token = next_token.to(self.device)
        else:
            # Handle non-tensor tokens if necessary
            pass

        if DEBUG >= 4:
            self.logger.debug("async_logit_sample result processed")

        return next_token
