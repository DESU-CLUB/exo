import numpy as np
import mlx.core as mx
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel
from .sharded_utils import load_shard, get_image_from_str
from ..shard import Shard
from typing import Optional
from exo.download.shard_download import ShardDownloader
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Tuple, Union, List, Dict


# Number of model instances per model ID in the pool
NUM_MODELS_IN_POOL = 2  # Adjust based on your resource capacity
MAX_CONCURRENT_INFERENCES = 4  # Global concurrency limit


class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

    #Model pools per model ID
    self.model_pools: Dict[str, asyncio.Queue] = {}
    self.model_pool_locks: Dict[str, asyncio.Lock] = {}

    #Tokenizers per model ID
    self.tokenizers: Dict[str, any] = {}

    #Global semaphore to limit concurrent inferences
    self.global_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)


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
          # Create a new model pool
          pool = asyncio.Queue(maxsize=NUM_MODELS_IN_POOL)
          for _ in range(NUM_MODELS_IN_POOL):
            model, tokenizer = await self.create_model_instance(shard)
            await pool.put(model)
          self.model_pools[model_id] = pool

          # Load and cache the tokenizer
          self.tokenizers[model_id] = tokenizer
    return self.model_pools[model_id]
  
  async def create_model_instance(self, shard: Shard) -> Tuple[StatefulShardedModel, any]:
    """
    Create a new model instance for the given shard.

    Args:
        shard: The model shard information.

    Returns:
        A ShardedModel instance.
    """
    model_path = await self.shard_downloader.ensure_shard(shard)
    loop = asyncio.get_running_loop()
    def load_shard_wrapper(): return asyncio.run(load_shard(model_path, shard))
    model_shard, tokenizer = await loop.run_in_executor(self.executor, load_shard_wrapper)
    stateful_sharded_model = await loop.run_in_executor(self.executor, StatefulShardedModel, shard, model_shard)
    return stateful_sharded_model, tokenizer

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    model_id = shard.model_id
    pool = await self.get_model_pool(model_id, shard)
    model = await pool.get()
    tokenizer = self.tokenizers[model_id]
    loop = asyncio.get_running_loop()
    try:
      async with self.global_semaphore:
        if image_str:
          # Handle multimodal input
          image = await get_image_from_str(image_str)
          tokenize = partial(self.tokenizer, prompt, image, return_tensors="np")
          inputs = await loop.run_in_executor(self.executor, tokenize)

          #Convert inputs to MLX arrays
          pixel_values = mx.array(inputs["pixel_values"])
          input_ids = mx.array(inputs["input_ids"])

          #Run inference
          output_data: np.ndarray = np.array(await loop.run_in_executor(self.executor, model.step, request_id, input_ids, pixel_values))
        else:
          input_ids = mx.array(await loop.run_in_executor(self.executor, tokenizer.encode, prompt))
          output_data: np.ndarray = np.array(await loop.run_in_executor(self.executor, model.step, request_id, input_ids))
        return output_data, "", output_data.size == 1 and output_data.item() == tokenizer.eos_token_id
    finally:
      await pool.put(model)

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    model_id = shard.model_id
    pool = await self.get_model_pool(model_id, shard)
    model = await pool.get()
    tokenizer = self.tokenizers[model_id]
    try:
      async with self.global_semaphore:
        output_data: np.ndarray = np.array(await asyncio.get_running_loop().run_in_executor(self.executor, model.step, request_id, mx.array(input_data)))
        return output_data, "", output_data.size == 1 and output_data.item() == tokenizer.eos_token_id
    finally:
      await pool.put(model)

