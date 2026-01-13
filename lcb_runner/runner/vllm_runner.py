try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    # print("Cannot import vllm")
    pass

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from time import sleep
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from lcb_runner.runner.base_runner import BaseRunner


class VLLMRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Check if server mode is enabled
        if args.vllm_server_url:
            # Server mode: use OpenAI-compatible client
            self.server_mode = True
            import os
            # Use OPENAI_API_KEY from environment if set, otherwise use dummy key
            api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
            self.client = OpenAI(
                base_url=args.vllm_server_url,
                api_key=api_key,
            )
            model_name = (
                model.model_name if args.local_model_path is None else args.local_model_path
            )
            self.client_kwargs = {
                "model": model_name,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
            }
            if args.stop:
                self.client_kwargs["stop"] = args.stop
            self.llm = None
            self.sampling_params = None
        else:
            # Direct mode: use vLLM directly
            self.server_mode = False
            model_tokenizer_path = (
                model.model_name if args.local_model_path is None else args.local_model_path
            )
            self.llm = LLM(
                model=model_tokenizer_path,
                tokenizer=model_tokenizer_path,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                enforce_eager=True,
                disable_custom_all_reduce=True,
                enable_prefix_caching=args.enable_prefix_caching,
                trust_remote_code=args.trust_remote_code,
            )
            self.sampling_params = SamplingParams(
                n=self.args.n,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=self.args.stop,
            )
            self.client = None
            self.client_kwargs = None

    def _run_single(self, prompt: str | list[dict[str, str]]) -> list[str]:
        if self.server_mode:
            # Server mode: use OpenAI-compatible API
            if isinstance(prompt, str):
                # Convert string prompt to chat format
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **self.client_kwargs,
                )
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                print("Exception: ", repr(e))
                print("Sleeping for 30 seconds...")
                sleep(30)
                return self._run_single(prompt)
            except Exception as e:
                print(f"Failed to run the model for {prompt}!")
                print("Exception: ", repr(e))
                raise e
            return [c.message.content for c in response.choices]
        else:
            # Direct mode: not used, run_batch handles it
            pass

    def run_batch(self, prompts: list[str | list[dict[str, str]]]) -> list[list[str]]:
        if self.server_mode:
            # Server mode: use OpenAI client with batch processing
            outputs = [None] * len(prompts)
            remaining_prompts = []
            remaining_indices = []
            
            # First, check cache and collect remaining prompts
            for prompt_index, prompt in enumerate(prompts):
                if self.args.use_cache:
                    if isinstance(prompt, list):
                        prompt_cache = json.dumps(prompt)
                    elif isinstance(prompt, tuple):
                        prompt_cache = prompt[0] + json.dumps(prompt[1])
                    else:
                        prompt_cache = prompt
                    if prompt_cache in self.cache:
                        if len(self.cache[prompt_cache]) == self.args.n:
                            outputs[prompt_index] = self.cache[prompt_cache]
                            continue
                
                remaining_prompts.append(prompt)
                remaining_indices.append(prompt_index)
            
            # For server mode, use threading to send concurrent requests
            # This allows vLLM server to process multiple requests in parallel
            # Threading works because each thread can use the same client instance
            if remaining_prompts:
                max_workers = min(len(remaining_prompts), self.args.multiprocess if self.args.multiprocess > 0 else 8)
                if max_workers > 1:
                    # Use ThreadPoolExecutor for concurrent requests
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all requests
                        future_to_index = {
                            executor.submit(self._run_single, prompt): (idx, prompt)
                            for idx, prompt in zip(remaining_indices, remaining_prompts)
                        }
                        # Collect results as they complete
                        for future in as_completed(future_to_index):
                            prompt_index, prompt = future_to_index[future]
                            try:
                                result = future.result()
                                outputs[prompt_index] = result
                                
                                if self.args.use_cache:
                                    if isinstance(prompt, list):
                                        prompt_cache = json.dumps(prompt)
                                    elif isinstance(prompt, tuple):
                                        prompt_cache = prompt[0] + json.dumps(prompt[1])
                                    else:
                                        prompt_cache = prompt
                                    self.cache[prompt_cache] = result
                            except Exception as e:
                                print(f"Failed to process prompt at index {prompt_index}: {e}")
                                outputs[prompt_index] = [""] * self.args.n
                else:
                    # Sequential processing if max_workers <= 1
                    for prompt_index, prompt in zip(remaining_indices, remaining_prompts):
                        result = self._run_single(prompt)
                        outputs[prompt_index] = result
                        
                        if self.args.use_cache:
                            if isinstance(prompt, list):
                                prompt_cache = json.dumps(prompt)
                            elif isinstance(prompt, tuple):
                                prompt_cache = prompt[0] + json.dumps(prompt[1])
                            else:
                                prompt_cache = prompt
                            self.cache[prompt_cache] = result
            
            return outputs
        else:
            # Direct mode: use vLLM directly (original implementation)
            outputs = [None for _ in prompts]
            remaining_prompts = []
            remaining_indices = []
            for prompt_index, prompt in enumerate(prompts):
                # For direct mode, prompts should be strings
                if isinstance(prompt, list):
                    # Convert chat messages to string if needed
                    # This is a fallback - ideally prompts should already be strings for direct mode
                    prompt_str = "\n".join([msg.get("content", "") for msg in prompt])
                else:
                    prompt_str = prompt
                
                if self.args.use_cache and prompt_str in self.cache:
                    if len(self.cache[prompt_str]) == self.args.n:
                        outputs[prompt_index] = self.cache[prompt_str]
                        continue
                remaining_prompts.append(prompt_str)
                remaining_indices.append(prompt_index)
            if remaining_prompts:
                vllm_outputs = self.llm.generate(remaining_prompts, self.sampling_params)
                if self.args.use_cache:
                    assert len(remaining_prompts) == len(vllm_outputs)
                    for index, remaining_prompt, vllm_output in zip(
                        remaining_indices, remaining_prompts, vllm_outputs
                    ):
                        self.cache[remaining_prompt] = [o.text for o in vllm_output.outputs]
                        outputs[index] = [o.text for o in vllm_output.outputs]
                else:
                    for index, vllm_output in zip(remaining_indices, vllm_outputs):
                        outputs[index] = [o.text for o in vllm_output.outputs]
            return outputs
