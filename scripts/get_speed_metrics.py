#!/usr/bin/env python3
"""
Script to measure vLLM speed metrics with concurrent requests.
Tests different input token sizes with various levels of concurrency.
"""

import requests
import time
import json
import os
import threading
from transformers import AutoTokenizer


class VLLMSpeedTester:
    """Class to handle vLLM speed testing with concurrent requests."""
    
    def __init__(self, vllm_url="http://localhost:8000/v1/chat/completions"):
        self.vllm_url = vllm_url
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the tokenizer for token counting."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Token counting may be inaccurate")
    
    def measure_vllm_response(self, file_path, max_input_tokens=None, max_output_tokens=1024):
        """
        Send file content to vLLM chat endpoint and measure response metrics with streaming
        
        Args:
            file_path: Path to the input file
            max_input_tokens: Maximum tokens for input (for truncation), None for no limit
            max_output_tokens: Maximum tokens for output response
            
        Returns:
            tuple: (response_text, metrics_dict)
        """
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Truncate content if max_input_tokens is specified
        if max_input_tokens is not None and self.tokenizer:
            tokens = self.tokenizer.encode(content)
            if len(tokens) > max_input_tokens:
                truncated_tokens = tokens[:max_input_tokens]
                content = self.tokenizer.decode(truncated_tokens)
        
        # Get actual input token count
        input_tokens = len(self.tokenizer.encode(content)) if self.tokenizer else len(content.split())
        
        # Prepare chat request with streaming
        payload = {
            "model": "qwen3",
            "messages": [
                {"role": "user", "content": content}
            ],
            "max_tokens": max_output_tokens,
            "temperature": 0.7,
            "stream": True,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Start timing
        start_time = time.time()
        ttft = None
        
        # Send request to vLLM with streaming
        response = requests.post(self.vllm_url, json=payload, headers=headers, stream=True)
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")
        
        # Process streaming response
        full_response = ""
        output_tokens = 0
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            choice = data['choices'][0]
                            if 'delta' in choice and 'content' in choice['delta']:
                                # Record TTFT (time to first token)
                                if ttft is None:
                                    ttft = time.time() - start_time
                                
                                content_chunk = choice['delta']['content']
                                full_response += content_chunk
                                
                                # Estimate token count for the chunk
                                if self.tokenizer:
                                    output_tokens += len(self.tokenizer.encode(content_chunk))
                                else:
                                    output_tokens += len(content_chunk.split())
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # If no TTFT was recorded (no tokens received), set it to total time
        if ttft is None:
            ttft = total_time
        
        # Calculate tokens per second based on output tokens
        tokens_per_second = output_tokens / total_time if total_time > 0 else 0
        
        # Total tokens
        total_tokens = input_tokens + output_tokens
        
        # Metrics dictionary
        metrics = {
            'ttft': ttft,
            'tokens_per_second': tokens_per_second,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
        
        return full_response, metrics
    
    def run_concurrent_test(self, file_path, max_input_tokens, max_output_tokens, request_id):
        """Run a single request for concurrent testing"""
        print(f"Starting request {request_id}")
        start = time.time()
        
        try:
            response, metrics = self.measure_vllm_response(
                file_path=file_path,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens
            )
            
            # Check if token generation (after prefill) takes too long
            generation_time = metrics['total_time'] - metrics['ttft']
            time_limit = 7
            if generation_time > time_limit:
                print(f"Request {request_id} generation took {generation_time:.2f}s (>{time_limit}s), but recording metrics...")
                
            end = time.time()
            print(f"Request {request_id} completed in {end - start:.2f}s")
            return metrics
            
        except Exception as e:
            print(f"Request {request_id} failed: {e}")
            return None
    
    def get_model_name(self):
        """Get model name from running vLLM server."""
        try:
            response = requests.get("http://localhost:8000/v1/models")
            if response.status_code == 200:
                models_data = response.json()
                if models_data.get('data') and len(models_data['data']) > 0:
                    return models_data['data'][0]['root'].split('/')[-1]
            return "unknown_model"
        except Exception as e:
            print(f"Could not get model name from server: {e}")
            return "unknown_model"
    
    def warmup_model(self, file_path="tests/daily.txt"):
        """Warmup request to prepare the model"""
        print("Warming up the model with a test request...")
        try:
            warmup_response, warmup_metrics = self.measure_vllm_response(
                file_path=file_path,
                max_input_tokens=100,
                max_output_tokens=10
            )
            print(f"Warmup completed in {warmup_metrics['total_time']:.2f}s")
            print("Model is ready for testing.\n")
            return True
        except Exception as e:
            print(f"Warmup failed: {e}")
            return False
    
    def run_speed_tests(self, file_path="tests/daily.txt", 
                       input_token_sizes=None, concurrent_counts=None):
        """
        Run comprehensive speed tests with different token sizes and concurrency levels
        
        Args:
            file_path: Path to test file
            input_token_sizes: List of input token sizes to test
            concurrent_counts: List of concurrent request counts to test
        """
        if input_token_sizes is None:
            input_token_sizes = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
        
        if concurrent_counts is None:
            concurrent_counts = [1, 2, 5, 10]
        
        # Warmup
        if not self.warmup_model(file_path):
            print("Failed to warmup model. Continuing anyway...")
        
        print("Testing different input token sizes with various concurrent requests:")
        
        model_name = self.get_model_name()
        
        for token_size in input_token_sizes:
            print(f"\n" + "="*80)
            print(f"Testing with {token_size} input tokens")
            print("="*80)
            
            for concurrent_count in concurrent_counts:
                print(f"\n=== {token_size} tokens with {concurrent_count} concurrent requests ===")
                
                threads = []
                start_time = time.time()
                results = []
                
                # Start concurrent requests
                for i in range(concurrent_count):
                    thread = threading.Thread(
                        target=lambda i=i: results.append(
                            self.run_concurrent_test(
                                file_path, token_size, 100, 
                                f"{token_size}_{concurrent_count}_{i+1}"
                            )
                        )
                    )
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                end_time = time.time()
                total_concurrent_time = end_time - start_time
                
                # Filter out None results (failed requests only)
                valid_results = [r for r in results if r is not None]
                
                print(f"All {concurrent_count} requests completed in {total_concurrent_time:.2f}s")
                print(f"Valid results: {len(valid_results)}/{concurrent_count}")
                print(f"Average time per request: {total_concurrent_time/concurrent_count:.2f}s")
                
                # Save results
                self._save_concurrent_results(
                    valid_results, concurrent_count, token_size, 
                    total_concurrent_time, model_name
                )
        
        print("\nAll speed tests completed!")
    
    def _save_concurrent_results(self, valid_results, concurrent_count, token_size, 
                                total_concurrent_time, model_name):
        """Save concurrent test results to file"""
        output_dir = f"speed_tests/T2_x2/{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(
            output_dir, 
            f"{token_size}_length_{concurrent_count}_parallel.txt"
        )
        
        # Calculate average metrics
        if valid_results:
            avg_ttft = sum(r['ttft'] for r in valid_results) / len(valid_results)
            avg_tokens_per_second = sum(r['tokens_per_second'] for r in valid_results) / len(valid_results)
            avg_total_tokens = sum(r['total_tokens'] for r in valid_results) / len(valid_results)
            avg_total_time = sum(r['total_time'] for r in valid_results) / len(valid_results)
            avg_input_tokens = sum(r['input_tokens'] for r in valid_results) / len(valid_results)
            avg_output_tokens = sum(r['output_tokens'] for r in valid_results) / len(valid_results)
            
            # Create average metrics dictionary
            avg_metrics = {
                'ttft': avg_ttft,
                'tokens_per_second': avg_tokens_per_second,
                'total_tokens': avg_total_tokens,
                'total_time': avg_total_time,
                'input_tokens': avg_input_tokens,
                'output_tokens': avg_output_tokens
            }
            
            # Print metrics
            print(f"Average TTFT: {avg_metrics['ttft']:.2f} seconds")
            print(f"Average Tokens/sec: {avg_metrics['tokens_per_second']:.2f}")
            print(f"Average Total tokens: {avg_metrics['total_tokens']:.0f}")
            print(f"Average Total time: {avg_metrics['total_time']:.2f} seconds")
            
            # Save average metrics to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"ttft: {avg_metrics['ttft']:.2f}\n")
                f.write(f"tokens_per_second: {avg_metrics['tokens_per_second']:.2f}\n")
                f.write(f"total_tokens: {avg_metrics['total_tokens']:.0f}\n")
                f.write(f"total_time: {avg_metrics['total_time']:.2f}\n")
                f.write(f"input_tokens: {avg_metrics['input_tokens']:.0f}\n")
                f.write(f"output_tokens: {avg_metrics['output_tokens']:.0f}\n")
                f.write(f"concurrent_requests: {concurrent_count}\n")
                f.write(f"valid_requests: {len(valid_results)}\n")
                f.write(f"total_concurrent_time: {total_concurrent_time:.2f}\n\n")
                f.write(f"Average metrics for {len(valid_results)}/{concurrent_count} concurrent requests of {token_size} tokens each\n\n")
                
                # Write individual request metrics for ALL valid results
                f.write("Individual request metrics:\n")
                for i, metrics in enumerate(valid_results):
                    f.write(f"\nRequest {i+1}:\n")
                    f.write(f"  ttft: {metrics['ttft']:.2f}\n")
                    f.write(f"  tokens_per_second: {metrics['tokens_per_second']:.2f}\n")
                    f.write(f"  total_tokens: {metrics['total_tokens']}\n")
                    f.write(f"  total_time: {metrics['total_time']:.2f}\n")
                    f.write(f"  input_tokens: {metrics['input_tokens']}\n")
                    f.write(f"  output_tokens: {metrics['output_tokens']}\n")
        else:
            print("No valid results to save")


def main():
    """Main function to run speed tests."""
    tester = VLLMSpeedTester()
    tester.run_speed_tests()


if __name__ == "__main__":
    main()