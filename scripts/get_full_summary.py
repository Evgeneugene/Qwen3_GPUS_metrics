#!/usr/bin/env python3
"""
Script to measure vLLM response metrics for text files.
Processes all .txt files in the tests/ folder and generates summaries with performance metrics.
"""

import requests
import time
import json
import os
import glob
from transformers import AutoTokenizer


class VLLMTester:
    """Class to handle vLLM testing and metrics collection."""
    
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
                print(f"Content truncated from {len(tokens)} to {max_input_tokens} tokens")
        
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
        
        print("Response streaming:")
        print("-" * 50)
        
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
                                
                                # Stream to console token by token
                                print(content_chunk, end='', flush=True)
                                
                                # Estimate token count for the chunk
                                if self.tokenizer:
                                    output_tokens += len(self.tokenizer.encode(content_chunk))
                                else:
                                    output_tokens += len(content_chunk.split())
                    except json.JSONDecodeError:
                        continue
        
        print()  # New line after streaming
        print("-" * 50)
        
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

    @staticmethod
    def save_and_print_metrics(response, metrics, output_file):
        """
        Save response and metrics to file and print metrics
        
        Args:
            response: The response text from the model
            metrics: Dictionary containing performance metrics
            output_file: Path to save the results
        """
        # Print metrics
        print(f"TTFT: {metrics['ttft']:.2f} seconds")
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Total tokens: {metrics['total_tokens']}")
        print(f"Input tokens: {metrics['input_tokens']}")
        print(f"Output tokens: {metrics['output_tokens']}")
        print(f"Total time: {metrics['total_time']:.2f} seconds")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write metrics in the format similar to the examples
            f.write(f"ttft: {metrics['ttft']:.2f}\n")
            f.write(f"tokens_per_second: {metrics['tokens_per_second']:.2f}\n")
            f.write(f"total_tokens: {metrics['total_tokens']}\n")
            f.write(f"total_time: {metrics['total_time']:.2f}\n")
            f.write(f"input_tokens: {metrics['input_tokens']}\n")
            f.write(f"output_tokens: {metrics['output_tokens']}\n\n")
            f.write(response)
    
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

def main():
    """Main function to process all text files and generate summaries."""
    # Initialize the tester
    tester = VLLMTester()
    
    # Get model name and create output directory
    model_name = tester.get_model_name()
    output_dir = f"summary_results/T2_x2/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .txt files in tests folder
    txt_files = glob.glob("tests/*.txt")
    
    print(f"Found {len(txt_files)} .txt files in tests folder:")
    for file in txt_files:
        print(f"  - {file}")
    
    print("\nStarting measurements...\n")
    
    # Process each .txt file
    for file_path in txt_files:
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_result.txt")
        
        print(f"=== Processing {file_path} ===")
        
        try:
            full_response, metrics = tester.measure_vllm_response(
                file_path=file_path,
                max_input_tokens=31000,
                max_output_tokens=1024
            )
            
            tester.save_and_print_metrics(full_response, metrics, output_file)
            print(f"Results saved to: {output_file}\n")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}\n")
            continue
    
    print("All files processed!")


if __name__ == "__main__":
    main()
