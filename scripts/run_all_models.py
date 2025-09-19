#!/usr/bin/env python3
"""
Automation script to test multiple Qwen3 models sequentially.

starts vLLM server for each model, runs both test scripts, then moves to the next model.
"""

import subprocess
import time
import os
import sys
import signal
import requests
from pathlib import Path


class VLLMModelTester:
    """Class to manage vLLM servers and run tests for multiple models."""
    
    def __init__(self):
        self.models = [
            {"name": "Qwen/Qwen3-0.6B", "max_model_len": 32768},
            {"name": "Qwen/Qwen3-1.7B", "max_model_len": 32768},
            {"name": "Qwen/Qwen3-4B", "max_model_len": 32768},
            {"name": "Qwen/Qwen3-8B", "max_model_len": 32768},
            {"name": "zhiqing/Qwen3-14B-INT8", "max_model_len": 28000},
        ]
        self.current_process = None
        self.port = 8000
        self.cuda_devices = "0,1"
        self.tensor_parallel_size = 2
    
    def _wait_for_server(self, timeout=300):
        """Wait for vLLM server to be ready"""
        print("Waiting for vLLM server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/v1/models", timeout=10)
                if response.status_code == 200:
                    print("vLLM server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print(".", end="", flush=True)
            time.sleep(5)
        
        print(f"\nTimeout waiting for server after {timeout} seconds")
        return False
    
    def _start_vllm_server(self, model_name, max_model_len=None):
        """Start vLLM server for a specific model"""
        print(f"\n{'='*80}")
        print(f"Starting vLLM server for model: {model_name}")
        print(f"{'='*80}")
        
        # Build command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--served-model-name", "qwen3",
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--port", str(self.port),
            "--max-num-seqs", "10",
            "--gpu-memory-utilization", "0.9",
            "--no-enable-prefix-caching"
        ]
        
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])
        
        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        
        print(f"Command: CUDA_VISIBLE_DEVICES={self.cuda_devices} {' '.join(cmd)}")
        
        # Start process
        try:
            self.current_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait for server to be ready
            if self._wait_for_server():
                return True
            else:
                self._stop_vllm_server()
                return False
                
        except Exception as e:
            print(f"Failed to start vLLM server: {e}")
            return False
    
    def _stop_vllm_server(self):
        """Stop the current vLLM server"""
        if self.current_process:
            print("\nStopping vLLM server...")
            try:
                # Send SIGTERM first
                self.current_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.current_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop gracefully
                    print("Force killing vLLM server...")
                    self.current_process.kill()
                    self.current_process.wait()
                
                print("vLLM server stopped.")
                
            except Exception as e:
                print(f"Error stopping server: {e}")
            
            self.current_process = None
        
        # Wait a bit for port to be released
        time.sleep(5)
    
    def _run_test_script(self, script_name):
        """Run a test script"""
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            print(f"Error: Script {script_path} not found!")
            return False
        
        print(f"\n{'-'*60}")
        print(f"Running {script_name}...")
        print(f"{'-'*60}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=False,
                text=True
            )
            print(f"\n{script_name} completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\nError running {script_name}: {e}")
            return False
        except Exception as e:
            print(f"\nUnexpected error running {script_name}: {e}")
            return False
    
    def _test_model(self, model_info):
        """Test a single model with both scripts"""
        model_name = model_info["name"]
        max_model_len = model_info["max_model_len"]
        
        print(f"\n{'#'*100}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'#'*100}")
        
        # Start vLLM server
        if not self._start_vllm_server(model_name, max_model_len):
            print(f"Failed to start server for {model_name}. Skipping...")
            return False
        
        success = True
        
        try:
            # Run summary script
            if not self._run_test_script("get_full_summary.py"):
                print(f"Summary script failed for {model_name}")
                success = False
            
            # Run speed metrics script
            if not self._run_test_script("get_speed_metrics.py"):
                print(f"Speed metrics script failed for {model_name}")
                success = False
                
        finally:
            # Always stop the server
            self._stop_vllm_server()
        
        if success:
            print(f"\n✅ Successfully tested {model_name}")
        else:
            print(f"\n❌ Some tests failed for {model_name}")
        
        return success
    
    def run_all_tests(self):
        """Run tests for all models"""
        print("🚀 Starting automated testing of all Qwen3 models")
        print(f"Models to test: {len(self.models)}")
        
        results = {}
        
        for i, model_info in enumerate(self.models, 1):
            model_name = model_info["name"]
            print(f"\n\n{'='*100}")
            print(f"PROGRESS: {i}/{len(self.models)} - {model_name}")
            print(f"{'='*100}")
            
            try:
                success = self._test_model(model_info)
                results[model_name] = success
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Testing interrupted by user")
                self._stop_vllm_server()
                break
            except Exception as e:
                print(f"\n❌ Unexpected error testing {model_name}: {e}")
                results[model_name] = False
                self._stop_vllm_server()
        
        # Print final summary
        self._print_summary(results)
    
    def _print_summary(self, results):
        """Print final test summary"""
        print(f"\n\n{'='*100}")
        print("🏁 FINAL TESTING SUMMARY")
        print(f"{'='*100}")
        
        successful = 0
        failed = 0
        
        for model_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{model_name:<30} {status}")
            if success:
                successful += 1
            else:
                failed += 1
        
        print(f"\n📊 Results: {successful} successful, {failed} failed out of {len(results)} models")
        
        if failed == 0:
            print("🎉 All models tested successfully!")
        else:
            print(f"⚠️  {failed} model(s) had issues during testing")


def main():
    """Main function"""
    # Check if we're in the right directory
    if not os.path.exists("tests"):
        print("Error: 'tests' directory not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Create output directories
    os.makedirs("summary_results/T2_x2", exist_ok=True)
    os.makedirs("speed_tests/T2_x2", exist_ok=True)
    
    tester = VLLMModelTester()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        print("\n\n⚠️  Received interrupt signal. Stopping...")
        tester._stop_vllm_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        tester.run_all_tests()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        tester._stop_vllm_server()
        sys.exit(1)


if __name__ == "__main__":
    main()
