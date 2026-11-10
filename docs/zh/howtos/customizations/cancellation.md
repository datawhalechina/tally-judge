# 取消长时间运行的任务 { #cancelling-long-running-tasks }

处理大型数据集或复杂评估时，部分 Ragas 操作可能耗时很长。取消功能允许你在需要时平稳终止这些任务，对生产环境尤为重要。

## 概览 { #overview }

Ragas 为以下操作提供取消支持。
- **`evaluate()`**，使用指标评估数据集
- **`generate_with_langchain_docs()`**，从文档生成测试集

取消机制是线程安全的，会尽可能平稳终止任务并保留部分结果。

## 基本用法 { #basic-usage }

### 可取消的评估 { #cancellable-evaluation }

可以先获取支持取消的执行器，再启动评估。

```py
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset

# Your dataset and metrics
dataset = EvaluationDataset(...)
metrics = [...]

# Get executor instead of running evaluation immediately
executor = evaluate(
    dataset=dataset,
    metrics=metrics,
    return_executor=True  # Key parameter
)

# Now you can:
# - Cancel: executor.cancel()
# - Check status: executor.is_cancelled()
# - Get results: executor.results()  # This blocks until completion
```

### 可取消的测试集生成 { #cancellable-test-set-generation }

测试集生成也采用类似方式。

```py
from ragas.testset.synthesizers.generate import TestsetGenerator

generator = TestsetGenerator(...)

# Get executor for cancellable generation
executor = generator.generate_with_langchain_docs(
    documents=documents,
    testset_size=100,
    return_executor=True  # Allow access to Executor to cancel
)

# Use the same cancellation interface
executor.cancel()
```

## 生产环境用法 { #production-patterns }

### 1. 超时取消 { #1-timeout-pattern }

自动取消超过时间限制的操作。

```py
import threading
import time

def evaluate_with_timeout(dataset, metrics, timeout_seconds=300):
    """Run evaluation with automatic timeout."""
    # Get cancellable executor
    executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)

    results = None
    exception = None

    def run_evaluation():
        nonlocal results, exception
        try:
            results = executor.results()
        except Exception as e:
            exception = e

    # Start evaluation in background thread
    thread = threading.Thread(target=run_evaluation)
    thread.start()

    # Wait for completion or timeout
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"Evaluation exceeded {timeout_seconds}s timeout, cancelling...")
        executor.cancel()
        thread.join(timeout=10)  # Custom timeout as per need
        return None, "timeout"

    return results, exception

# Usage
results, error = evaluate_with_timeout(dataset, metrics, timeout_seconds=600)
if error == "timeout":
    print("Evaluation was cancelled due to timeout")
else:
    print(f"Evaluation completed: {results}")
```

### 2. 信号处理（Ctrl+C） { #2-signal-handler-pattern-ctrlc }

允许用户通过键盘中断取消任务。

```py
import signal
import sys

def setup_cancellation_handler():
    """Set up graceful cancellation on Ctrl+C."""
    executor = None

    def signal_handler(signum, frame):
        if executor and not executor.is_cancelled():
            print("\nReceived interrupt signal, cancelling evaluation...")
            executor.cancel()
            print("Cancellation requested. Waiting for graceful shutdown...")
        sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    return lambda exec: setattr(signal_handler, 'executor', exec)

# Usage
set_executor = setup_cancellation_handler()

executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
set_executor(executor)

print("Running evaluation... Press Ctrl+C to cancel gracefully")
try:
    results = executor.results()
    print("Evaluation completed successfully")
except KeyboardInterrupt:
    print("Evaluation was cancelled")
```

### 3. Web 应用 { #3-web-application-pattern }

在 Web 应用中，请求中止时可以取消对应操作。

```py
from flask import Flask, request
import threading
import uuid

app = Flask(__name__)
active_evaluations = {}

@app.route('/evaluate', methods=['POST'])
def start_evaluation():
    # Create unique evaluation ID
    eval_id = str(uuid.uuid4())

    # Get dataset and metrics from request
    dataset = get_dataset_from_request(request)
    metrics = get_metrics_from_request(request)

    # Start cancellable evaluation
    executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
    active_evaluations[eval_id] = executor

    # Start evaluation in background
    def run_eval():
        try:
            results = executor.results()
            # Store results somewhere
            store_results(eval_id, results)
        except Exception as e:
            store_error(eval_id, str(e))
        finally:
            active_evaluations.pop(eval_id, None)

    threading.Thread(target=run_eval).start()

    return {"evaluation_id": eval_id, "status": "started"}

@app.route('/evaluate/<eval_id>/cancel', methods=['POST'])
def cancel_evaluation(eval_id):
    executor = active_evaluations.get(eval_id)
    if executor:
        executor.cancel()
        return {"status": "cancelled"}
    return {"error": "Evaluation not found"}, 404
```

## 进阶用法 { #advanced-usage }

### 检查取消状态 { #checking-cancellation-status }

```py
executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)

# Start in background
def monitor_evaluation():
    while not executor.is_cancelled():
        print("Evaluation still running...")
        time.sleep(5)
    print("Evaluation was cancelled")

threading.Thread(target=monitor_evaluation).start()

# Cancel after some condition
if some_condition():
    executor.cancel()
```

### 部分结果 { #partial-results }

如果任务在执行过程中被取消，仍可能获取部分结果。

```py
executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)

try:
    results = executor.results()
    print(f"Completed {len(results)} evaluations")
except Exception as e:
    if executor.is_cancelled():
        print("Evaluation was cancelled - may have partial results")
    else:
        print(f"Evaluation failed: {e}")
```

### 自定义取消逻辑 { #custom-cancellation-logic }

```py
class EvaluationManager:
    def __init__(self):
        self.executors = []

    def start_evaluation(self, dataset, metrics):
        executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
        self.executors.append(executor)
        return executor

    def cancel_all(self):
        """Cancel all running evaluations."""
        for executor in self.executors:
            if not executor.is_cancelled():
                executor.cancel()
        print(f"Cancelled {len(self.executors)} evaluations")

    def cleanup_completed(self):
        """Remove completed executors."""
        self.executors = [ex for ex in self.executors if not ex.is_cancelled()]

# Usage
manager = EvaluationManager()

# Start multiple evaluations
exec1 = manager.start_evaluation(dataset1, metrics)
exec2 = manager.start_evaluation(dataset2, metrics)

# Cancel all if needed
manager.cancel_all()
```

## 最佳实践 { #best-practices }

### 1. 为生产环境设置超时 { #1-always-use-timeouts-in-production }
```py
# Good: Always set reasonable timeouts
results, error = evaluate_with_timeout(dataset, metrics, timeout_seconds=1800)  # 30 minutes

# Avoid: Indefinite blocking
results = executor.results()  # Could block forever
```

### 2. 平稳处理取消 { #2-handle-cancellation-gracefully }
```py
try:
    results = executor.results()
    process_results(results)
except Exception as e:
    if executor.is_cancelled():
        log_cancellation()
        cleanup_partial_work()
    else:
        log_error(e)
        handle_failure()
```

### 3. 向用户反馈状态 { #3-provide-user-feedback }
```py
def run_with_progress_and_cancellation(executor):
    print("Starting evaluation... Press Ctrl+C to cancel")

    # Monitor progress in background
    def show_progress():
        while not executor.is_cancelled():
            # Show some progress indication
            print(".", end="", flush=True)
            time.sleep(1)

    progress_thread = threading.Thread(target=show_progress)
    progress_thread.daemon = True
    progress_thread.start()

    try:
        return executor.results()
    except KeyboardInterrupt:
        print("\nCancelling...")
        executor.cancel()
        return None
```

### 4. 清理资源 { #4-clean-up-resources }
```py
def managed_evaluation(dataset, metrics):
    executor = None
    try:
        executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
        return executor.results()
    except Exception as e:
        if executor:
            executor.cancel()
        raise
    finally:
        # Clean up any temporary resources
        cleanup_temp_files()
```

## 限制 { #limitations }

- **异步操作**。取消发生在任务级，无法中止正在进行的单次 LLM 调用。
- **部分状态**。取消后可能留下部分结果或临时文件。
- **生效时机**。取消采用协作机制，任务需要定期检查取消状态。
- **外部依赖**。部分外部服务可能无法立即响应取消。

## 故障排查 { #troubleshooting }

### 取消未生效 { #cancellation-not-working }
```py
# Check if cancellation is set
if executor.is_cancelled():
    print("Cancellation was requested")
else:
    print("Cancellation not requested yet")

# Ensure you're calling cancel()
executor.cancel()
assert executor.is_cancelled()
```

### 取消后任务仍在运行 { #tasks-still-running-after-cancellation }
```py
# Give time for graceful shutdown
executor.cancel()
time.sleep(2)  # Allow tasks to detect cancellation

# Force cleanup if needed
import asyncio
try:
    loop = asyncio.get_running_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()
except RuntimeError:
    pass  # No event loop running
```

取消功能让长时间运行的 Ragas 操作更易控制，有助于在生产部署中管理资源并改善用户体验。
