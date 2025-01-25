import queue
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TaskQueue:
    def __init__(self, num_workers=4):
        self.queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.logger = logging.getLogger("TaskQueue")

    def enqueue(self, func, *args, **kwargs):
        """Adds a task to the queue."""
        future = self.executor.submit(self._run_task, func, *args, **kwargs)
        self.logger.info(f"Task submitted: {func.__name__} with args: {args}, kwargs: {kwargs}")
        return future
    
    def _run_task(self, func, *args, **kwargs):
        """Runs a task, handling exceptions and retries."""
        retries = kwargs.pop('retries', 3)  # Default to 3 retries
        retry_delay = kwargs.pop('retry_delay', 5)  # Default retry delay of 5 seconds
        task_id = kwargs.pop('task_id', None)

        for attempt in range(retries):
            try:
                self.logger.info(f"Running task {func.__name__} (attempt {attempt + 1})")
                if task_id:
                    result = func(*args, **kwargs, task_id=task_id)
                else:
                    result = func(*args, **kwargs)
                self.logger.info(f"Task {func.__name__} completed successfully")
                return result
            except Exception as e:
                self.logger.error(f"Task {func.__name__} failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    self.logger.info(f"Retrying task {func.__name__} in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Task {func.__name__} failed after multiple retries.")
                    # Optionally move the task to a dead-letter queue here
                    # self._move_to_dead_letter_queue(func, *args, **kwargs)
                    raise  # Re-raise the exception to signal failure to the caller
    
    def _move_to_dead_letter_queue(self, func, *args, **kwargs):
        """ Placeholder function to move failed tasks to a dead-letter queue """
        self.logger.error(f"Placeholder: Moving task {func.__name__} to dead-letter queue with args {args} and kwargs {kwargs}.")

    def shutdown(self):
        """Shuts down the task queue."""
        self.executor.shutdown(wait=True)
        self.logger.info("Task queue shut down.")

# Example usage (you can create a global instance or pass it around):
task_queue = TaskQueue()