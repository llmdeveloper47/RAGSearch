import os
from locust import User, task, between
import subprocess
import random
import json

# Load queries from JSON

json_path = os.path.join("evaluation", "sample_query_answer_pairs.json")
with open(json_path, "r") as file:
    queries = list(json.load(file).keys())

queries = queries[:200]

class CLILoadTestUser(User):
    """Simulates a user calling the CLI."""
    wait_time = between(1, 3)  # Wait time between tasks (seconds)

    @task
    def run_cli(self):
        """Task to execute the CLI with a random query."""
        query = random.choice(queries)  # Pick a random query
        result = subprocess.run(
            ["python", "src/search_cli.py", "--query", query],
            capture_output=True,
            text=True
        )

        # Log errors or validate output
        output = result.stdout  # Access stdout of the subprocess
        assert output is not None, "CLI returned None as output."
        assert isinstance(output, str), "Output is not a string."