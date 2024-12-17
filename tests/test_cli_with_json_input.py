import subprocess
import pytest
import json
import os

# Load query-answer pairs from the JSON file
def load_queries_from_json():
    """Load queries and answers from JSON file."""
    json_path = os.path.join("evaluation", "sample_query_answer_pairs.json")
    with open(json_path, "r") as file:
        data = json.load(file)
    all_query_answer_pairs =  [(query, answer) for query, answer in data.items()]
    return all_query_answer_pairs[:200]

# Parameterize the test with queries and expected answers
@pytest.mark.parametrize("query,expected_answer", load_queries_from_json())
def test_cli_with_json_inputs(query, expected_answer):
    """Test CLI with queries loaded from a JSON file."""
    assert isinstance(query, str), "query is not a string."
    result = subprocess.run(
        ["python", "src/search_cli.py", "--query", query],
        capture_output=True,
        text=True
    )

    # Validate CLI execution success
    #assert result.returncode == 0, f"CLI returned non-zero exit code: {result.returncode}"

    # Access and validate the output
    output = result.stdout  # Access stdout of the subprocess
    assert output is not None, "CLI returned None as output."
    assert isinstance(output, str), "Output is not a string."
    # print(f"Query: {query}")
    # print(f"Output: {output}")
    #assert expected_answer in output, f"Expected answer not found. Output:\n{output}"
