"""
A/B test tool-calling reliability across local Ollama models.

Usage:
    pip install ollama
    python test_tool_calling.py
"""

import ollama

MODELS_TO_TEST = [
    "qwen2.5:7b-instruct-q4_0",
    "qwen3:8b",
]

# Define a couple of fake tools. Each test case says which tool call
# we EXPECT the model to make (or None if it should NOT call a tool).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search local files by keyword",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

TEST_CASES = [
    {
        "prompt": "What's the weather like in Tokyo right now?",
        "expected_tool": "get_weather",
    },
    {
        "prompt": "Find any files on my system that mention 'invoice'.",
        "expected_tool": "search_files",
    },
    {
        "prompt": "What's the capital of France?",
        "expected_tool": None,  # should NOT call a tool for this
    },
    {
        "prompt": "Can you check the weather in Osaka and also search my files for 'resume'?",
        "expected_tool": "multiple",  # ideally calls both tools
    },
]


def run_test(model, case):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": case["prompt"]}],
        tools=TOOLS,
    )

    tool_calls = response["message"].get("tool_calls") or []
    called_names = [tc["function"]["name"] for tc in tool_calls]

    expected = case["expected_tool"]
    if expected is None:
        passed = len(called_names) == 0
    elif expected == "multiple":
        passed = len(called_names) >= 2
    else:
        passed = expected in called_names

    return {
        "prompt": case["prompt"],
        "expected": expected,
        "called": called_names,
        "passed": passed,
        "raw_content": response["message"].get("content", ""),
    }


def main():
    for model in MODELS_TO_TEST:
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model}")
        print("=" * 60)

        results = []
        for case in TEST_CASES:
            try:
                result = run_test(model, case)
            except Exception as e:
                result = {
                    "prompt": case["prompt"],
                    "expected": case["expected_tool"],
                    "called": [],
                    "passed": False,
                    "raw_content": f"ERROR: {e}",
                }
            results.append(result)

            status = "PASS" if result["passed"] else "FAIL"
            print(f"\n[{status}] {result['prompt']}")
            print(f"  expected: {result['expected']}")
            print(f"  called:   {result['called']}")
            if not result["called"]:
                print(f"  model said instead: {result['raw_content'][:150]}")

        passed_count = sum(r["passed"] for r in results)
        print(f"\n>>> {model}: {passed_count}/{len(results)} passed")


if __name__ == "__main__":
    main()