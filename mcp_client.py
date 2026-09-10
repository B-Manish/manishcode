"""
Minimal MCP client that bridges any MCP server's tools to a local Ollama model.

Setup:
    pip install mcp ollama

Usage example (connects to the official MCP filesystem server via npx):
    python mcp_client.py --server-cmd "npx -y @modelcontextprotocol/server-filesystem C:\\Users\\manis\\Documents"

Then chat interactively. The model will call tools as needed.
"""

import argparse
import asyncio
import shlex

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "qwen3:8b"


def mcp_tools_to_ollama_tools(mcp_tools):
    """Convert MCP tool definitions into the format Ollama's tool-calling API expects."""
    ollama_tools = []
    for tool in mcp_tools:
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )
    return ollama_tools


async def run_chat_loop(server_cmd: str):
    args = shlex.split(server_cmd)
    server_params = StdioServerParameters(command=args[0], args=args[1:])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools from the MCP server
            tools_response = await session.list_tools()
            ollama_tools = mcp_tools_to_ollama_tools(tools_response.tools)

            print(f"Connected. {len(ollama_tools)} tool(s) available:")
            for t in ollama_tools:
                print(f"  - {t['function']['name']}: {t['function']['description']}")

            messages = []

            print("\nType your message ('quit' to exit).\n")
            while True:
                user_input = input(">>> ").strip()
                if user_input.lower() in ("quit", "exit"):
                    break

                messages.append({"role": "user", "content": user_input})

                # Keep looping while the model wants to call tools
                while True:
                    response = ollama.chat(
                        model=MODEL,
                        messages=messages,
                        tools=ollama_tools,
                        think=False,  # flip to True if you want reasoning traces
                    )
                    msg = response["message"]
                    messages.append(msg)

                    tool_calls = msg.get("tool_calls") or []
                    if not tool_calls:
                        # Final answer, no more tools needed
                        print(f"\n{msg.get('content', '')}\n")
                        break

                    # Execute each requested tool call against the MCP server
                    for call in tool_calls:
                        name = call["function"]["name"]
                        args_dict = call["function"]["arguments"]
                        print(f"  [calling tool] {name}({args_dict})")

                        result = await session.call_tool(name, arguments=args_dict)
                        result_text = "\n".join(
                            block.text for block in result.content if hasattr(block, "text")
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "content": result_text,
                                "name": name,
                            }
                        )
                    # Loop back: model sees tool result(s) and decides next step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-cmd",
        required=True,
        help='Command to launch the MCP server, e.g. "npx -y @modelcontextprotocol/server-filesystem C:\\path"',
    )
    args = parser.parse_args()
    asyncio.run(run_chat_loop(args.server_cmd))


if __name__ == "__main__":
    main()