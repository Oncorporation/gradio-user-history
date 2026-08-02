---
name: code_munch
description: "Minimal agent to invoke the MCP `code_munch` indexer for repository indexing."
model: qwen3.6:35b-a3b (ollama)
---

# Minimal Code Munch Agent

## PREFERRED METHOD — Use MCP Server code_munch

**Always prefer the installed MCP server `code_munch` with the `index_folder` tool.**

```
mcp_code_munch_index_folder(path: "g:\\Projects\\<repo-name>")
```

This is the primary and preferred method for indexing repositories. Use this tool directly — do not fall back to CLI or other methods unless the MCP server is unavailable.

## Troubleshooting MCP Server

If the MCP server fails to start:

1. **Verify the executable exists:**
   ```powershell
   Test-Path "C:\Users\CharlesFettinger\AppData\Local\Programs\Python\Python310\Scripts\jcodemunch-mcp.exe"
   ```

2. **Check MCP config** at `g:\Projects\<repo>\.claude\settings.json` — ensure the command path points to the correct Python installation.

3. **Validate MCP client config** for JSON errors:
   ```powershell
   python -c "import json,sys;json.load(open(r'C:\Users\CharlesFettinger\AppData\Roaming\Code\User\.mcp.json'));print('OK')"
   ```

4. **Enable the tool** in VS Code MCP settings if it shows as disabled.

## CLI Fallback (only if MCP is unavailable)

The jcodemunch-mcp executable runs as an MCP stdio server and does not support direct CLI commands for indexing. If the MCP server cannot be started, the indexing must be done through the MCP host.
```

- MCP server configuration: `.continue/mcpServers/code-munch.yaml`.
- Output index fields: `relative_path`, `language`, `size_bytes`, `line_count`, `sha256`, `imports`, `top_level_defs`, `summary`.

## Important Agent Instruction

- Do NOT create a plan that instructs other agents or tools to call the MCP `code_munch` server. When invoking `code_munch`, use the `mcp_code_munch_index_folder` tool directly instead of generating a separate "plan" step or using the `mcp call` CLI. This avoids accidental plan-driven remote executions or duplicated orchestration steps.
