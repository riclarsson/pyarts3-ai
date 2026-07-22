# pyarts3_ai

`pyarts3_ai` is a specialized interface designed to expose the **ARTS (Atmospheric Radiative Transfer Simulator)** workspace to AI agents. It transforms the ARTS API from a static library into a searchable semantic space, allowing LLMs to interactively discover and construct working simulation runs.

## Installation

```bash
pip install -e .
```

### Dependencies

See YAML file

### Quick-use

```python
import pyarts3_ai as ai

ai.startup()
res = ai.cross_search("Optimal Estimation")

print(res[0]['type'])
print(res[0]['name'])
```

### Start an MCP server

Model Context Protocol servers allow AIs with tool support to execute
select commands inside their workflow.

## VSCODE

Search for adding MCP servers.  If you have done it before, add only the server `pyarts3-tools`.

A full example if nothing exists would read:

```json
{
	"servers": {
		"pyarts3-tools": {
			"type": "stdio",
			"command": "python3",
			"args": [
				"-m",
				"pyarts3_ai.mcp_server"
			]
		}
	},
	"inputs": []
}
```

Please ensure `pyarts3_ai` is properly installed in the executing environment.
