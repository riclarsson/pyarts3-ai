# pyarts3_ai

`pyarts3_ai` is a specialized interface designed to expose the **ARTS (Atmospheric Radiative Transfer Simulator)** workspace to AI agents. It transforms the ARTS API from a static library into a searchable semantic space, allowing LLMs to interactively discover and construct working simulation runs.

## Core Features

### Semantic Discovery
Instead of relying on exact keyword matches, `pyarts3_ai` uses state-of-the-art embedding models to find workspace entities based on physical concepts.
*   **Two-Stage Retrieval**: Combines fast Bi-Encoder retrieval (Cosine Similarity) with high-precision Cross-Encoder re-ranking.
*   **Granular Indexing**: Implements sentence-level splitting to prevent "information dilution" in long descriptions, ensuring specific technical terms are not lost in large blocks of text.
*   **Hybrid Search Ready**: Designed to support both conceptual meaning and exact identifier matches.

### Unified Workspace Interface
The library provides a single entry point to interact with all ARTS workspace entities:
- **Variables (`wsvs`)**: State variables within the simulation.
- **Methods (`wsms`)**: Functions that manipulate or produce variables.
- **Agendas (`wsas`)**: Sequences of methods.
- **Groups (`wsgs`)** and **Scenarios (`wssns`)**.

### Bottom-Up Reasoning Support
Designed to facilitate an agentic workflow where the AI:
1. Searches for a target output variable (e.g., "spectral radiance").
2. Identifies the methods that produce that variable.
3. Recursively discovers the required input variables and their producers until a complete, valid execution DAG is formed.

## Installation

```bash
pip install -e .
```

### Dependencies

Dependency | Purpose |
|-----|-----|
| pyarts3 | Core ARTS simulation library|
| sentence-transformers | Generates embeddings and performs re-ranking |
| nltk | Handles sentence tokenization for granular indexing |
| scikit-learn | Provides cosine similarity calculations |
| numpy | Numerical operations for vector manipulation |

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
