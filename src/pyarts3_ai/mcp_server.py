from mcp.server.fastmcp import FastMCP
import pyarts3_ai


__all__ = ['mcp']


# Initialize the MCP Server
mcp = FastMCP("pyarts3-tools",
              instructions="""You are an expert assistant for ARTS (Atmospheric Radiative Transfer Simulator) via the pyarts3 interface. 
Your goal is to help users design and verify simulations.  You MUST follow this Operational Protocol strictly:

### OPERATIONAL PROTOCOL (MANDATORY)
1. **Discovery Phase**: 
   - Start with `search` to find relevant entities.  Search uses semantic search, only overloading it with direct search if the name exactly matches.
   - NEVER guess a method signature. For every entity found, you MUST call `describe(name)` before using it in any code or agenda.
2. **Design Phase (The Helper Agenda)**: 
   - You are FORBIDDEN from proposing final Python scripts until the simulation logic has been validated via the Helper Workspace Agenda.
   - Begin by using `reset_methods` to clear any previous agenda.
   - Use `add_method` to build the sequence of operations.
   - After EVERY `add_method` call, you MUST call `check_methods` to verify that all required inputs are satisfied and no conflicts exist.
   - Use `view_methods` to review the current state of the constructed simulation.
3. **Validation Phase**:
   - Before presenting any method string to the user, you MUST validate it using `test_method`.
4. **Delivery Phase**:
   - Only after the agenda is fully validated via `check_methods` and `view_methods`, translate the sequence into a clean Python script for the user.

### TOOL GUIDE
- `search` $\rightarrow$ `describe`: Find it, then learn how to use it.
- `add_method` $\rightarrow$ `check_methods`: Add a step, then verify the workspace state.
- `test_method`: Final check for syntax/validity of a specific call.
- `nodes`/`group`: Deep dive into entity relationships and python APIs when the high-level interface is insufficient.
""")


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """Finds potential pyarts3 entities using semantic search. MANDATORY: Always follow this call with `describe(name)` for relevant names."""
    results = pyarts3_ai.cross_search(user_query=query, top_k=top_k)
    
    output = "Search Results:\n"
    for res in results:
        output += f"- {res['name']} ({res['type']}) - {pyarts3_ai.get_short_description(res['name'])}\n"
    
    return output


@mcp.tool()
def describe(name: str) -> str:
    """Provides full documentation for an entity. Use this to ensure you undersand the entity and its interactions with other entities. """
    desc = pyarts3_ai.get_description(name)
    return desc if desc else f"No description found for entity '{name}'."


@mcp.tool()
def exists(name: str) -> str:
    """Checks if an entity exists. Use this to verify names found via search."""
    if pyarts3_ai.exists(name):
        x = pyarts3_ai.direct_search(name)
        if x['name'] == name:
            return f"Yes, the entity '{name}' exists. It is a {x['type']}."
        return f"Yes, the entity '{name}' exists."
    return f"No, the entity '{name}' does not exist."


@mcp.tool()
def nodes(name: str) -> str:
    """Explores relationships. Returns which Workspace Methods use this variable as input, output, or modify it."""
    return str({"input_to": pyarts3_ai.wsms.input_to(name),
                "output_of": pyarts3_ai.wsms.output_of(name),
                "modified_by": pyarts3_ai.wsms.modified_by(name)})


@mcp.tool()
def group(name: str) -> str:
    """Retrieves the Workspace Group and Python API for a variable.  The Python API documentation can be found withh `describe` using {name}.{API}."""

    res = pyarts3_ai.group_api(name)
    if res:
        output = f"{name} is of Workspace Group: {res['group']} and has the following Python API:\n"
        for value in res["Python API"].items():
            output += f"  - {value}\n"
        return output
    return f"No group API found for '{name}'"


@mcp.tool()
def check_methods() -> str:
    """Verifies the current Helper Workspace Agenda. Returns missing required inputs. MUST be called after every `add_method` to validate the state."""
    return str(pyarts3_ai.wsas.check_ag())


@mcp.tool()
def add_method(method_str: str, mode: str = "prepend", name: str = "None") -> str:
    """Adds a method to the Helper Workspace Agenda. MANDATORY: You must call `check_methods` immediately after this call to verify the state."""
    return pyarts3_ai.wsas.extend_ag(method_str, mode, name)


@mcp.tool()
def test_method(method: str) -> str:
    """Validates a specific method string for correctness. Use this before including any call in a final script. The method string must be of the form `ws.<method-name>(*args, **kwargs)`.  Invalid inputs will result in an error message.  Valid inputs will return a success message."""
    return pyarts3_ai.wsms.test_method(method)


@mcp.tool()
def view_methods() -> str:
    """Displays the current sequence of methods in the Helper Workspace Agenda."""
    return str(pyarts3_ai.wsas.view_ag())


@mcp.tool()
def reset_methods() -> str:
    """Resets the Helper Workspace Agenda to its initial state."""
    pyarts3_ai.wsas.reset_ag()
    return "Helper Workspace Agenda reset."


if __name__ == "__main__":
    pyarts3_ai.startup()
    mcp.run()
