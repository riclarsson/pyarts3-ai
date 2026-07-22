from mcp.server.fastmcp import FastMCP
import pyarts3_ai


__all__ = ['mcp']


# Initialize the MCP Server
mcp = FastMCP("pyarts3-tools",
              instructions="""Tools to search through the public interface of ARTS, via it's pyarts3 python interface.

ARTS is a radiative transfer simulator that can be used to simulate the propagation of electromagnetic waves through the atmosphere.
It is widely used in atmospheric science, remote sensing, climate research, and planetary science.

The tools allow you to search through the pyarts3 entities,
which include WSVs (Workspace Variables), WSGs (Workspace Groups), and other entities.
The search is semantic, allowing for natural language queries. The results are sorted by relevance based on cross-encoder scoring.

Once you have the name of an entity, you can use the `describe` tool to get its description, though this requires an exact name match,
which conveniently you know from the search results. The `describe` tool will return an empty string if the entity does not exist.

The `exists` tool can be used to check if an entity exists by name.

CRITICAL WORKFLOW:
1. Use `search` to find potential entities by human language.
2. Once you have a list of names, YOU MUST use the `describe` tool on the most relevant 
   entity/entities to get their actual purpose and usage before providing a final answer. 
   Do not assume the name is self-explanatory.
3. Use `exists` only if you are unsure if a specific name provided by the user is valid."""
              )


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """ Searches across all pyarts3 entities and returns the top-k results. """
    results = pyarts3_ai.cross_search(user_query=query, top_k=top_k)

    return str([{'name': res['name'], 'type': res['type']} for res in results])


@mcp.tool()
def describe(name: str) -> str:
    """ Returns the description of a specific pyarts3 entity from its name. """
    desc = pyarts3_ai.get_description(name)
    return desc if desc else f"No description found for entity '{name}'."


@mcp.tool()
def exists(name: str) -> str:
    """ Checks if a specific pyarts3 entity exists. """
    if pyarts3_ai.exists(name):
        return f"Yes, the entity '{name}' exists."
    return f"No, the entity '{name}' does not exist."


if __name__ == "__main__":
    pyarts3_ai.startup()
    mcp.run()
