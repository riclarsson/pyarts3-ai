import pyarts3_ai

if __name__ == "__main__":

  help = """Commands:
- Type 'help' to display this help message.
- Type 'exit' to quit the tool.
- Type 'describe' to get the description of a specific entity by name.
- Type 'short' to get the short description of a specific entity by name.
- Type 'group' to get API information from a workspace variable or workspace group.
- Type 'top-k' to specify the number of top results to return (default is 5).

Otherwise just enter your search query and press Enter to see the results."""

  print("Starting server...")
  pyarts3_ai.startup()
  print (f"""Welcome to the pyarts3-ai interactive search tool!

You can enter a search query to find the top-k relevant pyarts3 entities.
The search uses a semantic engine, so you can use normal language queries.

{help}
""")

  top_k = 5

  while True:
    query = input(">>> ")
    if query.lower() == 'exit':
        break
  
    if query.lower() == 'describe':
        name = input("describe >>> ")
        name = name.replace(' ', '')
        print (f"Getting description for entity '{name}':\n")
        print(pyarts3_ai.get_description(name))
    elif query.lower() == 'short':
        name = input("short >>> ")
        name = name.replace(' ', '')
        print (f"Getting short description for entity '{name}':\n")
        print(pyarts3_ai.get_short_description(name))
    elif query.lower() == 'group':
        name = input("group >>> ")
        name = name.replace(' ', '')
        print (f"Getting group API for entity '{name}':\n")
        print(pyarts3_ai.group_api(name))
    elif query.lower() == 'top-k':
        try:
            top_k = max(1, int(input("Enter the number of top results to return: ")))
        except Exception as e:
            print(f"Invalid input. Please enter a valid integer for top-k.  Error: {e}")
    elif query.lower() == 'help':
        print(help)
    else:
        results = pyarts3_ai.cross_search(query, top_k)
        print("Search results:")
        for res in results:
          print(f"{res['name']} ({res['type']}) - {pyarts3_ai.get_short_description(res['name'])}")
