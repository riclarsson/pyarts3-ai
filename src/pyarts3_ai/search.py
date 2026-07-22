import pyarts3_ai

if __name__ == "__main__":

  help = """Commands:
- Type 'help' to display this help message.
- Type 'exit' to quit the tool.
- Type 'describe' to get the description of a specific entity by name.
- Type 'top-k' to specify the number of top results to return (default is 5).

Otherwise just enter your search query and press Enter to see the results."""

  print("Starting server...")
  pyarts3_ai.startup()
  print (f"""Welcome to the pyarts3-ai interactive search tool!

You can enter a search query to find the top-k relevant pyarts3 entities.
The search uses a semantic engine, so you can use normal language queries.

{help}
""")

  while True:
    query = input(">>> ")
    if query.lower() == 'exit':
        break
  
    if query.lower() == 'describe':
        name = input("describe >>> ")
        name = name.replace(' ', '')
        print (f"Getting description for entity '{name}':\n")
        print(pyarts3_ai.get_description(name))
    elif query.lower() == 'top-k':
        top_k = input("Enter the number of top results to return: ")
        try:
            top_k = int(top_k)
            while top_k <= 0:
                print("Please enter a positive integer for top-k.")
                top_k = input("Enter the number of top results to return: ")
                top_k = int(top_k)
        except ValueError:
            print("Invalid input. Please enter a valid integer for top-k.")
            continue
    elif query.lower() == 'help':
        print(help)
    else:
        results = pyarts3_ai.cross_search(query)
        print("Search results:")
        for res in results:
          print(f"Name: {res['name']}, Type: {res['type']}")
