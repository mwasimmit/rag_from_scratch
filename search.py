"""
Script to demonstrate the use of RAG models with ChromaDB.

Given a query, the script will retrieve related text from ChromaDB and use
the retrieved text as a resource to answer the query using a RAG model.

The script will also compare the answer given by a non-RAG model (mistral) with
the answer given by the RAG model (llama3.1).

Usage:

    python search.py <query>

"""
import sys, chromadb, ollama


# Set up the ChromaDB client
chromaclient = chromadb.HttpClient(host="localhost", port=8000)
collection = chromaclient.get_or_create_collection(name="buildragwithpython")


# Get the query from the command line arguments
query = " ".join(sys.argv[1:])

# Embed the query and use it to retrieve related documents from ChromaDB
queryembed = ollama.embed(model="nomic-embed-text", input=query)['embeddings']
relateddocs = '\n\n'.join(collection.query(query_embeddings=queryembed, n_results=10)['documents'][0])

# Construct the prompt for the RAG model
prompt = f"{query} - Answer that question using the following text as a resource: {relateddocs}"


# Generate answers using the non-RAG model and the RAG model
noragoutput = ollama.generate(model="phi3", prompt=query, stream=False)
print(f"Answered without RAG: {noragoutput['response']}")
print("---")
ragoutput = ollama.generate(model="phi3", prompt=prompt, stream=False)

print(f"Answered with RAG: {ragoutput['response']}")
#sol
