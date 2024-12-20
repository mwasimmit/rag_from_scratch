
"""Import text documents from a folder and add them to a ChromaDB collection
with embeddings and metadata."""

import chromadb

from functions import readtextfiles, chunksplitter, getembedding


def main():
    """Entry point of the script."""
    chromaclient = chromadb.HttpClient(host="localhost", port=8000)
    textdocspath = "./text"
    text_data = readtextfiles(textdocspath)

    if "buildragwithpython" in [collection.name for collection in chromaclient.list_collections()]:
        chromaclient.delete_collection("buildragwithpython")

    collection = chromaclient.create_collection(name="buildragwithpython", metadata={"hnsw:space": "cosine"})
    
    for filename, text in text_data.items():
        chunks = chunksplitter(text)
        embeds = getembedding(chunks)

        chunknumber = list(range(len(chunks)))
        ids = [filename + str(index) for index in chunknumber]
        metadatas = [{"source": filename} for index in chunknumber]

        collection.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)


if __name__ == "__main__":
    main()
