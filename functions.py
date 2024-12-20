import os
import re
import ollama

def readtextfiles(path):
    """
    Read all text files in the given directory and return
    them in a dictionary with the filename as the key
    and the content as the value.
    """
    text_contents = {}
    directory = os.path.join(path)

    try:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                text_contents[filename] = content
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
    except PermissionError:
        print(f"Permission denied to read directory '{directory}' or one of its files.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return text_contents



def chunksplitter(text, chunk_size=100):
    """
    Split a given text into chunks of a maximum size of `chunk_size` words.

    Parameters
    ----------
    text : str
        The text to be split into chunks.
    chunk_size : int, optional
        The maximum number of words in each chunk. Defaults to 100.

    Returns
    -------
    list
        A list of strings, each representing a chunk of the text.
    """
    words = re.findall(r'\S+', text)

    chunks = []
    current_chunk = []
    word_count = 0

    # Iterate over each word in the text
    for word in words:
        # Add the word to the current chunk
        current_chunk.append(word)
        # Increment the word count
        word_count += 1

        # If the current chunk has reached the maximum size, add it to the
        # list of chunks and reset the current chunk
        if word_count >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            word_count = 0

    # If there are any remaining words in the current chunk, add it to the
    # list of chunks
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def getembedding(chunks):
  
  embeds = ollama.embed(model="nomic-embed-text", input=chunks)
  return embeds.get('embeddings', [])