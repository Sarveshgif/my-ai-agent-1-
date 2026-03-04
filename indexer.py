import os
import json
import glob
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def merge_small_parents(chunks, min_size):
    if not chunks: return []
    merged, current = [], None
    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                current.metadata[k] = f"{current.metadata[k]} -> {v}" if k in current.metadata else v
        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None
    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
        else:
            merged.append(current)
    return merged

def split_large_parents(chunks, max_size, splitter_overlap):
    split_chunks = []
    large_splitter = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=splitter_overlap)
    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
        else:
            split_chunks.extend(large_splitter.split_documents([chunk]))
    return split_chunks

def clean_small_chunks(chunks, min_size):
    cleaned = []
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                cleaned[-1].page_content += "\n\n" + chunk.page_content
            elif i < len(chunks) - 1:
                chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)
    return cleaned