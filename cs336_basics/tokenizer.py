#@title train BPE
import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import defaultdict, Counter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_any = False
            for s in split_special_tokens:
              found_at = mini_chunk.find(s)
              if found_at != -1:
                  chunk_boundaries[bi] = initial_position + found_at
                  found_any = True
                  break
            if found_any:
              break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pre_tokenize_a_chunk(chunk: str, special_tokens: list[str],) -> list[tuple]:
  tokens = []
  pattern = '(' + '|'.join(re.escape(s) for s in special_tokens) + ')'
  for c in re.split(pattern, chunk):
    if not c or (c in special_tokens):
      continue
    for x in re.finditer(PAT, c):
      tokens.append(tuple(bytes([b]) for b in x.group(0).encode('utf-8')))
  return tokens


# replace /content/assignment1-basics/tests/adapters.py:run_train_bpe with this
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
    verbose: bool = False,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    with open(input_path, "rb") as file_handle:
      boundaries = find_chunk_boundaries(
          file_handle,
          num_processes,
          [s.encode() for s in special_tokens],
      )

      # pre-tokenize
      pre_tokens = Counter()
      pair_counts = Counter()  # TODO: optimize with heap
      with multiprocessing.Pool() as pool:
        jobs = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
          file_handle.seek(start)
          chunk = file_handle.read(end - start).decode("utf-8", errors="ignore")
          # print(len(chunk.split()))
          # jobs.append(
          #     pool.apply_async(pre_tokenize_a_chunk, args=(chunk,))
          # )
          jobs.append(pre_tokenize_a_chunk(chunk, special_tokens))
          if verbose:
            break
        for job in jobs:
          for tok in job:#.get():
            pre_tokens[tok] += 1
            if len(tok) > 1:
              for a, b in zip(tok[:-1], tok[1:]):
                pair_counts[(a, b)] += 1
    if verbose:
      # print(jobs)
      print(pre_tokens)
      print(pair_counts)

    # Vocab init
    vocab = {}
    # print(special_tokens)
    for t in special_tokens:
      vocab[len(vocab)] = t.encode('utf-8')
    for t in range(256):
      vocab[len(vocab)] = bytes([t])
    if len(vocab) > vocab_size:
      raise ValueError(f"Vocab is already bigger than {vocab_size=}: {len(vocab)}!")
    
    # Merge pairs
    merges = []
    while len(vocab) < vocab_size:
      # Find max pair      
      max_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
      del pair_counts[max_pair]
      # Merge max pair in the existing words
      merges.append(
          (max_pair[0], max_pair[1])
      )
      if verbose:
        print(max_pair)
      new_tok = max_pair[0] + max_pair[1]
      vocab[len(vocab)] = new_tok
      # Update pre-tokens with merged token
      new_pre_tokens = {}
      for pre_tok, cnt in pre_tokens.items():
        new_pre_tok = []
        i = 0
        while i < len(pre_tok):
          if (i < len(pre_tok) -1) and (pre_tok[i], pre_tok[i+1]) == max_pair:
            new_pre_tok.append(new_tok)
            i += 2
            # if len(new_pre_tok) > 1:
            #   pair_counts[(new_pre_tok[-2], new_tok)] += cnt
          else:
            new_pre_tok.append(pre_tok[i])
            # if len(new_pre_tok) > 1 and (new_pre_tok[-2] == max_pair):
            #   pair_counts[(new_tok, pre_tok[i])] += cnt
            i += 1
        new_pre_tokens[tuple(new_pre_tok)] = cnt
        del new_pre_tok
      pre_tokens = new_pre_tokens
      del new_pre_tokens
      # count the pairs again
      pair_counts = Counter()
      for word, cnt in pre_tokens.items():        
        for a, b in zip(word[:-1], word[1:]):
          pair_counts[(a, b)] += cnt
    return vocab, merges

