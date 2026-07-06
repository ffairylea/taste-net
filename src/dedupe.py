"""
dedupe.py

Finds and removes exact duplicate images in a folder, by comparing
file content (a hash), not just filename — catches duplicates even
if they got renamed differently on download (e.g. image.jpg vs
image(1).jpg).

Usage, from inside src/:
    python dedupe.py ../data/yes
    python dedupe.py ../data/no
"""

import sys
import os
import hashlib


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def dedupe_folder(folder):
    seen = {}
    removed = 0
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        h = file_hash(fpath)
        if h in seen:
            print(f"  removing duplicate: {fname} (same as {seen[h]})")
            os.remove(fpath)
            removed += 1
        else:
            seen[h] = fname

    print(f"\nDone. Removed {removed} duplicates. {len(seen)} unique images remain.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dedupe.py <folder>")
        sys.exit(1)
    dedupe_folder(sys.argv[1])