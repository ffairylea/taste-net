"""
fetch_unsplash_no.py

Pulls a set of varied, randomized photos from Unsplash via their
official API, for use as the 'no' contrast set.

Usage, from inside src/:
    python fetch_unsplash_no.py YOUR_ACCESS_KEY
"""

import sys
import os
import time
import ssl
import urllib.request
import json

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

CATEGORIES = [
    "architecture", "food", "nature", "technology",
    "urban", "wildlife", "industrial", "landscape",
]

OUT_DIR = "../data/no"
IMAGES_PER_CATEGORY = 20


def fetch_random_photo(access_key, query):
    url = f"https://api.unsplash.com/photos/random?query={query}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Client-ID {access_key}"
    })
    with urllib.request.urlopen(req, context=SSL_CONTEXT) as response:
        data = json.loads(response.read())
    return data["urls"]["regular"], data["id"]


def download_image(url, path):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=SSL_CONTEXT) as response:
        with open(path, "wb") as f:
            f.write(response.read())


def count_existing(category):
    if not os.path.isdir(OUT_DIR):
        return 0
    return len([f for f in os.listdir(OUT_DIR) if f.startswith(f"{category}_")])


def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_unsplash_no.py YOUR_ACCESS_KEY")
        sys.exit(1)

    access_key = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)

    total = 0
    for category in CATEGORIES:
        have = count_existing(category)
        need = IMAGES_PER_CATEGORY - have
        if need <= 0:
            print(f"\n'{category}': already have {have}, skipping.")
            continue

        print(f"\nFetching '{category}' images ({have} already saved, need {need} more)...")
        for i in range(need):
            try:
                url, photo_id = fetch_random_photo(access_key, category)
                out_path = os.path.join(OUT_DIR, f"{category}_{photo_id}.jpg")
                download_image(url, out_path)
                total += 1
                print(f"  [{total}] saved {category}_{photo_id}.jpg")
                time.sleep(1.5)
            except Exception as e:
                print(f"  skipped one: {e}")
                time.sleep(2)

    print(f"\nDone. Downloaded {total} images to {OUT_DIR}")


if __name__ == "__main__":
    main()