"""
parse_pinterest_export.py

Reads a Pinterest "Subject Access Request" data export and rebuilds
a clean image dataset from it.

WHY THIS EXISTS

The Imageye browser-extension method produced 68% near-duplicates
(see docs and check_near_duplicates.py). Pinterest virtualises its
feed, so repeatedly scrolling and re-scraping captures the same pin
many times at whatever resolution happened to be rendered. Those
copies are not byte-identical, so MD5 deduplication misses them
entirely, and they leak across the train/test split.

The official export has no such problem: each saved pin appears
exactly once, as a database record. Deduplication becomes
unnecessary rather than difficult.

WHAT THE EXPORT CONTAINS

Each pin record in pins/*.html looks like:

    Title: paha bicilmez
    Details: sanat
    Image: 0d6ca25d670dce7640b6df13f4bf07af
    Board Id: 999588148484591389
    Canonical Link: https://i.pinimg.com/736x/0d/6c/a2/0d6ca...jpg

The `Image` field is a hash, and the URL is reconstructible from it:

    https://i.pinimg.com/736x/<h[0:2]>/<h[2:4]>/<h[4:6]>/<hash>.jpg

confirmed by the one record that includes a Canonical Link. So an
image URL exists for every pin even when the link field is empty.

`Board Name` reads "No data" in the pins file, so board names have
to come from boards/*.html and be joined on Board Id.

USAGE, from anywhere:

    python3 parse_pinterest_export.py ~/Desktop/pinterest_export

    # then, once the board list looks right:
    python3 parse_pinterest_export.py ~/Desktop/pinterest_export \\
        --download --board "fleurs" --out ../data/boards/fleurs

PRIVACY NOTE: the export also contains IP addresses, search history,
sessions and message data. Keep the folder outside the repository
and delete it when finished. This script reads only pins and boards.
"""

import os
import re
import sys
import time
import html
import urllib.request

RES = "736x"          # resolution directory; originals is also valid
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def read_all(folder):
    """Concatenate every .html file in a subfolder of the export."""
    if not os.path.isdir(folder):
        return ""
    parts = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".html"):
            with open(os.path.join(folder, f), encoding="utf-8",
                      errors="ignore") as fh:
                parts.append(fh.read())
    return "\n".join(parts)


def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def parse_boards(export_dir):
    """Extract board names from boards/*.html.

    IMPORTANT LIMITATION: the boards file lists names (inside <a>
    tags) and creation dates, but NO board id. The pins file has
    `Board Id` but reports `Board Name: No data`. There is therefore
    no shared key to join the two files on.

    This returns names in file order only. Matching names to ids has
    to be done by inspection — see the sample pin links printed by
    main(), which open the pin on Pinterest where its board is
    visible."""
    raw = read_all(os.path.join(export_dir, "boards"))
    if not raw:
        return []
    names = []
    for m in re.finditer(
        r'<a href="https://www\.pinterest\.com/[^/]+/([^/"]+)/"[^>]*>\s*'
        r'([^<]+?)\s*</a>', raw):
        names.append((m.group(2).strip(), m.group(1).strip()))
    return names


def parse_pins(export_dir):
    """Return a list of dicts, one per saved pin.

    Each record in the export begins with an anchor holding the
    pin's own Pinterest URL, followed by `Key: value` lines:

        <a href="https://www.pinterest.com/pin/9995.../">...</a>
        Title: ...
        Image: 0d6ca25d670dce7640b6df13f4bf07af
        Board Id: 999588148484591389

    So the file is split on those anchors, and each chunk parsed for
    its fields. The pin URL is kept because it is the only practical
    way to find out which board an id refers to — the export never
    states it."""
    raw = read_all(os.path.join(export_dir, "pins"))

    # Split on the pin permalink anchors, keeping the URL itself.
    chunks = re.split(r'<a href="(https://www\.pinterest\.com/pin/\d+/)"',
                      raw)

    pins = []
    # chunks alternates: [preamble, url1, body1, url2, body2, ...]
    for i in range(1, len(chunks) - 1, 2):
        url, body = chunks[i], chunks[i + 1]
        text = re.sub(r"<br\s*/?>", "\n", body)
        text = re.sub(r"<[^>]+>", "\n", text)
        text = html.unescape(text)

        rec = {"pin url": url}
        for line in text.split("\n"):
            m = re.match(r"^\s*([A-Za-z ]+):\s*(.*)$", line)
            if not m:
                continue
            k, v = m.group(1).strip().lower(), m.group(2).strip()
            if v and v.lower() != "no data" and k not in rec:
                rec[k] = v
        if "image" in rec:
            pins.append(rec)
    return pins


def image_url(image_hash, res=RES):
    """Rebuild the CDN URL from the image hash.

    Pattern confirmed against the one record in the export that
    carried a full Canonical Link: the first three byte-pairs of the
    hash become nested directory names."""
    h = image_hash
    return f"https://i.pinimg.com/{res}/{h[0:2]}/{h[2:4]}/{h[4:6]}/{h}.jpg"


def download(pins, out_dir, delay=0.3):
    os.makedirs(out_dir, exist_ok=True)
    ok = fail = skip = 0
    for i, p in enumerate(pins, 1):
        h = p["image"]
        dest = os.path.join(out_dir, f"{h}.jpg")
        if os.path.exists(dest):
            skip += 1
            continue
        try:
            req = urllib.request.Request(image_url(h),
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            if len(data) < 1000:
                fail += 1
                continue
            with open(dest, "wb") as f:
                f.write(data)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"  failed {h[:12]}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(pins)}  ok {ok}  failed {fail}  skip {skip}")
        time.sleep(delay)   # be polite to the CDN
    print(f"\nDone. {ok} downloaded, {fail} failed, {skip} already present.")
    print(f"Saved to {out_dir}")
    print("Filenames are image hashes, so re-running never creates "
          "duplicates.")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    export_dir = os.path.expanduser(sys.argv[1])

    boards = parse_boards(export_dir)
    pins = parse_pins(export_dir)
    print(f"Parsed {len(pins)} pins from the export\n")

    by_board = {}
    for p in pins:
        by_board.setdefault(p.get("board id", "unknown"), []).append(p)

    if boards:
        print(f"Board names found in the export ({len(boards)}):")
        print("  " + ", ".join(n for n, _ in boards))
        print("""
These names cannot be matched to ids automatically: the boards file
has names without ids, and the pins file has ids without names. Use
the sample links below to identify each board by eye.
""")

    print(f"{'board id':>22s} {'pins':>6s}   open this pin to see its board")
    print("-" * 78)
    for bid, ps in sorted(by_board.items(), key=lambda kv: -len(kv[1])):
        sample = ps[0].get("pin url", "")
        print(f"{bid:>22s} {len(ps):>6d}   {sample}")

    print("""
Open a sample link for each board you want. Pinterest shows which
board the pin is saved to, which gives you the id -> name mapping.
Boards with only a handful of pins can be ignored.
""")

    if "--download" not in sys.argv:
        print("""
Nothing downloaded. To fetch one board's images:

    python3 parse_pinterest_export.py <export_dir> --download \\
        --board-id <ID> --out ../data/boards/<name> --limit 233

--limit caps how many are fetched, sampled at random with a fixed
seed. Use the same value for every board to keep classes balanced —
balanced classes make the confusion matrix interpretable and set the
chance baseline to exactly 1/n_boards.

Every pin appears once in the export, and files are named by image
hash, so the result contains no duplicates by construction.""")
        return

    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    if not out:
        raise SystemExit("--download requires --out <folder>")

    selected = None
    if "--board-id" in sys.argv:
        bid = sys.argv[sys.argv.index("--board-id") + 1]
        selected = by_board.get(bid)
        if not selected:
            raise SystemExit(f"No pins with board id {bid}")
    elif "--all-boards" in sys.argv:
        # Every pin the user has ever saved, pooled across all
        # boards. For the Phase 1 "would I save this at all?"
        # question this is the correct positive class: sampling from
        # a single board makes the model a detector for that board's
        # aesthetic rather than for the user's taste in general. The
        # original data/yes came only from the pink board, which is
        # why Phase 1 was arguably measuring "is this pink".
        selected = list(pins)
        print(f"Pooling all {len(selected)} pins across "
              f"{len(by_board)} boards")
    else:
        raise SystemExit("--download requires --board-id or --all-boards "
                         "(names cannot be joined to ids; see above)")

    # --limit takes a random sample rather than the first N.
    # Pins appear in save order, so first-N would bias toward
    # whichever period the board was most active — a random sample
    # is representative of the board as a whole. Seeded so the same
    # subset is reproducible across runs and across boards.
    if "--limit" in sys.argv:
        n = int(sys.argv[sys.argv.index("--limit") + 1])
        if n < len(selected):
            import random
            random.Random(42).shuffle(selected)
            selected = selected[:n]
            print(f"Sampled {n} of the board's pins (seed=42)")

    print(f"\nDownloading {len(selected)} images to {out}")
    download(selected, out)


if __name__ == "__main__":
    main()
