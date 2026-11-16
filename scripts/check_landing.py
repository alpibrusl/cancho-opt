#!/usr/bin/env python3
"""Gate of docs/design.md section 13: the pages follow cancho-www's landing pattern and are well formed. Standard library only.

    python3 scripts/check_landing.py

Fails on: a missing id of the pattern or the wrong order (stories < how < results < try < docs), no Stories link in the header, no stories region, not exactly
one <h1>, a title or description outside the SEO limits, an in-page link that names no id, a relative link that is not a file, an external request (script,
stylesheet, image or font), and a missing alt or aria-label on an svg/img."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
REQUIRED = ["stories", "how", "results", "try", "docs"]
errors = []


def err(page, msg):
    errors.append(f"{page}: {msg}")


def check(name):
    s = open(os.path.join(DOCS, name), encoding="utf-8").read()
    ids = re.findall(r'\sid="([^"]+)"', s)
    sections = re.findall(r'<section[^>]*\sid="([^"]+)"', s)
    if name == "index.html":
        for need in REQUIRED:
            if need not in sections:
                err(name, f"no section with id {need!r}")
        present = [i for i in sections if i in REQUIRED]
        if present != [i for i in REQUIRED if i in present]:
            err(name, f"required sections in the order {present}")
        if "<!--stories-->" not in s:
            err(name, "no stories region")
    header = re.search(r"<header.*?</header>", s, re.S)
    links = re.findall(r"<a[^>]*>([^<]*)</a>", header.group(0)) if header else []
    if "Stories" not in links:
        err(name, f"the header has no Stories link ({links})")
    if len(links) > 10:
        err(name, f"the header has {len(links)} links")
    if len(re.findall(r"<h1[ >]", s)) != 1:
        err(name, "not exactly one <h1>")
    t = re.search(r"<title>([^<]*)</title>", s)
    d = re.search(r'<meta name="description" content="([^"]*)"', s)
    if not t or not 10 <= len(t.group(1)) <= 70:
        err(name, f"the title is {len(t.group(1)) if t else 0} characters (10 to 70)")
    if not d or not 50 <= len(d.group(1)) <= 160:
        err(name, f"the description is {len(d.group(1)) if d else 0} characters (50 to 160)")
    for href in re.findall(r'href="([^"]+)"', s):
        if href.startswith("#"):
            if href[1:] not in ids:
                err(name, f"{href} names no id")
        elif href.startswith(("http://", "https://", "mailto:")):
            continue
        else:
            path = href.split("#")[0]
            if path and not os.path.exists(os.path.join(DOCS, path)):
                err(name, f"{href} is not a file of docs/")
            if "#" in href and path.endswith(".html"):
                target = open(os.path.join(DOCS, path), encoding="utf-8").read() if os.path.exists(os.path.join(DOCS, path)) else ""
                if f'id="{href.split("#")[1]}"' not in target:
                    err(name, f"{href} names no id")
    for tag in re.findall(r"<(?:script|img|link|iframe)[^>]*>", s):
        m = re.search(r'(?:src|href)="(https?://[^"]+)"', tag)
        if m and "canonical" not in tag and "alternate" not in tag:
            err(name, f"an external request: {m.group(1)}")
        if tag.startswith("<img") and "alt=" not in tag:
            err(name, "an <img> without alt")
    for svg in re.findall(r"<svg[^>]*>", s):
        if "aria-label" not in svg and "aria-labelledby" not in svg:
            err(name, "an <svg> with no accessible name")


for page in ("index.html", "stories.html"):
    check(page)
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("check_landing: ok")
