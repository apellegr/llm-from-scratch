"""Extract clean text from a WikiMed ZIM file for training."""

import re
import os
import sys

from libzim.reader import Archive


def strip_html(html: str) -> str:
    """Remove HTML tags, scripts, styles, and normalize whitespace."""
    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    # Remove CSS comments that leak through
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    # Remove CSS/JS artifacts
    text = re.sub(r'@media[^{]*\{[^}]*\}', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def extract(zim_path: str, output_path: str, target_mb: float = 2.0):
    """Extract articles until we reach target_mb of clean text."""
    zim = Archive(zim_path)
    target_bytes = int(target_mb * 1024 * 1024)

    print(f"ZIM: {os.path.basename(zim_path)}")
    print(f"Total entries: {zim.entry_count}")
    print(f"Target: {target_mb} MB of clean text")
    print()

    articles = []
    total_chars = 0
    skipped = 0

    for i in range(zim.entry_count):
        if total_chars >= target_bytes:
            break

        entry = zim._get_entry_by_id(i)

        # Skip redirects and entries without titles
        if entry.is_redirect or not entry.title:
            skipped += 1
            continue

        try:
            item = entry.get_item()
            content = bytes(item.content).decode('utf-8', errors='replace')
        except Exception:
            skipped += 1
            continue

        # Skip non-HTML content
        if '<p>' not in content:
            skipped += 1
            continue

        text = strip_html(content)

        # Skip short articles (stubs, disambiguation pages)
        if len(text) < 500:
            skipped += 1
            continue

        # Format as: title followed by article text
        article_text = f"{entry.title}\n\n{text}\n\n"
        articles.append(article_text)
        total_chars += len(article_text)

        if len(articles) % 100 == 0:
            print(f"  Extracted {len(articles)} articles, {total_chars / 1024 / 1024:.1f} MB", end='\r')

    print(f"\n\nDone:")
    print(f"  Articles: {len(articles)}")
    print(f"  Skipped: {skipped}")
    print(f"  Total text: {total_chars / 1024 / 1024:.2f} MB")

    # Write to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(articles)

    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    zim_path = "/home/apellegr/workspace/Treehouse/data/services/kiwix/zim/wikipedia_en_medicine_nopic_2026-01.zim"
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "wikimed_train.txt")
    output_path = os.path.normpath(output_path)

    target_mb = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    extract(zim_path, output_path, target_mb)
