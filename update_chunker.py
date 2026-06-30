#!/usr/bin/env python3
"""Update chunk_transcribe.py: sentence-level dedup + 2s overlap."""
path = 'voice-engine-swift/chunk_transcribe.py'
with open(path) as f:
    c = f.read()

old_dedup = '''def dedup_overlap(prev_text, new_text):
    """Find where prev_text ends and new_text begins; return the non-overlapping part."""
    # Split into sentences/clauses for alignment
    import re
    prev_sentences = re.split(r'(?<=[.!?])\\s+', prev_text.strip())
    new_sentences = re.split(r'(?<=[.!?])\\s+', new_text.strip())
    if not prev_sentences or not new_sentences:
        return new_text

    # Try matching last N characters of prev against start of new
    prev_clean = prev_text.strip().lower()
    new_clean = new_text.strip().lower()

    # Find longest overlapping substring at the boundary
    # Start from 40 chars and work down to 10
    for overlap_len in range(min(80, len(prev_clean)), 5, -1):
        tail = prev_clean[-overlap_len:]
        if new_clean.startswith(tail):
            return new_text[len(tail):].strip()
        # Also check if tail starts matching after some offset (partial word)
        for offset in range(1, 20):
            if len(tail) <= offset: break
            if new_clean.startswith(tail[offset:]):
                # Return everything after the partial match
                after_match = new_clean[len(tail)-offset:]
                # Find the actual text in the original
                idx = new_text.lower().find(after_match[:20])
                if idx >= 0:
                    return new_text[idx:].strip()
                return after_match

    return new_text'''

new_dedup = '''def dedup_overlap(prev_text, new_text):
    """Sentence-level dedup: drop sentences from new_text that already appeared."""
    import re
    prev_sents = re.split(r'(?<=[.!?])\\s+', prev_text.strip())
    new_sents = re.split(r'(?<=[.!?])\\s+', new_text.strip())
    if not prev_sents or not new_sents:
        return new_text

    # Normalize: lowercase + strip punctuation
    def norm(s):
        return s.strip().lower().rstrip(".,!?;")

    # Get last 1-2 sentences of previous chunk
    tail = [norm(s) for s in (prev_sents[-2:] if len(prev_sents) >= 2 else prev_sents[-1:])]

    # Drop leading new sentences that match
    for skip in range(min(len(new_sents), 4)):
        h = norm(new_sents[skip])
        for t in tail:
            if not h or not t: continue
            # Match on significant overlap: 15+ chars identical, or one contains the other
            if (len(t) > 8 and len(h) > 8 and
                (h.startswith(t) or t.startswith(h) or
                 t[:15] == h[:15] or
                 t in h or h in t)):
                return ' '.join(new_sents[skip+1:]).strip()
        # Short fragments at start are always overlap
        if len(new_sents[skip].split()) <= 4:
            continue
        break

    return new_text'''

c = c.replace(old_dedup, new_dedup)

# Update overlap defaults
c = c.replace('def transcribe_long(bench, path, overlap_s=1.0):',
              'def transcribe_long(bench, path, overlap_s=2.0):')
c = c.replace('overlap_s=1.0)', 'overlap_s=2.0)')

with open(path, 'w') as f:
    f.write(c)
print("Updated: sentence-level dedup + 2s overlap")
