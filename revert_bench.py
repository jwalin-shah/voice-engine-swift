#!/usr/bin/env python3
"""Revert bench.py back to state.write_state API (CPU decoder)."""
path = "voice-engine-swift/bench.py"
with open(path) as f:
    c = f.read()

# Remove predict inputs added by previous patches
c = c.replace(
    '                    "cross_k": cross_k,\n'
    '                    "cross_v": cross_v,\n'
    '                    "cross_mask": cross_mask,\n',
    "noop_placeholder_xyz"
).replace("noop_placeholder_xyz", "")

# Check if state writes exist; if not, add them
if 'state.write_state("cross_k"' not in c:
    c = c.replace(
        "# cross_k/v/mask passed as predict inputs\n\n        attn_mask",
        "state = self.decoder.make_state()\n        state.write_state(\"cross_k\", cross_k)\n        state.write_state(\"cross_v\", cross_v)\n        state.write_state(\"cross_mask\", cross_mask)\n\n        attn_mask"
    )
    # Handle case where the comment might be slightly different
    c = c.replace(
        "# cross_k/v/mask passed as predict inputs\n        attn_mask",
        "state = self.decoder.make_state()\n        state.write_state(\"cross_k\", cross_k)\n        state.write_state(\"cross_v\", cross_v)\n        state.write_state(\"cross_mask\", cross_mask)\n        attn_mask"
    )

with open(path, "w") as f:
    f.write(c)
print("Reverted bench.py to state.write_state API")
