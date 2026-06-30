#!/usr/bin/env python3
"""Patch bench.py to pass cross_k/v/mask as predict inputs (not state writes)."""
path = "voice-engine-swift/bench.py"
with open(path) as f:
    content = f.read()

old = "state.write_state('cross_k', cross_k)\n        state.write_state('cross_v', cross_v)\n        state.write_state('cross_mask', cross_mask)"
new = "# cross_k/v/mask passed as predict inputs"

content = content.replace(old, new)
# Also add cross_k/v/mask to the predict() call
old_predict = '''out = self.decoder.predict(
                {
                    "input_ids": np.array([[tokens[-1]]], dtype=np.int32),
                    "attn_mask": attn_mask,
                    "cos": cos,
                    "sin": sin,
                    "write_onehot": onehot,
                },
                state=state,
            )'''

new_predict = '''out = self.decoder.predict(
                {
                    "input_ids": np.array([[tokens[-1]]], dtype=np.int32),
                    "attn_mask": attn_mask,
                    "cos": cos,
                    "sin": sin,
                    "write_onehot": onehot,
                    "cross_k": cross_k,
                    "cross_v": cross_v,
                    "cross_mask": cross_mask,
                },
                state=state,
            )'''

content = content.replace(old_predict, new_predict)
with open(path, 'w') as f:
    f.write(content)
print("Patched bench.py for input-based cross_k/v/mask")
