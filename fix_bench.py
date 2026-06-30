#!/usr/bin/env python3
"""Fix bench.py: pass cross_k/v/mask as predict inputs, not state writes."""
path = "voice-engine-swift/bench.py"
with open(path) as f:
    lines = f.readlines()

# Find and remove state.write_state lines for cross_k/v/mask
new_lines = []
skip_next = 0
for i, line in enumerate(lines):
    if skip_next > 0:
        skip_next -= 1
        continue
    if 'state.write_state("cross_k", cross_k)' in line:
        continue  # will be passed as inputs
    if 'state.write_state("cross_v", cross_v)' in line:
        continue
    if 'state.write_state("cross_mask", cross_mask)' in line:
        continue
    new_lines.append(line)

# Check if cross_k/v/mask are already in predict inputs
has_cross_inputs = any('"cross_k": cross_k' in l for l in new_lines)
if not has_cross_inputs:
    # Add them after the "write_onehot": onehot, line
    for i, line in enumerate(new_lines):
        if '"write_onehot": onehot,' in line:
            new_lines.insert(i + 1, '                    "cross_k": cross_k,\n')
            new_lines.insert(i + 2, '                    "cross_v": cross_v,\n')
            new_lines.insert(i + 3, '                    "cross_mask": cross_mask,\n')
            break

with open(path, 'w') as f:
    f.writelines(new_lines)
print("Fixed bench.py: cross_k/v/mask are now predict inputs")
