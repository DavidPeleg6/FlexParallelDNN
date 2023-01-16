"""
a helper file to print strategies. can be deleted
"""

import json
print('a strategy that gathers a gathered tensor')
print(json.dumps({f"fc{i}": [True, False] for i in range(1, 8)}))

print('a strategy that splits a split tensor')
print(json.dumps({f"fc{i}": [False, True] for i in range(1, 8)}))

print('a strategy that leaves everything as a vanilla data parallel')
print(json.dumps({f"fc{i}": [False, False] for i in range(1, 8)}))

print('a strategy that splits and gathers all tensors')
print(json.dumps({f"fc{i}": [True, True] for i in range(1, 8)}))

print('a strategy that gathers on the first layer and splits on the last')
new_strat = {f"fc{i}": [False, False] for i in range(1, 8)}
new_strat['fc1'] = [True, False]
new_strat['fc7'] = [False, True]
print(json.dumps(new_strat))

print()

