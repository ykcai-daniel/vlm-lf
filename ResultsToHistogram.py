import json
import torch
import matplotlib.pyplot as plt

with open('results/IMG_1752.mp4_lang.json') as f:
    data = json.load(f)

scores = []
for result in data['result']:
    if 'scores' in result:
        scores.extend(result['scores'])

plt.hist(scores, bins=20, color='b', edgecolor='black')
plt.title('Scores Distribution ('+f.name+')')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()