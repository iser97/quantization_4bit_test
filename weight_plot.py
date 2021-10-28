import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

with open('weight.txt', mode='r', encoding='utf-8') as r:
    file = r.readlines()
    lines = []
    for line in file:
        if line != '\n':
            lines.append(json.loads(line))

lines = [np.array(item["weight"]).flatten() for item in lines if "action" in item["name_scope"]]
res = 0
for line in lines:
    res = np.append(res, line)
res = res[1:]
c = Counter(res)
x = range(0, 16)
y = list(c.values())
y_sum = sum(y)
y_prob = [item/y_sum for item in y]
y_prob = [sum(y_prob[:index]) for index in range(1, len(y_prob)+1)]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_ylim(0, 1.1)
ax2.bar(x, y, color='b', width=0.3, label="Count")
ax1.plot(x, y_prob, color='r', linewidth=2, marker='o', markerfacecolor='white', markersize=3, label="Cumulative probability")

ax1.set_xlabel("Weight-bitwise MAC value")
ax1.set_ylabel("Cumulative probability")
ax2.set_ylabel("Count")
# plt.legend(labels=["a", "b"], loc="lower right", fontsize=6)
ax1.legend(loc="upper left", bbox_to_anchor=(0.4, 1.1), fontsize=12)
ax2.legend(loc="upper right", bbox_to_anchor=(0.4, 1.1), fontsize=12)
plt.show()
plt.savefig('figure.jpg')

        
