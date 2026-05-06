import matplotlib.pyplot as plt
import numpy as np

# Data
movies = ['Pellet Power', 'Ghosts!', 'Pac is Bac', 'Not a Pizza', 'Endless Maze']
a_scores = [1, 3, 4, 3, 2]
b_scores = [1, 2, 5, 4, 3]
profits = [False, True, False, True, True]

# Separate by profit
pos_a = [a for a, p in zip(a_scores, profits) if p]
pos_b = [b for b, p in zip(b_scores, profits) if p]
neg_a = [a for a, p in zip(a_scores, profits) if not p]
neg_b = [b for b, p in zip(b_scores, profits) if not p]

plt.figure(figsize=(8, 6))
plt.scatter(pos_a, pos_b, marker='+', s=300, c='blue', label='Profitable (+)', linewidths=2.5)
plt.scatter(neg_a, neg_b, marker='_', s=300, c='red', label='Non-profitable (-)', linewidths=2.5)

# Annotate points
for name, a, b in zip(movies, a_scores, b_scores):
    plt.annotate(name, (a, b), textcoords='offset points', xytext=(8, 5), fontsize=10)

plt.xlabel("Critic A's Score")
plt.ylabel("Critic B's Score")
plt.title('Task 1: Movie Profit Prediction - Data Visualization')
plt.xlim(0, 5.5)
plt.ylim(0, 6)
plt.xticks(range(6))
plt.yticks(range(7))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('task1_plot.png', dpi=150)
plt.show()
print('Plot saved as task1_plot.png')
