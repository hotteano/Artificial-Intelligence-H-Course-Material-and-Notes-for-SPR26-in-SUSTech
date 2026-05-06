"""
Task 3: Soft Margin SVM with LinearSVC (adjustable C)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# Dataset
X = np.array([[1, 1], [3, 2], [4, 5], [3, 4], [2, 3]], dtype=float)
y = np.array([-1, 1, -1, 1, 1], dtype=float)

movie_names = ['Pellet Power', 'Ghosts!', 'Pac is Bac', 'Not a Pizza', 'Endless Maze']


def plot_svm(ax, C, X, y, movie_names):
    """Train soft-margin linear SVM with LinearSVC and plot."""
    clf = LinearSVC(C=C, max_iter=10000, dual='auto')
    clf.fit(X, y)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    # Data points
    pos = X[y == 1]
    neg = X[y == -1]
    ax.scatter(pos[:, 0], pos[:, 1], marker='+', s=300, c='blue', linewidths=2.5, label='Profitable (+1)', zorder=5)
    ax.scatter(neg[:, 0], neg[:, 1], marker='_', s=300, c='red', linewidths=2.5, label='Non-profitable (-1)', zorder=5)

    # Annotate
    for name, pt in zip(movie_names, X):
        ax.annotate(name, pt, textcoords='offset points', xytext=(8, 5), fontsize=9, zorder=6)

    # Decision boundary and margins
    xx = np.linspace(0, 5.5, 100)
    if abs(w[1]) > 1e-6:
        yy_db = -(w[0] * xx + b) / w[1]
        yy_m1 = -(w[0] * xx + b - 1) / w[1]
        yy_m2 = -(w[0] * xx + b + 1) / w[1]
        ax.plot(xx, yy_db, 'k-', linewidth=2, label='Decision Boundary', zorder=3)
        ax.plot(xx, yy_m1, 'k--', linewidth=1, alpha=0.5, zorder=3)
        ax.plot(xx, yy_m2, 'k--', linewidth=1, alpha=0.5, zorder=3)
    else:
        x_db = -b / w[0] if abs(w[0]) > 1e-6 else 0
        ax.axvline(x=x_db, color='k', linewidth=2, label='Decision Boundary', zorder=3)

    # Highlight misclassified points
    preds = clf.predict(X)
    for pt, true_y, pred in zip(X, y, preds):
        if pred != true_y:
            ax.scatter([pt[0]], [pt[1]], s=600, facecolors='none', edgecolors='orange', linewidths=2.5, zorder=7)

    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Critic A's Score")
    ax.set_ylabel("Critic B's Score")

    w_norm = np.linalg.norm(w)
    margin = 2.0 / w_norm if w_norm > 1e-6 else float('inf')
    n_err = sum(preds != y)

    ax.set_title(f'C = {C}\nw=[{w[0]:.3f}, {w[1]:.3f}], b={b:.3f}\n'
                 f'Margin={margin:.3f}, Errors={n_err}/5')
    ax.set_xticks(range(6))
    ax.set_yticks(range(7))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

    return clf


# Compare multiple C values
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 10.0]

for ax, C in zip(axes, C_values):
    plot_svm(ax, C, X, y, movie_names)

plt.tight_layout()
plt.savefig('task3_svm_comparison.png', dpi=150)
plt.show()
print("Saved: task3_svm_comparison.png")

# Print detailed results for each C
print("\n" + "=" * 60)
for C in C_values:
    clf = LinearSVC(C=C, max_iter=10000, dual='auto')
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    preds = clf.predict(X)
    n_err = sum(preds != y)
    print(f"\nC = {C:6.2f}: w=[{w[0]:7.4f}, {w[1]:7.4f}], b={b:7.4f}, errors={n_err}/5")
    for name, pt, true_y, pred in zip(movie_names, X, y, preds):
        score = np.dot(w, pt) + b
        status = "OK" if pred == true_y else "WRONG"
        print(f"  {name:15s}: score={score:8.4f}, pred={pred:2.0f}, true={true_y:2.0f}  [{status}]")
