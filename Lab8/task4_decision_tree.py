"""
Task 4: Decision Tree Model Construction (using Entropy)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Dataset
X = np.array([[1, 1], [3, 2], [4, 5], [3, 4], [2, 3]], dtype=float)
y = np.array([0, 1, 0, 1, 1], dtype=int)  # 0=No, 1=Yes
feature_names = ["A Score", "B Score"]
class_names = ["No", "Yes"]
movie_names = ['Pellet Power', 'Ghosts!', 'Pac is Bac', 'Not a Pizza', 'Endless Maze']

# ---------- 1. Full decision tree (unrestricted depth, entropy) ----------
clf_full = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_full.fit(X, y)

fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf_full, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, ax=ax, impurity=True, proportion=True)
ax.set_title('Full Decision Tree (no depth limit, Entropy)')
plt.tight_layout()
plt.savefig('task4_tree_full.png', dpi=150)
plt.show()
print(f"Full tree depth: {clf_full.get_depth()}")
print(f"Full tree leaves: {clf_full.get_n_leaves()}")
print(f"Training accuracy: {clf_full.score(X, y):.2%}")

# ---------- 2. Trees with different max_depth ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

depths = [1, 2, 3, 4, 5, None]
for ax, depth in zip(axes, depths):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
    clf.fit(X, y)
    title = f'max_depth={depth}' if depth is not None else 'max_depth=None (full)'
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, ax=ax, impurity=True, proportion=True)
    acc = clf.score(X, y)
    n_leaves = clf.get_n_leaves()
    ax.set_title(f'{title}\nTrain Acc={acc:.1%}, Leaves={n_leaves}')

plt.tight_layout()
plt.savefig('task4_tree_depths.png', dpi=150)
plt.show()
print("Saved: task4_tree_depths.png")

# ---------- 3. Pruning: cost complexity pruning path ----------
clf_prune = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_prune.fit(X, y)
path = clf_prune.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

leaves = []
train_acc = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X, y)
    leaves.append(clf.get_n_leaves())
    train_acc.append(clf.score(X, y))

axes[0].plot(ccp_alphas, leaves, marker='o', drawstyle='steps-post')
axes[0].set_xlabel('ccp_alpha')
axes[0].set_ylabel('Number of Leaves')
axes[0].set_title('Tree Size vs Pruning Strength')
axes[0].grid(True, alpha=0.3)

axes[1].plot(ccp_alphas, impurities, marker='o', drawstyle='steps-post')
axes[1].set_xlabel('ccp_alpha')
axes[1].set_ylabel('Total Entropy')
axes[1].set_title('Entropy vs Pruning Strength')
axes[1].grid(True, alpha=0.3)

axes[2].plot(ccp_alphas, train_acc, marker='o', drawstyle='steps-post')
axes[2].set_xlabel('ccp_alpha')
axes[2].set_ylabel('Training Accuracy')
axes[2].set_title('Training Accuracy vs Pruning Strength')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task4_tree_pruning.png', dpi=150)
plt.show()
print("Saved: task4_tree_pruning.png")

# ---------- 4. Feature importance ----------
print("\nFeature importances (full tree, entropy):")
for name, imp in zip(feature_names, clf_full.feature_importances_):
    print(f"  {name}: {imp:.4f}")

# ---------- 5. Detailed predictions for full tree ----------
print("\nPredictions (full tree, entropy):")
for name, pt, true_y in zip(movie_names, X, y):
    pred = clf_full.predict([pt])[0]
    prob = clf_full.predict_proba([pt])[0]
    status = "OK" if pred == true_y else "WRONG"
    print(f"  {name:15s}: pred={pred}, prob=[{prob[0]:.2f}, {prob[1]:.2f}], true={true_y}  [{status}]")
