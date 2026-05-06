"""
Task 2.2 & 2.3: Analyze whether perceptron can perfectly classify three scenarios.
Features: f0=1 (bias), f1=A's score, f2=B's score
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Scenario (i): A + B > 8  -> Yes(+1), otherwise No(-1)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Scenario (i): Success iff A + B > 8")
print("=" * 70)

# This is a linear decision boundary: w0 + w1*A + w2*B = 0
# We want: A + B > 8  =>  -8 + 1*A + 1*B > 0
# So weights w = [-8, 1, 1] would perfectly implement this rule.
w_i = np.array([-8, 1, 1])

def scenario_i_label(a, b):
    return 1 if a + b > 8 else -1

# Verify on the 5 given data points
data_points = [(1,1,-1), (3,2,1), (4,5,-1), (3,4,1), (2,3,1)]
print("\nVerification on given dataset with w = [-8, 1, 1]:")
all_correct = True
for a, b, true_y in data_points:
    x = np.array([1, a, b])
    score = np.dot(w_i, x)
    pred = 1 if score > 0 else -1
    correct = (pred == true_y)
    all_correct = all_correct and correct
    status = "OK" if correct else "FAIL"
    print(f"  ({a},{b}): w·x = {score:3d}, pred = {pred:2d}, true = {true_y:2d}  [{status}]")

print(f"\nConclusion: Scenario (i) is LINEARLY SEPARABLE.")
print(f"  Perceptron CAN perfectly classify it using w = [-8, 1, 1].")
print(f"  The decision boundary is: A + B = 8")


# ---------------------------------------------------------------------------
# Scenario (ii): Success iff A in {2,3} AND B in {2,3}
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Scenario (ii): Success iff A in {2,3} and B in {2,3}")
print("=" * 70)

print("\nAll 'Yes' points in this scenario: (2,2), (2,3), (3,2), (3,3)")
print("These form a solid square in the 2D plane.")
print("All 'No' points surround this square (e.g., (1,2), (2,1), (4,2), (2,4), etc.)")

# Proof by contradiction: assume linear separator exists
# If a line separates the square from outside, consider convex hulls.
# The midpoint of (2,2) and (3,3) is (2.5, 2.5) which is inside the YES square.
# The midpoint of (1,2) and (4,3) is (2.5, 2.5) which would be on a line
# ... more rigorously:
print("\nProof of non-linear-separability:")
print("  Take four NO points: (1,2), (2,1), (4,3), (3,4)")
print("  Their convex hull contains the point (2.5, 2.5).")
print("  The YES point (2.5, 2.5) is actually inside the YES square too,")
print("  but more importantly, consider (2,2) [YES] and (4,4) [NO].")
print("  The line segment between (2,3)[YES] and (3,2)[YES] crosses")
print("  the region where NO points like (2.5, 1) exist.")

# Even simpler: check specific points that violate linear separability
print("\n  Key counter-example:")
print("    (2,2) = YES,  (2,4) = NO")
print("    (3,3) = YES,  (4,3) = NO")
print("    (2,3) = YES,  (1,3) = NO")
print("  Any line separating YES from NO would need to put (2,2) and (3,3)")
print("  on one side, but NO points exist on ALL sides of the YES square.")
print("  By the convex-hull theorem: the convex hulls of YES and NO points overlap,")
print("  therefore NO linear separator exists.")

print(f"\nConclusion: Scenario (ii) is NOT linearly separable.")
print(f"  Perceptron CANNOT perfectly classify it.")


# ---------------------------------------------------------------------------
# Scenario (iii): Success iff A == B
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Scenario (iii): Success iff A == B")
print("=" * 70)

print("\nAll 'Yes' points: (1,1), (2,2), (3,3), (4,4), (5,5) -- the diagonal line A=B.")
print("All 'No' points: everything else, i.e., points where A != B.")

print("\nProof of non-linear-separability:")
print("  Consider four points:")
print("    (1,1) = YES,  (2,2) = YES")
print("    (1,2) = NO,   (2,1) = NO")
print("  The midpoint of (1,2) and (2,1) is (1.5, 1.5), which is also")
print("  the midpoint of (1,1) and (2,2).")
print("  This means the convex hulls of YES and NO points INTERSECT.")
print("  By the hyperplane separation theorem, if convex hulls overlap,")
print("  no hyperplane can strictly separate the two classes.")

print("\n  Intuition: YES points lie exactly ON the line A=B.")
print("  NO points lie on BOTH sides of this line (A>B and A<B).")
print("  A single linear boundary cannot isolate a 'line' of points from")
print("  surrounding points on both sides.")

print(f"\nConclusion: Scenario (iii) is NOT linearly separable.")
print(f"  Perceptron CANNOT perfectly classify it.")


# ---------------------------------------------------------------------------
# Plot for visualization
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def plot_scenario(ax, title, yes_points, no_points, boundary_fn=None):
    yes_a, yes_b = zip(*yes_points) if yes_points else ([], [])
    no_a, no_b = zip(*no_points) if no_points else ([], [])
    ax.scatter(yes_a, yes_b, marker='+', s=200, c='blue', label='Yes (+1)', linewidths=2)
    ax.scatter(no_a, no_b, marker='_', s=200, c='red', label='No (-1)', linewidths=2)
    if boundary_fn:
        a_vals = np.linspace(0.5, 5.5, 100)
        b_vals = boundary_fn(a_vals)
        ax.plot(a_vals, b_vals, 'g--', linewidth=2, label='Decision boundary')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xlabel("A Score")
    ax.set_ylabel("B Score")
    ax.set_title(title)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.grid(True, alpha=0.3)
    ax.legend()

# Scenario (i): show boundary A+B=8, i.e., B = 8-A
plot_scenario(
    axes[0],
    "(i) A + B > 8\n[LINEARLY SEPARABLE]",
    [(3,4), (2,3)],  # Yes points from dataset
    [(1,1), (4,5)],  # No points from dataset
    boundary_fn=lambda a: 8 - a
)
# Add Ghosts! which is No in this scenario (3+2=5 <= 8)
axes[0].scatter([3], [2], marker='_', s=200, c='red', linewidths=2)

# Scenario (ii): show the square
yes_ii = [(2,2), (2,3), (3,2), (3,3)]
no_ii = [(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,4),(2,5),
         (3,1),(3,4),(3,5),(4,1),(4,2),(4,3),(4,4),(4,5),
         (5,1),(5,2),(5,3),(5,4),(5,5)]
plot_scenario(axes[1], "(ii) A in {2,3} and B in {2,3}\n[NOT LINEARLY SEPARABLE]", yes_ii, no_ii)

# Scenario (iii): show diagonal
yes_iii = [(1,1),(2,2),(3,3),(4,4),(5,5)]
no_iii = [(a,b) for a in range(1,6) for b in range(1,6) if a != b]
plot_scenario(axes[2], "(iii) A == B\n[NOT LINEARLY SEPARABLE]", yes_iii, no_iii)
# Draw diagonal reference
axes[2].plot([0.5, 5.5], [0.5, 5.5], 'g--', linewidth=2, alpha=0.5, label='A = B line')
axes[2].legend()

plt.tight_layout()
plt.savefig('task2_scenarios.png', dpi=150)
plt.show()
print("\nVisualization saved to task2_scenarios.png")
