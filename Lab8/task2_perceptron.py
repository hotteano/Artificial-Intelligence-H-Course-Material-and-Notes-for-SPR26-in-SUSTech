"""
Task 2.1: Perceptron Model Training
Features: f0=1 (bias), f1=A's score, f2=B's score
"""

import numpy as np

# Dataset: [f0, f1, f2], label
# Label: +1 for Yes (Profitable), -1 for No (Non-profitable)
data = [
    (np.array([1, 1, 1]), -1),    # Pellet Power: No
    (np.array([1, 3, 2]), +1),    # Ghosts!: Yes
    (np.array([1, 4, 5]), -1),    # Pac is Bac: No
    (np.array([1, 3, 4]), +1),    # Not a Pizza: Yes
    (np.array([1, 2, 3]), +1),    # Endless Maze: Yes
]

movie_names = ['Pellet Power', 'Ghosts!', 'Pac is Bac', 'Not a Pizza', 'Endless Maze']

def sign(x):
    """Sign function: returns +1 if x > 0, -1 if x < 0, and -1 if x == 0 (treat 0 as misclassified)"""
    if x > 0:
        return 1
    else:
        return -1

def train_perceptron(data, eta=1.0, max_epochs=10, verbose=True):
    """Train a perceptron and return final weights and history."""
    w = np.array([0.0, 0.0, 0.0])  # Initialize weights to zero
    history = []
    
    for epoch in range(1, max_epochs + 1):
        errors = 0
        if verbose:
            print(f"\n========== Epoch {epoch} ==========")
            print(f"Initial weights: w = {w}")
        
        for i, (x, y_true) in enumerate(data):
            y_pred = sign(np.dot(w, x))
            
            if y_pred != y_true:
                # Weight update rule: w = w + eta * y_true * x
                w_new = w + eta * y_true * x
                if verbose:
                    print(f"\nSample {i+1} ({movie_names[i]}): x = {x}, y_true = {y_true}")
                    print(f"  w·x = {np.dot(w, x)}, y_pred = {y_pred} ≠ y_true = {y_true}  --> MISMATCH")
                    print(f"  Update: w_new = w + η·y·x = {w} + {eta} * {y_true} * {x} = {w_new}")
                w = w_new
                errors += 1
                history.append({
                    'epoch': epoch,
                    'sample': i + 1,
                    'movie': movie_names[i],
                    'x': x.copy(),
                    'y_true': y_true,
                    'w_before': w - eta * y_true * x,
                    'w_after': w.copy()
                })
            else:
                if verbose:
                    print(f"Sample {i+1} ({movie_names[i]}): x = {x}, y_true = {y_true}, y_pred = {y_pred}  --> CORRECT")
        
        if verbose:
            print(f"\nEpoch {epoch} summary: {errors} errors, weights = {w}")
        
        if errors == 0:
            if verbose:
                print(f"\n*** Converged at epoch {epoch}! ***")
            break
    
    return w, history


print("=" * 60)
print("Task 2.1: Perceptron Training")
print("=" * 60)
print("\nDataset (with bias feature f0=1):")
for name, (x, y) in zip(movie_names, data):
    label = "Yes (+1)" if y == 1 else "No (-1)"
    print(f"  {name:15s}: x = {x}, y = {label}")

print("\n" + "=" * 60)
print("Perceptron Update Rule:")
print("  1. Compute prediction: y_pred = sign(w · x)")
print("  2. If y_pred ≠ y_true (misclassified), update weights:")
print("       w ← w + η · y_true · x")
print("  3. If correctly classified, do nothing")
print("=" * 60)

final_w, history = train_perceptron(data, eta=1.0, max_epochs=10)

print("\n" + "=" * 60)
print("RESULT SUMMARY")
print("=" * 60)

if history:
    first_update = history[0]
    print(f"\n>>> First Update Details:")
    print(f"    Sample: {first_update['movie']} (Sample {first_update['sample']})")
    print(f"    Input x = {first_update['x']}")
    print(f"    True label y = {first_update['y_true']}")
    print(f"    Weights BEFORE update: w = {first_update['w_before']}")
    print(f"    Weights AFTER  update: w = {first_update['w_after']}")
else:
    print("\nNo updates needed (all samples correctly classified from start).")

print(f"\n>>> Final Weights after training: w = {final_w}")
print(f"    (w0 = {final_w[0]:.4f}, w1 = {final_w[1]:.4f}, w2 = {final_w[2]:.4f})")

# Verify final classification
print("\n>>> Final Classification Verification:")
for name, (x, y) in zip(movie_names, data):
    score = np.dot(final_w, x)
    pred = sign(score)
    status = "[OK] CORRECT" if pred == y else "[X] WRONG"
    print(f"    {name:15s}: w·x = {score:7.4f}, pred = {pred:2d}, true = {y:2d}  {status}")
