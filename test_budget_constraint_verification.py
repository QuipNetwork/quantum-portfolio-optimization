#!/usr/bin/env python3
"""
Verify that budget constraint is present in generated BQMs.
"""
import json
import sys

# Read the generated job
with open('test_budget_constraint/train.json', 'r') as f:
    job = json.load(f)

h = job['h']
J = job['J']

print("=" * 80)
print("BUDGET CONSTRAINT VERIFICATION")
print("=" * 80)
print()
print(f"Total linear terms (h): {len(h)}")
print(f"Total quadratic terms (J): {len(J)}")
print()

# Check if budget constraint terms are present
# Budget constraint adds:
# - Linear: -2λw for each variable
# - Quadratic: λw_i*w_j for each pair

# Look at coefficient magnitudes
h_values = sorted(h, key=lambda x: abs(x))
print("Sample h coefficients (smallest to largest magnitude):")
for i in [0, len(h_values)//4, len(h_values)//2, 3*len(h_values)//4, -1]:
    print(f"  h[{i if i >= 0 else len(h_values)+i}] = {h_values[i]:.6f}")
print()

# The budget constraint should create very strong negative linear terms
# (from -2λw term) and strong positive quadratic terms (from λw_i*w_j)
strong_negative = [x for x in h if x < -100]
strong_positive = [x for x in h if x > 100]

print(f"Strong negative h coefficients (< -100): {len(strong_negative)}")
if strong_negative:
    print(f"  Range: {min(strong_negative):.2f} to {max(strong_negative):.2f}")

print(f"Strong positive h coefficients (> 100): {len(strong_positive)}")
if strong_positive:
    print(f"  Range: {min(strong_positive):.2f} to {max(strong_positive):.2f}")
print()

# Check J coefficients (format: [[i, j, value], ...])
J_coeffs = [entry[2] for entry in J]  # Extract just the coupling values
J_values = sorted(J_coeffs, key=lambda x: abs(x))
print("Sample J coefficients (smallest to largest magnitude):")
for i in [0, len(J_values)//4, len(J_values)//2, 3*len(J_values)//4, -1]:
    print(f"  J[{i if i >= 0 else len(J_values)+i}] = {J_values[i]:.6f}")
print()

strong_J_positive = [x for x in J_coeffs if x > 10]
strong_J_negative = [x for x in J_coeffs if x < -10]

print(f"Strong positive J coefficients (> 10): {len(strong_J_positive)}")
if strong_J_positive:
    print(f"  Range: {min(strong_J_positive):.2f} to {max(strong_J_positive):.2f}")

print(f"Strong negative J coefficients (< -10): {len(strong_J_negative)}")
if strong_J_negative:
    print(f"  Range: {min(strong_J_negative):.2f} to {max(strong_J_negative):.2f}")
print()

# With budget_penalty=50.0 and weight_values up to ~0.502:
# Net linear: -2λw + λw = -λw ≈ -50 * 0.502 = -25
# Off-diagonal quadratic: 2λw_i*w_j ≈ 2 * 50 * 0.502² = 25.2
print("Expected from budget constraint (λ=50.0, max weight≈0.502):")
print("  Net linear: -λw ≈ -25")
print("  Off-diagonal quadratic: 2λw_i*w_j ≈ +25")
print()

# Look for budget constraint signatures
budget_linear_found = any(-30 < x < -20 for x in h)
budget_quad_found = any(20 < x < 30 for x in strong_J_positive) if strong_J_positive else False

if budget_linear_found:
    budget_linear_vals = [x for x in h if -30 < x < -20]
    print(f"✓ Budget constraint LINEAR terms found: {len(budget_linear_vals)} terms in range [-30, -20]")
    print(f"  Sample values: {sorted(budget_linear_vals)[:5]}")
else:
    print("✗ Budget constraint LINEAR terms NOT found (expected values around -25)")

print()

if budget_quad_found:
    budget_quad_vals = [x for x in J_coeffs if 20 < x < 30]
    print(f"✓ Budget constraint QUADRATIC terms found: {len(budget_quad_vals)} terms in range [20, 30]")
    print(f"  Sample values: {sorted(budget_quad_vals)[-5:]}")
else:
    print("✗ Budget constraint QUADRATIC terms NOT found (expected values around +25)")

print("=" * 80)
