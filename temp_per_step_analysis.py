#!/usr/bin/env python
import json
import numpy as np
from control_lab.ident.zoh_ident import identify_per_step, tf_string_s, tf_string_z

results = identify_per_step("input/degrausMotorTQ/controle.lvm", "input/degrausMotorTQ/saida.lvm")

print("=" * 80)
print("PER-STEP IDENTIFICATION RESULTS")
print("=" * 80)

for r in results:
    print(f"\nStep {r['step_num']} (u_level={r['u_level']:.1f}V, time={r['time_at_step']:.0f}s):")
    print(f"  K={r['gain']:.6f}, ζ={r['zeta']:.4f}, ωₙ={r['omega_n']:.6f} rad/s")
    print(f"  delay={r['delay']:.1f}s, fit RMSE={r['fit_rmse']:.4f}")
    print(f"  G(s) = {tf_string_s(np.array(r['num_s']), np.array(r['den_s']))}")
    print(f"  G(z) = {tf_string_z(np.array(r['num_z']), np.array(r['den_z']))}")

# Print summary for nonlinearity diagnosis
print(f"\n{'=' * 80}")
print("NONLINEARITY DIAGNOSIS:")
print("=" * 80)

K_vals = [r["gain"] for r in results]
zeta_vals = [r["zeta"] for r in results]
wn_vals = [r["omega_n"] for r in results]

K_var = (max(K_vals) - min(K_vals)) / np.mean(K_vals) * 100 if K_vals else 0
zeta_var = (max(zeta_vals) - min(zeta_vals)) / np.mean(zeta_vals) * 100 if zeta_vals else 0
wn_var = (max(wn_vals) - min(wn_vals)) / np.mean(wn_vals) * 100 if wn_vals else 0

print(f"Gain K: {K_var:.1f}% relative variation")
print(f"Damping ζ: {zeta_var:.1f}% relative variation")
print(f"Natural freq ωₙ: {wn_var:.1f}% relative variation")
print(f"\nInterpretation:")
if K_var > 10 or zeta_var > 20:
    print("  → NONLINEAR PLANT (parameters vary significantly across steps)")
else:
    print("  → Linear plant (parameters stable)")

# Save to JSON for notebook use
with open("input/impulse_response/zoh_per_step_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to input/impulse_response/zoh_per_step_analysis.json")
