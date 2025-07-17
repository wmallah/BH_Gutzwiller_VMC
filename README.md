For each κ (outer loop) ← [Simulated Annealing / Nelder-Mead]
│
├─► Estimate n_max (based on κ)
│
├─► Tune μ (inner loop)
│     └─ Run short VMC with adaptive μ updates
│     └─ Goal: find μ such that ⟨N⟩ ≈ N_target
│     └─ If not achieved → return Inf (penalize κ)
│
├─► With (κ, μ*) fixed:
│     └─ Run production VMC
│     └─ Estimate energy ⟨H⟩
│
├─► Store (κ, μ*, ⟨N⟩, E) in trace
│
└─► Optimizer selects next κ based on energy