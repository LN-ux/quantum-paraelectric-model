# quantum-paraelectric-model

This code is part of an implementation of a constrained 4-state quantum clock model on a 2D lattice inspired by the paraelectric-to-ferroelectric transitions in titanium dioxide (TiO₂).

# hilbert_space.py
Defines a custom Hilbert space for a 4-state clock model with a constraint inspired by TiO₂. This includes:
- Clock: a lattice of discrete clock states.
- TiO2Constraint: enforces forbidden neighbor configurations inspired by the TiO₂ lattice geometry.
- TiO2LocalRule: a metropolis rule for efficient sampling of valid configurations under the constraint.


A translationally invariant residual CNN is used to model the ground state, following the architecture introduced by Zakari Denis and Giuseppe Carleo in "Accurate neural quantum states for interacting lattice bosons" (arXiv:2404.07869, 2024).
