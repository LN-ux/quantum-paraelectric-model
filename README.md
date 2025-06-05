# quantum-paraelectric-model

Under the supervision of Dr. Zakari Denis and Prof. Giuseppe Carleo.

This code implements a constrained 4-state quantum clock model on a 2D lattice inspired by the paraelectric-to-ferroelectric transitions in titanium dioxide (TiO₂).

## hilbert_space.py
Defines a custom Hilbert space for a 4-state clock model with a constraint inspired by TiO₂. This includes:
- Clock: a lattice of discrete clock states.
- TiO2Constraint: enforces forbidden neighbor configurations inspired by the TiO₂ lattice geometry.
- TiO2LocalRule: a metropolis rule for efficient sampling of valid configurations under the constraint.

## hamiltonian.py
Defines a TiO2Hamiltonian operator class for a 4-state clock model with constraints.

## ansatz.py
Defines ResNetTransInvJastrow, the neural network used to model the ground state. It follows the architecture introduced by Zakari Denis and Giuseppe Carleo in "Accurate neural quantum states for interacting lattice bosons" (arXiv:2404.07869, 2024).

## main.py
This is the main training script that runs variational Monte Carlo (VMC) optimization of the ground state. It performs the following:
- Loads simulation parameters from a JSON file.
- Constructs the constrained clock Hilbert space, defines the Hamiltonian and the local rule for sampling, and instantiates the ResNetTransInvJastrow ansatz.
- Sets up a NetKet MCState and VMC driver with SR preconditioning.
- Logs observables (energy, polarization, local order) at each step.
- Periodically saves model parameters.

## Environment
This code was developed and tested using JAX v0.5.1 and NetKet v3.16.1.post1.

To run, at minimum:
python main.py --parameters params.json --jobid your_jobid
