# Short-Peptide-Grafted-Interfaces-Regulate-Protein-Cluster-Coacervation
# Overview

The simulations investigate protein cluster coacervation regulated by short peptide grafted interfaces using a coarse-grained molecular dynamics model.
Interactions between coarse-grained beads are modeled using a soft Lennard-Jones potential, soft reversible bonding potential and harmonic bonded interactions.

All simulations were performed in a rectangular box with periodic boundary conditions in x and y and constrained condition in z in the presence of surface grafted-peptides, with a Langevin thermostat maintaining the temperature at 300 K(room temperature).


# Directory 
ALL simulations were performed Using 3-cluster configuration as the initial configuration, the mobility of grafted-peptides are performed using *velocity* and *fix setforce* command in LAMMPS MD software.
### 1.Immobile grafted-peptide system 
### 2.Mobile grafted-peptide system
### 3.Data analyse

# The correspondence between the number of grafted-peptides and surface coverage(molecules/μm²)
####  150  ------  416(molecules/μm²)
####  300  ------  833(molecules/μm²)
####  400  ------  1111(molecules/μm²)
####  600  ------  1666(molecules/μm²)

