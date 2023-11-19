# Tutorial
2D advection-diffusion problem
# Advection-Diffusion Solver with Deal.ii

This repository contains a C++ program using the Deal.ii library to solve a 2D advection-diffusion problem. The code utilizes finite element methods, sparse matrices, and linear solvers to simulate the evolution of the advection-diffusion equation.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The code solves an advection-diffusion problem in 2D, where the solution is influenced by advection, diffusion, and spatially varying coefficients. The simulation includes mesh generation, system setup, assembly of the linear system, solver configuration, grid refinement, and result visualization.

## Features

- Advection-diffusion problem simulation in 2D.
- Finite element method using Deal.ii library.
- Mesh generation and grid refinement.
- Output of VTK files for visualization.
- Error estimation and adaptive grid refinement.

## Dependencies

- [Deal.ii](https://www.dealii.org/): A C++ library for finite element computations.

## Installation

1. Install the Deal.ii library by following the instructions on the [Deal.ii website](https://www.dealii.org/).
2. Clone this repository: `git clone https://github.com/your-username/advection-diffusion-solver.git`
3. Compile the code using a C++ compiler with Deal.ii support.

## Usage ## Output

Run the executable after compiling the code. The program will perform a series of cycles, generating VTK files for grid and solution visualization. Check the `solver_info.txt` and `mesh_info.txt` files for solver details and mesh information.

## Output

VTK files: grid-<cycle>.vtk and solution-<cycle>.vtk for grid and solution visualization.
Text files: mesh_info.txt and solver_info.txt for mesh information and solver details.
