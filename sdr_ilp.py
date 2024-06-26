from collections import defaultdict
from dataclasses import dataclass
from ortools.linear_solver import pywraplp
from time import time


@dataclass
class Solution:
    """Data class to represent the solution of the ILP."""

    objective: int
    rotations: list[int]

    # input rotation i is reconstructed via the composition
    # of the rotations in reconstructions[i]
    reconstructions: list[list[int]]

    solve_time_seconds: float


def find_optimal_rotations(
    rotations: list[int],
    N: int,
    latency_weight: float = 0.5,
    key_material_weight: float = 0.5,
) -> Solution:
    """Find an optimal subset of rotations to represent the input rotations.

    Args:
        rotations: A list of integers representing the input rotations.
        N: The dimension fo the tensor being rotated.

    Returns:
        A Solution dataclass representing the optimal subset of rotations.
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")

    base_set = range(-N + 1, N)

    # 0-1 variables if a given rotation is selected
    choose_rotation_vars = {i: solver.IntVar(0, 1, f"rot_{i}") for i in base_set}

    # 0-1 variables if a given rotation is used to represent an input rotation
    representation_vars = {
        (i, j): solver.IntVar(0, 1, f"rep_{i}_{j}")
        for i in range(len(rotations))
        for j in base_set
    }

    # build indices
    base_to_rep_vars = defaultdict(list)
    for (i, j), var in representation_vars.items():
        base_to_rep_vars[j].append(var)
    input_to_rep_vars = defaultdict(list)
    for (i, j), var in representation_vars.items():
        # values are (j, var) because we need to access j in the constraint building
        input_to_rep_vars[i].append((j, var))

    # Constraint: if a rotation is not selected, its representation vars must
    # be zero. Note: if it is selected, the rep var may still be zero.
    for i, rot_var in choose_rotation_vars.items():
        for rep_var in base_to_rep_vars[i]:
            solver.Add(rep_var <= rot_var)

    # Constraint: each input rotation must be represented by its rep_vars
    for i, rotation in enumerate(rotations):
        # rotation = sum_{j} rep_vars[j] * j
        constraint = solver.Constraint(rotation, rotation)
        for j, var in input_to_rep_vars[i]:
            constraint.SetCoefficient(var, j)

    objective = solver.Objective()
    objective.SetMinimization()
    for rot_var in choose_rotation_vars.values():
        objective.SetCoefficient(rot_var, key_material_weight)
    for rep_var in representation_vars.values():
        objective.SetCoefficient(rep_var, latency_weight)

    start = time()
    status = solver.Solve()
    end = time()
    elapsed = end - start

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        return Solution(
            objective=solver.Objective().Value(),
            rotations=[
                i for i, var in choose_rotation_vars.items() if var.solution_value()
            ],
            reconstructions=[
                [j for j, var in input_to_rep_vars[i] if var.solution_value()]
                for i in range(len(rotations))
            ],
            solve_time_seconds=elapsed,
        )
    else:
        raise ValueError(f"Unexpected solver status: {status}")


if __name__ == "__main__":
    rotations = [3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    N = 16

    solution = find_optimal_rotations(rotations, N, key_material_weight=0.5)
    print(solution)
    # Solution(
    #     objective=11.0,
    #     rotations=[3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     reconstructions=[[3], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
    #     solve_time_seconds=0.09973740577697754,
    # )

    solution = find_optimal_rotations(rotations, N, key_material_weight=10)
    print(solution)
    # Solution(
    #     objective=50.49999999999999,
    #     rotations=[3, 4, 6, 8],
    #     reconstructions=[
    #         [3],
    #         [6],
    #         [3, 4],
    #         [8],
    #         [3, 6],
    #         [4, 6],
    #         [3, 8],
    #         [4, 8],
    #         [3, 4, 6],
    #         [6, 8],
    #         [3, 4, 8],
    #     ],
    #     solve_time_seconds=20.313071727752686,
    # )
