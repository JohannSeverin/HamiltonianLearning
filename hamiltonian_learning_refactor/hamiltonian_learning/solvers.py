# Sketch at the moment to generate the parameterization of the Lindblad Master Equation
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from typing import Union, List, Tuple, Literal
import jax.numpy as jnp
import os.path as osp
from functools import partial

import diffrax

from jax import jit
import jax
import jaxtyping

jax.config.update("jax_enable_x64", True)


# TERMS THAT GO INTO THE LINDBLAD MASTER EQUATION
# TODO: Add the ability to add time dependent terms to the master equation as well. Maybe start with Hamiltonian
@jit
def _get_unitary_term(rho, hamiltonian):
    """
    Compute the unitary term of the Master equation.
    Has to be jax numpy arrays ending with a square matrix as the last term
    """
    return -1j * (
        jnp.einsum("...ij, ...jk -> ...ik", hamiltonian, rho)
        - jnp.einsum("...ij, ...jk -> ...ik", rho, hamiltonian)
    )


@jit
def _get_dissipation_term(rho, jump_operators):
    """
    Compute the dissipator term of the Lindblad Master Equation over a list of jump operators
    """
    drho = jnp.einsum(
        "nij, ...jk, nkl -> ...il",
        jump_operators,
        rho,
        jnp.swapaxes(jump_operators.conj(), -1, -2),
    )

    drho -= 0.5 * (
        jnp.einsum(
            "nij, ...jk -> ...ik",
            jnp.swapaxes(jump_operators.conj(), -1, -2) @ jump_operators,
            rho,
        )
        + jnp.einsum(
            "...ij, njk -> ...ik",
            rho,
            jnp.swapaxes(jump_operators.conj(), -1, -2) @ jump_operators,
        )
    )

    return drho


# @jit
def _lindblad_master_equation(rho, hamiltonian, jump_operators):
    """
    The differential equation that governs the Lindblad master equation.

    Parameters:
    - rho: The density matrix representing the quantum state.
    - hamiltonian: The Hamiltonian operator.
    - jump_operators: A list of jump operators representing the dissipation terms.

    Returns:
    - drho: The time derivative of the density matrix.

    """
    drho = _get_unitary_term(rho, hamiltonian)
    drho += _get_dissipation_term(rho, jump_operators)

    return drho


# How does this look for time dependence. Maybe write firset adn then split in proper sections

from diffrax import backward_hermite_coefficients, CubicInterpolation


class Solver:
    """
    A class representing a solver for Hamiltonian learning.

    Parameters:
    - times: An array of time points.
    - initial_states: An array of initial states.
    - initial_stepsize: The initial step size for the solver. Default is 1.0.
    - max_steps: The maximum number of steps for the solver. Default is 1000.
    - ode_solver: The ODE solver to use. Can be "Dopri5" or "Dopri8". Default is "Dopri5".
    - stepsize_controller: The step size controller to use. Can be "basic" or "adaptive". Default is "basic".
    - adjoint: A boolean indicating whether to use the adjoint method. Default is False.
    - tolerance: The tolerance used for the adaptive step size controller. Default is 1e-6.

    Methods:
    - create_solver: Creates a solver function for evolving states.

    """

    def __init__(
        self,
        times: jnp.ndarray,
        # initial_states: jnp.ndarray,
        initial_stepsize: float = 1.0,
        max_steps: int = 10000,
        ode_solver: Literal["Dopri5", "Dopri8"] = "Dopri5",
        stepsize_controller: Literal["basic", "adaptive"] = "basic",
        adjoint: bool = False,
        tolerance: float = 1e-6,  # Used for adaptive stepsize controller
    ):
        # Time and State setup
        self.times = times
        self.start_time = times[0]
        self.end_time = times[-1]

        # self.initial_states = initial_states

        # Solver Setup
        self.initial_stepsize = initial_stepsize
        self.max_steps = max_steps
        self.ode_solver = getattr(diffrax, ode_solver)()
        self.stepsize_controller = (
            diffrax.PIDController(atol=tolerance, rtol=tolerance)
            if stepsize_controller == "adaptive"
            else diffrax.ConstantStepSize()
        )
        self.adjoint = (
            diffrax.BacksolveAdjoint(solver=self.ode_solver)
            if adjoint
            else diffrax.RecursiveCheckpointAdjoint()
        )
        self.saveat = (
            diffrax.SaveAt(ts=times) if times is not None else diffrax.SaveAt(t1=True)
        )

    def create_solver(self):
        """
        Creates a solver function for evolving states.

        Returns:
        - A function that takes an initial state, a Hamiltonian, and optional jump operators,
          and returns the evolved states.

        """

        # @jit
        def dynamics(t, rho, args):
            """
            Differential equation governing the system
            """

            drho = _lindblad_master_equation(rho, args[0], args[1])

            return drho

        self.dynamics = dynamics
        term = diffrax.ODETerm(dynamics)

        # @jit
        def evolve_states(initial_state, hamiltonian, jump_operators):

            return diffrax.diffeqsolve(
                terms=term,
                solver=self.ode_solver,
                y0=initial_state,
                t0=self.start_time,
                t1=self.end_time,
                stepsize_controller=self.stepsize_controller,
                dt0=self.initial_stepsize,
                args=[hamiltonian, jump_operators],
                adjoint=self.adjoint,
                max_steps=self.max_steps,
                saveat=self.saveat,
            ).ys

        return evolve_states


class TimeDependentSolver(Solver):

    def __init__(
        self,
        times: jnp.ndarray,
        # initial_states: jnp.ndarray,
        initial_stepsize: float = 1.0,
        max_steps: int = 10000,
        ode_solver: Literal["Dopri5", "Dopri8"] = "Dopri5",
        stepsize_controller: Literal["basic", "adaptive"] = "basic",
        adjoint: bool = False,
        tolerance: float = 1e-6,  # Used for adaptive stepsize controller
    ):
        # Time and State setup
        self.times = times
        self.start_time = times[0]
        self.end_time = times[-1]

        # self.initial_states = initial_states

        # Solver Setup
        self.initial_stepsize = initial_stepsize
        self.max_steps = max_steps
        self.ode_solver = getattr(diffrax, ode_solver)()
        self.stepsize_controller = (
            diffrax.PIDController(atol=tolerance, rtol=tolerance)
            if stepsize_controller == "adaptive"
            else diffrax.ConstantStepSize()
        )
        self.adjoint = (
            diffrax.BacksolveAdjoint(solver=self.ode_solver)
            if adjoint
            else diffrax.RecursiveCheckpointAdjoint()
        )
        self.saveat = (
            diffrax.SaveAt(ts=times) if times is not None else diffrax.SaveAt(t1=True)
        )

    def create_solver(self, ts):
        # Use interpolation to create the solver

        def evolve_states(initial_state, ts, Hs, jump_operators):
            coffs = backward_hermite_coefficients(ts, Hs)
            cubic_interp = CubicInterpolation(ts, coffs)

            def dynamics(t: float, rho: jaxtyping.PyTree, args) -> jaxtyping.PyTree:
                hamiltonian = cubic_interp.evaluate(t)

                drho = _lindblad_master_equation(rho, hamiltonian, jump_operators)

                return drho

            term = diffrax.ODETerm(dynamics)

            return diffrax.diffeqsolve(
                terms=term,
                solver=self.ode_solver,
                y0=initial_state,
                t0=self.start_time,
                t1=self.end_time,
                stepsize_controller=self.stepsize_controller,
                dt0=self.initial_stepsize,
                adjoint=self.adjoint,
                max_steps=self.max_steps,
                saveat=self.saveat,
            ).ys

        return evolve_states


# Tests
# if __name__ == "__main__":
#     import sys, pathlib

#     sys.path.append(str(pathlib.Path(__file__).parent.parent))
#     from parameterization import Parameterization

#     NQUBITS = 3
#     H_LOCALITY = 3
#     L_LOCALITY = 3

#     parameters = Parameterization(
#         NQUBITS, hamiltonian_locality=H_LOCALITY, lindblad_locality=L_LOCALITY
#     )

#     initial_states = jnp.stack([jnp.eye(2**NQUBITS, dtype=jnp.complex128)] * 1000)

#     solver = Solver(
#         times=jnp.linspace(0, 1000, 100),
#         initial_stepsize=1.0,
#         max_steps=1000,
#         ode_solver="Dopri5",
#         stepsize_controller="adaptive",
#         adjoint=False,
#         tolerance=1e-6,
#     )

#     hamiltonian_generator = parameters.get_hamiltonian_generator()
#     jump_operator_generator = parameters.get_jump_operator_generator()
#     evolve_states = solver.create_solver()

#     @jit
#     def loss_fn(hamiltonian_params, lindbladian_params):
#         hamiltonian = hamiltonian_generator(hamiltonian_params)
#         jump_operators = jump_operator_generator(lindbladian_params)
#         ground_last = evolve_states(initial_states, hamiltonian, jump_operators)[-1]
#         return ground_last[..., 0, 0].sum().real

#     hamiltonian_params = parameters.hamiltonian_params
#     lindbladian_params = parameters.lindbladian_params

#     grad_func = jax.grad(loss_fn, argnums=(0, 1))

#     loss_fn(hamiltonian_params, lindbladian_params)
#     grads = grad_func(hamiltonian_params, lindbladian_params)


if __name__ == "__main__":
    # Test the time dependent solver
    import sys, pathlib

    sys.path.append(str(pathlib.Path(__file__).parent.parent))

    NQUBITS = 2
    H_LOCALITY = 2
    L_LOCALITY = 1

    from parameterization import InterpolatedParameterization

    parameters = InterpolatedParameterization(
        NQUBITS,
        hamiltonian_locality=H_LOCALITY,
        lindblad_locality=L_LOCALITY,
        times=jnp.arange(0, 40, 4),
        hamiltonian_amplitudes=1.0,
    )

    params = parameters.hamiltonian_params
    generator = parameters.get_hamiltonian_generator()

    initial_states = jnp.stack(
        [jnp.zeros((2**NQUBITS, 2**NQUBITS), dtype=jnp.complex128)] * 20
    )
    initial_states = initial_states.at[:, 0, 0].set(1.0)

    ts = jnp.arange(0, 40, 4)

    solver = TimeDependentSolver(
        times=ts,
        initial_stepsize=1.0,
        max_steps=1000,
        ode_solver="Dopri5",
        stepsize_controller="adaptive",
        adjoint=False,
        tolerance=1e-6,
    )

    evolve_states = solver.create_solver(ts)

    hamiltonian = generator(params)
    jump_operators = parameters.get_jump_operator_generator()(
        parameters.lindbladian_params
    )

    coffs = backward_hermite_coefficients(ts, hamiltonian)
    cubic_interp = CubicInterpolation(ts, coffs)

    cubic_interp.evaluate(5)

    evolved_states = evolve_states(initial_states, ts, hamiltonian, jump_operators)
