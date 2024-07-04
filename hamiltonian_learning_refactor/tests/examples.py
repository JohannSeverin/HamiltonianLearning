import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jax import grad, jit
import optax
import diffrax


sigmaz = jnp.diag(jnp.array([1, -1], dtype=jnp.complex128))
H = 0.1 * sigmaz


def schroedinger(t, state, args=None):
    return -1j * H @ state


from diffrax import ODETerm, Dopri5, diffeqsolve, SaveAt

term = ODETerm(schroedinger)
solver = Dopri5()

y0 = jnp.array([jnp.sqrt(2), jnp.sqrt(2)], dtype=jnp.complex128) / 2


from functools import partial


@partial(jit, static_argnames=("data",))
def func(y0, data):
    sol = diffeqsolve(
        term,
        solver,
        y0=y0,
        t0=0.0,
        t1=1.0,
        dt0=1e-3,
        saveat=SaveAt(ts=jnp.linspace(0, 1, 1000)),
    )
    return sol.ys[-1]


func(y0)
