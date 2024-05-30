import jax
import jax.numpy as jnp


def propagate_uncertainties(function, arguments, errors):
    """
    Compute the uncertainty of a function given the uncertainties of its arguments
    """
    value = function(arguments)
    jacobian = jax.jacfwd(function)(arguments)
    covariance = jnp.diag(errors**2) if errors.ndim == 1 else errors

    return value, jnp.sqrt(
        jnp.einsum("ij, jk, ki->i", jacobian, covariance, jacobian.T)
    )
