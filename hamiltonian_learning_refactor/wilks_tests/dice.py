import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

random_seed = 10
key = jax.random.PRNGKey(random_seed)

# Generate data according to the models
p_even = np.ones(6) / 6
p_scewed = np.array([0, 0, 0, 0, 0, 1.00])

# Generate data
from tensorflow_probability.substrates import jax as tfp

distribution = tfp.distributions.Multinomial

# Generate data
n_samples = 1000
count = 100


even_data = distribution(probs=p_even, total_count=n_samples).sample(
    sample_shape=1000, seed=key
)
likelihood_even = -distribution(probs=p_even, total_count=n_samples).log_prob(even_data)

scewed_data = distribution(probs=p_scewed, total_count=n_samples).sample(
    sample_shape=1000, seed=key
)

likelihood_scewed = -distribution(probs=p_scewed, total_count=n_samples).log_prob(
    scewed_data
)


fig, ax = plt.subplots()

ax.hist(likelihood_even, bins=30, density=True, alpha=0.5, label="Even")
ax.hist(likelihood_scewed, bins=30, density=True, alpha=0.5, label="Scewed")

ax.legend()
