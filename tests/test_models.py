import numpy as np

from fdd.models import build_autoencoder, reconstruction_error

PARAMS = {"hidden_units": 16, "latent_divisor": 3, "noise_std": 0.1, "learning_rate": 1e-3}


def test_template_dimensions():
    for n_in, latent in [(1, 1), (3, 1), (9, 3)]:
        m = build_autoencoder(n_in, PARAMS, seed=0)
        assert m.output_shape == (None, n_in)
        assert m.get_layer("latent").units == latent


def test_reconstruction_error_shape():
    m = build_autoencoder(3, PARAMS, seed=0)
    x = np.zeros((10, 3), np.float32)
    assert reconstruction_error(m, x).shape == (10,)
