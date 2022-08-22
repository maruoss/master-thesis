import numpy as np

from utils.preprocess import binary_categorize, multi_categorize

# Run check via 'python -m tests.test_preprocess' (pyhon tests/test_preprocess.py cant find modules)
if __name__ == "__main__":
    """Check label functions whethey they produce correct labels from dummy returns..."""

    dummy_ret = np.array([-0.049, -0.051, -0.0249, -0.0251, -0.01, 0.01, 0.0249, 0.0251, 0.049, 0.051, 0.1, -0.05, 0.05, 1, 2, -1, -2])

    # Binary
    solution = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0])
    # Vectorize f so that we can apply elementwise to dummy_ret.
    v_f = np.vectorize(binary_categorize)
    test = v_f(dummy_ret)
    assert (solution == test).all(), "Binary function is broken. Please fix first."
    print("The labelfn 'binary' has been tested successfully.")
    
    # Multi3
    solution = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0])
    # Vectorize f so that we can apply elementwise to dummy_ret.
    v_f = np.vectorize(multi_categorize)
    test = v_f(dummy_ret, 3)
    assert (solution == test).all(), "Multi3 function is broken. Please fix first."
    print("The labelfn 'multi3' has been tested successfully.")

    # Multi5
    solution = np.array([1, 0, 2, 1, 2, 2, 2, 3, 3, 4, 4, 1, 3, 4, 4, 0, 0])
    # Vectorize f so that we can apply elementwise to dummy_ret.
    v_f = np.vectorize(multi_categorize)
    test = v_f(dummy_ret, 5)
    assert (solution == test).all(), "Multi5 function is broken. Please fix first."
    print("The labelfn 'multi5' has been tested successfully.")