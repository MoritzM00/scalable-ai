"""Test the `main` module."""

from scalai.dpnn.model import AlexNet


def test_instantiate_alexnet():
    """Test instantiation of AlexNet."""
    model = AlexNet()
    assert model is not None
    assert model.__class__.__name__ == "AlexNet"
