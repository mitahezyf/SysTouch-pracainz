import torch

from app.sign_language import SignLanguageMLP


def test_sign_language_mlp_forward_shape():
    model = SignLanguageMLP(input_size=63, hidden_size=64, num_classes=26)
    x = torch.randn(1, 63)
    out = model(x)
    assert out.shape == (1, 26)


def test_sign_language_mlp_batch_forward():
    model = SignLanguageMLP(input_size=63, hidden_size=32, num_classes=26)
    x = torch.randn(8, 63)
    out = model(x)
    assert out.shape == (8, 26)
