import torch

from app.sign_language import SignLanguageMLP


def test_sign_language_mlp_forward_shape():
    model = SignLanguageMLP(input_size=88, hidden_size=64, num_classes=26)
    model.eval()  # tryb eval aby BatchNorm nie wymagal >1 probki
    x = torch.randn(1, 88)
    out = model(x)
    assert out.shape == (1, 26)


def test_sign_language_mlp_batch_forward():
    model = SignLanguageMLP(input_size=88, hidden_size=32, num_classes=26)
    x = torch.randn(8, 88)
    out = model(x)
    assert out.shape == (8, 26)
