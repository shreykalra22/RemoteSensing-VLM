import torch

from app.models.fusion import CrossModalFusionClassifier, FusionConfig


def test_fusion_forward_and_gradient_flow():
    model = CrossModalFusionClassifier(FusionConfig(embedding_dim=32, heads=4, layers=1, text_vocab_size=64))
    logits = model(torch.randn(2, 2, 32, 32), torch.randn(2, 10, 32, 32), torch.tensor([[1, 2], [3, 0]]))
    assert logits.shape == (2, 19)
    torch.nn.BCEWithLogitsLoss()(logits, torch.zeros_like(logits)).backward()
    assert model.s1_encoder.network[0].weight.grad is not None
    assert model.text_encoder.embedding.weight.grad is not None


def test_fusion_accepts_missing_text():
    model = CrossModalFusionClassifier(FusionConfig(embedding_dim=32, heads=4, layers=1))
    assert model(torch.randn(1, 2, 32, 32), torch.randn(1, 10, 32, 32)).shape == (1, 19)
