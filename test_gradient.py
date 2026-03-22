"""Gradient flow test for encoder + DHBP (actual BP) + loss pipeline."""

import torch
from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule
from net.loss import SegmentationLoss

print("=== Gradient Flow Test (Actual BP) ===")

encoder = ContrastiveEncoder(pretrained=True)
dhbp = DHBPModule(n_classes=6)
criterion = SegmentationLoss(n_classes=6)

x = torch.randn(2, 3, 256, 256)
labels = torch.randint(0, 6, (2, 256, 256))

p1, p2, p3 = encoder.encode(x)
logits = dhbp(p1, p2, p3)
logits_up = torch.nn.functional.interpolate(
    logits, size=(256, 256), mode="bilinear", align_corners=False
)
loss, components = criterion(logits_up, labels)
print(f"Loss: {components}")

loss.backward()

checks = {
    # Encoder gradients (must flow — encoder is fine-tuned)
    "encoder.stem[0].weight": encoder.encoder.stem[0].weight.grad,
    "encoder.layer1[0].conv1.weight": encoder.encoder.layer1[0].conv1.weight.grad,
    "encoder.layer2[0].conv1.weight": encoder.encoder.layer2[0].conv1.weight.grad,
    "encoder.layer3[0].conv1.weight": encoder.encoder.layer3[0].conv1.weight.grad,
    # Unary potential heads
    "dhbp.unary_1.net[0].weight": dhbp.unary_1.net[0].weight.grad,
    "dhbp.unary_2.net[0].weight": dhbp.unary_2.net[0].weight.grad,
    "dhbp.unary_3.net[0].weight": dhbp.unary_3.net[0].weight.grad,
    # Pairwise potential heads (THE NOVEL PART)
    "dhbp.pairwise_12.net[0].weight": dhbp.pairwise_12.net[0].weight.grad,
    "dhbp.pairwise_12.net[3].weight": dhbp.pairwise_12.net[3].weight.grad,
    "dhbp.pairwise_23.net[0].weight": dhbp.pairwise_23.net[0].weight.grad,
    "dhbp.pairwise_23.net[3].weight": dhbp.pairwise_23.net[3].weight.grad,
}

all_ok = True
for name, grad in checks.items():
    has_grad = grad is not None
    norm = f"{grad.norm().item():.6f}" if has_grad else "NONE"
    status = "OK" if has_grad else "FAIL"
    print(f"  {status} {name}: grad_norm={norm}")
    if not has_grad:
        all_ok = False

zero_count = 0
for name, p in list(encoder.named_parameters()) + list(dhbp.named_parameters()):
    if p.grad is not None and p.grad.abs().max() == 0:
        zero_count += 1
        print(f"  WARN {name}: grad is all zeros")

print(f"\nZero-grad params: {zero_count}")
print("PASS" if all_ok else "FAIL")
