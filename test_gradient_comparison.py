"""
Compare gradient norms reaching the unary head:
  Path A: loss(b1_final, labels) — through full BP chain (current training)
  Path B: loss(phi_1, labels) — direct supervision (auxiliary loss)

If Path A gradients are much weaker than Path B, the BP chain
is diluting the signal and auxiliary loss would help.
"""

import torch
import torch.nn.functional as F
from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule
from net.loss import FocalLoss

print("=== Gradient Comparison: BP chain vs Direct ===\n")

encoder = ContrastiveEncoder(pretrained=True)
dhbp = DHBPModule(n_classes=6)
focal = FocalLoss(gamma=2.0, class_weights=torch.tensor([1.0, 1.5, 1.0, 1.0, 3.0, 1.2]))

x = torch.randn(2, 3, 256, 256)
labels = torch.randint(0, 6, (2, 256, 256))

# =============================================
# PATH A: Gradient through full BP chain
# (this is how training currently works)
# =============================================
encoder.zero_grad()
dhbp.zero_grad()

p1, p2, p3 = encoder.encode(x)
b1_final = dhbp(p1, p2, p3)
b1_up = F.interpolate(b1_final, size=(256, 256), mode='bilinear', align_corners=False)
loss_bp = focal(b1_up, labels)
loss_bp.backward()

print("PATH A: loss(b1_final) — through BP chain")
print(f"  Loss value: {loss_bp.item():.4f}")

grad_a = {}
grad_a['unary_1.net[0].weight'] = dhbp.unary_1.net[0].weight.grad.norm().item()
grad_a['unary_1.net[3].weight'] = dhbp.unary_1.net[3].weight.grad.norm().item()
grad_a['unary_2.net[0].weight'] = dhbp.unary_2.net[0].weight.grad.norm().item()
grad_a['unary_3.net[0].weight'] = dhbp.unary_3.net[0].weight.grad.norm().item()
grad_a['pairwise_12.alpha_net[0].weight'] = dhbp.pairwise_12.alpha_net[0].weight.grad.norm().item()
grad_a['pairwise_12.residual_net[0].weight'] = dhbp.pairwise_12.residual_net[0].weight.grad.norm().item()
grad_a['encoder.layer1[0].conv1.weight'] = encoder.encoder.layer1[0].conv1.weight.grad.norm().item()

for name, norm in grad_a.items():
    print(f"    {name}: {norm:.6f}")

# =============================================
# PATH B: Direct gradient to unary head
# (what auxiliary loss would provide)
# =============================================
encoder.zero_grad()
dhbp.zero_grad()

p1, p2, p3 = encoder.encode(x)
# Use unary output directly — no BP
phi_1 = dhbp.unary_1(p1)
phi_1 = F.log_softmax(phi_1, dim=1)
phi_1_up = F.interpolate(phi_1, size=(256, 256), mode='bilinear', align_corners=False)
loss_direct = F.cross_entropy(phi_1_up, labels)
loss_direct.backward()

print(f"\nPATH B: loss(phi_1) — direct to unary head")
print(f"  Loss value: {loss_direct.item():.4f}")

grad_b = {}
grad_b['unary_1.net[0].weight'] = dhbp.unary_1.net[0].weight.grad.norm().item()
grad_b['unary_1.net[3].weight'] = dhbp.unary_1.net[3].weight.grad.norm().item()
grad_b['unary_2.net[0].weight'] = dhbp.unary_2.net[0].weight.grad.norm().item() if dhbp.unary_2.net[0].weight.grad is not None else 0.0
grad_b['unary_3.net[0].weight'] = dhbp.unary_3.net[0].weight.grad.norm().item() if dhbp.unary_3.net[0].weight.grad is not None else 0.0
grad_b['pairwise_12.alpha_net[0].weight'] = dhbp.pairwise_12.alpha_net[0].weight.grad.norm().item() if dhbp.pairwise_12.alpha_net[0].weight.grad is not None else 0.0
grad_b['pairwise_12.residual_net[0].weight'] = dhbp.pairwise_12.residual_net[0].weight.grad.norm().item() if dhbp.pairwise_12.residual_net[0].weight.grad is not None else 0.0
grad_b['encoder.layer1[0].conv1.weight'] = encoder.encoder.layer1[0].conv1.weight.grad.norm().item()

for name, norm in grad_b.items():
    print(f"    {name}: {norm:.6f}")

# =============================================
# COMPARISON
# =============================================
print(f"\n{'='*60}")
print(f"COMPARISON: BP chain vs Direct")
print(f"{'='*60}")
print(f"{'Component':<45} {'BP chain':>10} {'Direct':>10} {'Ratio':>8}")
print(f"{'-'*45} {'-'*10} {'-'*10} {'-'*8}")

for name in grad_a:
    a = grad_a[name]
    b = grad_b.get(name, 0.0)
    ratio = a / b if b > 0 else float('inf') if a > 0 else 0.0
    flag = ""
    if b > 0 and a / b < 0.1:
        flag = " ← 10x WEAKER through BP"
    elif b > 0 and a / b < 0.5:
        flag = " ← weaker through BP"
    elif b == 0:
        flag = " (no direct grad — only through BP)"
    print(f"  {name:<43} {a:>10.6f} {b:>10.6f} {ratio:>7.2f}x{flag}")

print(f"\nINTERPRETATION:")
unary1_ratio = grad_a['unary_1.net[0].weight'] / grad_b['unary_1.net[0].weight'] if grad_b['unary_1.net[0].weight'] > 0 else 0
if unary1_ratio < 0.1:
    print(f"  Unary gradients are {1/unary1_ratio:.0f}x WEAKER through BP → auxiliary loss NEEDED")
elif unary1_ratio < 0.5:
    print(f"  Unary gradients are {1/unary1_ratio:.1f}x weaker through BP → auxiliary loss would help")
else:
    print(f"  Unary gradients are comparable ({unary1_ratio:.2f}x) → auxiliary loss NOT needed")
    print(f"  The unary collapse is caused by something else (data, initialization, etc.)")
