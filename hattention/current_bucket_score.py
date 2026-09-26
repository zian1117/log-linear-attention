"""Score a rank-one current-token memory without cancelling beta gradients."""
import torch


def current_bucket_score(k, v, beta, u, q, temperature, floor, norm2=None):
    """Compute the floored Frobenius-normalized score of beta*k*v^T.

    k/q and v/u have matching final dimensions; normalization of these vectors
    remains the caller's responsibility. Above the floor, beta contributes only
    its sign, so its magnitude has exactly zero score gradient. Below the floor,
    the original linear beta dependence remains. A supplied norm2 preserves the
    caller's existing branch decision and conditioning diagnostics.
    """
    key2 = k.square().sum(-1)
    value2 = v.square().sum(-1)
    if norm2 is None:
        norm2 = beta.square() * key2 * value2
    # clamp_min differentiates its input at equality; use that same branch.
    above = norm2 >= floor * floor
    # Protect the inactive branch before rsqrt. Masking an infinite derivative
    # afterwards would still allow zero * infinity to produce NaNs at zero k/v.
    base2 = torch.where(above, key2 * value2, torch.ones_like(norm2))
    gain = torch.where(above, beta.sign() * base2.rsqrt(), beta / floor)
    numerator = (q * k).sum(-1) * (u * v).sum(-1)
    return temperature * numerator * gain
