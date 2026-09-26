"""Score single-write memories without cancelling their scale gradients."""
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


def previous_bucket_score(k, v, beta, decay, u, q, temperature, floor, norm2):
    """Score the first token's memory after the second token's erase.

    Inputs contain two tokens in their penultimate (vector) or final (scalar)
    axis; decay contains their 2-by-2 pairwise decay factors. Return a singleton
    token axis, matching the active half of a two-token Fenwick period.
    Existing norm diagnostics remain necessary when the erase nearly cancels
    the effective key. The value read is computed separately and is unchanged.
    """
    first, second = k[..., :1, :], k[..., 1:, :]
    effective_key = first - beta[..., 1:, None] * (first * second).sum(-1, keepdim=True) * second
    scale = beta[..., :1] * decay[..., 1:, 0]
    return current_bucket_score(effective_key, v[..., :1, :], scale,
                                u[..., 1:, :], q[..., 1:, :],
                                temperature, floor, norm2)
