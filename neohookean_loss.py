import torch
import torch.nn.functional as F


def _batched_index_select(points, idx):
    """Select points with batched indices.

    points: (B, N, C)
    idx:    (B, N, K)
    return: (B, N, K, C)
    """
    bsz, n_points, channels = points.shape
    k = idx.shape[-1]
    expanded = points.unsqueeze(1).expand(-1, n_points, -1, -1)
    return torch.gather(expanded, 2, idx.unsqueeze(-1).expand(-1, -1, -1, channels))


@torch.no_grad()
def knn_indices(xyz, k=24):
    """KNN indices excluding self."""
    bsz, n_points, _ = xyz.shape
    k = min(k, max(n_points - 1, 1))
    dist = torch.cdist(xyz, xyz)
    diag = torch.arange(n_points, device=xyz.device)
    dist[:, diag, diag] = float("inf")
    return torch.topk(dist, k=k, dim=-1, largest=False).indices


def estimate_displacement_gradient(xyz, disp, k=24, reg=1e-4):
    """Estimate local displacement gradient with KNN least squares.

    Solves dU ~= dX @ G at every point, where:
      G[a, b] = d u_b / d x_a

    Returns:
      grad_u: (B, N, 3, 3)
    """
    xyz = xyz.float()
    disp = disp.float()
    idx = knn_indices(xyz, k=k)

    nbr_xyz = _batched_index_select(xyz, idx)
    nbr_disp = _batched_index_select(disp, idx)

    d_x = nbr_xyz - xyz.unsqueeze(2)      # (B, N, K, 3)
    d_u = nbr_disp - disp.unsqueeze(2)    # (B, N, K, 3)

    xt = d_x.transpose(-1, -2)            # (B, N, 3, K)
    ata = xt @ d_x                        # (B, N, 3, 3)
    atb = xt @ d_u                        # (B, N, 3, 3)

    eye = torch.eye(3, device=xyz.device, dtype=xyz.dtype).view(1, 1, 3, 3)
    ata = ata + reg * eye

    try:
        grad = torch.linalg.solve(ata, atb)
    except RuntimeError:
        grad = torch.linalg.pinv(ata) @ atb
    return grad


def lame_parameters(E, nu, eps=1e-6):
    """Convert Young's modulus and Poisson ratio to Lamé parameters."""
    E = E.float().view(-1, 1).clamp_min(eps)
    nu = nu.float().view(-1, 1).clamp(min=-0.95, max=0.495)
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def neo_hookean_energy_from_grad(
    grad_u,
    E,
    nu,
    det_eps=1e-6,
    inversion_weight=10.0,
    return_terms=False,
):
    """Compressible Neo-Hookean energy from displacement gradient.

    grad_u is G[a,b] = d u_b / d x_a, so F = I + G^T.
    Energy density:
      W = mu/2 * (tr(F^T F) - 3) - mu * log(J) + lambda/2 * log(J)^2
    """
    bsz, n_points, _, _ = grad_u.shape
    eye = torch.eye(3, device=grad_u.device, dtype=grad_u.dtype).view(1, 1, 3, 3)
    F_def = eye + grad_u.transpose(-1, -2)

    detF_raw = torch.linalg.det(F_def)
    detF = detF_raw.clamp_min(det_eps)
    logJ = torch.log(detF)
    trace_c = (F_def * F_def).sum(dim=(-1, -2))

    mu, lam = lame_parameters(E, nu)
    mu = mu.view(bsz, 1)
    lam = lam.view(bsz, 1)

    energy_density = 0.5 * mu * (trace_c - 3.0) - mu * logJ + 0.5 * lam * (logJ ** 2)
    energy = energy_density.mean()

    inversion_penalty = F.relu(det_eps - detF_raw).pow(2).mean()
    loss = energy + inversion_weight * inversion_penalty

    if return_terms:
        return {
            "loss": loss,
            "energy": energy,
            "inversion": inversion_penalty,
            "detF_mean": detF_raw.mean(),
            "detF_min": detF_raw.min(),
        }
    return loss


def neohookean_loss(
    src_xyz,
    src_disp,
    E,
    nu,
    k=24,
    reg=1e-4,
    weight_energy=1.0,
    inversion_weight=10.0,
    return_terms=False,
):
    """Neo-Hookean regularization for a point-cloud displacement field.

    Args:
        src_xyz:  (B, N, 3), source coordinates.
        src_disp: (B, N, 3), predicted displacement, usually pred - src.
        E:        (B, 1) or (B,), Young's modulus.
        nu:       (B, 1) or (B,), Poisson ratio.

    Returns:
        scalar loss by default. If return_terms=True, returns a dict.
    """
    device_type = "cuda" if src_xyz.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        grad_u = estimate_displacement_gradient(src_xyz, src_disp, k=k, reg=reg)
        terms = neo_hookean_energy_from_grad(
            grad_u,
            E,
            nu,
            inversion_weight=inversion_weight,
            return_terms=True,
        )

    terms["loss"] = weight_energy * terms["loss"]
    terms["energy"] = weight_energy * terms["energy"]
    if return_terms:
        return terms
    return terms["loss"]


# Backward-compatible alias for older imports.
neo_hookean_loss = neohookean_loss
