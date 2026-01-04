import torch

# ============================================================
# DATA LOSS: Chi-squared (experimental likelihood)
# ============================================================

def chi2_data_loss(P_pred, P_true, dP):
    """
    Chi-squared loss for experimental data.

    Parameters
    ----------
    P_pred : torch.Tensor
        PINN-predicted elastic probability
    P_true : torch.Tensor
        Experimental elastic probability
    dP : torch.Tensor
        Experimental uncertainty on P_true

    Returns
    -------
    torch.Tensor
        Mean chi-squared loss
    """
    return torch.mean(((P_pred - P_true) / dP) ** 2)


# ============================================================
# PHYSICS LOSS: Smoothness (curvature penalty)
# ============================================================

def smoothness_loss(P_pred, E_n):
    """
    Penalizes rapid curvature in P_el(E).
    Implements a second-derivative regularization.

    Parameters
    ----------
    P_pred : torch.Tensor
        PINN output
    E_n : torch.Tensor
        Normalized energy (requires_grad=True)

    Returns
    -------
    torch.Tensor
        Smoothness loss
    """
    dP_dE = torch.autograd.grad(
        P_pred,
        E_n,
        grad_outputs=torch.ones_like(P_pred),
        create_graph=True,
    )[0]

    d2P_dE2 = torch.autograd.grad(
        dP_dE,
        E_n,
        grad_outputs=torch.ones_like(dP_dE),
        create_graph=True,
    )[0]

    return torch.mean(d2P_dE2 ** 2)


# ============================================================
# TOTAL PINN LOSS
# ============================================================

def pinn_loss(
    model,
    E_n,
    P_true,
    dP,
    lambda_smooth=1e-2,
):
    """
    Total physics-informed loss:
        L = L_data + lambda_smooth * L_smooth
    """
    E_n.requires_grad_(True)
    P_pred = model(E_n)

    L_data = chi2_data_loss(P_pred, P_true, dP)
    L_smooth = smoothness_loss(P_pred, E_n)

    return L_data + lambda_smooth * L_smooth