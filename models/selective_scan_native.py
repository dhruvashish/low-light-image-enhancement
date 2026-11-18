import torch
import torch.nn.functional as F

def selective_scan_ref(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False
):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    L = u.shape[-1]

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

    if B.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    elif B.dim() == 4:
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    else:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)

    x = torch.zeros((batch, dim, dstate), device=u.device, dtype=torch.float32)
    ys = []

    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 3:
            y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        elif C.dim() == 4:
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        else:
            y = torch.einsum('bdn,dn->bd', x, C)
        ys.append(y)

    out = torch.stack(ys, dim=-1)
    if D is not None:
        out = out + u * D[..., None]
    if z is not None:
        out = out * F.silu(z)

    out = out.to(dtype=dtype_in)
    return (out, x) if return_last_state else out

def selective_scan_fn(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False
):
    # Vectorized / grouped or looped call
    if u.dim() == 3 and B.dim() == 4 and B.shape[1] == 4: # (B, K, d_state, L)
        # K-direction batching
        B_size, Kd, L = u.shape
        K = 4
        d = Kd // K
        u_4 = u.view(B_size, K, d, L)
        delta_4 = delta.view(B_size, K, d, L)
        A_4 = A.view(K, d, -1)
        D_4 = D.view(K, d) if D is not None else [None] * K
        db_4 = delta_bias.view(K, d) if delta_bias is not None else [None] * K

        out_y = []
        for i in range(K):
            yi = selective_scan_ref(
                u_4[:, i], delta_4[:, i], A_4[i], B[:, i], C[:, i],
                D_4[i], z=None, delta_bias=db_4[i],
                delta_softplus=delta_softplus, return_last_state=False
            )
            out_y.append(yi)
        out = torch.cat(out_y, dim=1) # (B, Kd, L)
        return out
    else:
        return selective_scan_ref(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
            delta_softplus=delta_softplus, return_last_state=return_last_state
        )
