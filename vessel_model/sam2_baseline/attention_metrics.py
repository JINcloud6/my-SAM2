import torch


def sample_query_metrics_from_attn(
    attn_bhqs: torch.Tensor,
    num_ptr: int,
    max_q: int = 512,
    query_mask: torch.Tensor | None = None,
):
    """
    attn_bhqs: [B,H,Sq,Sk] (CPU tensor ok)
    Return: entropy/top1/ptrmass samples arrays (len<=max_q)
    """
    if attn_bhqs.dim() != 4:
        raise ValueError(f"Unexpected attn shape: {attn_bhqs.shape}")
    p = attn_bhqs.mean(dim=1)[0]  # [Sq,Sk]

    eps = 1e-8
    p2 = p.clamp_min(eps)

    ent_q = -(p2 * p2.log()).sum(dim=-1)  # [Sq]
    top1_q = p.max(dim=-1).values  # [Sq]

    if num_ptr > 0:
        ptr_q = p[:, -num_ptr:].sum(dim=-1)  # [Sq]
    else:
        ptr_q = torch.zeros_like(ent_q)

    if query_mask is not None:
        query_mask = query_mask.to(ent_q.device).bool().flatten()
        if query_mask.numel() == ent_q.numel():
            keep_idx = torch.nonzero(query_mask, as_tuple=False).flatten()
            if keep_idx.numel() > 0:
                ent_q = ent_q[keep_idx]
                top1_q = top1_q[keep_idx]
                ptr_q = ptr_q[keep_idx]

    sq = ent_q.numel()
    if sq > max_q:
        idx = torch.randint(0, sq, (max_q,), device=ent_q.device)
        ent_q = ent_q[idx]
        top1_q = top1_q[idx]
        ptr_q = ptr_q[idx]

    return ent_q.detach().cpu().numpy(), top1_q.detach().cpu().numpy(), ptr_q.detach().cpu().numpy()


def mean_attn_entropy(attn_bhqs: torch.Tensor, eps: float = 1e-8) -> float:
    p = attn_bhqs.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1)  # [B,H,Sq]
    return ent.mean().item()


def mean_attn_top1(attn_bhqs: torch.Tensor) -> float:
    top1 = attn_bhqs.max(dim=-1).values  # [B,H,Sq]
    return top1.mean().item()


def mean_pointer_mass(attn_bhqs: torch.Tensor, num_ptr: int) -> float:
    if num_ptr <= 0:
        return 0.0
    sk = attn_bhqs.shape[-1]
    ptr_mass = attn_bhqs[..., sk - num_ptr: sk].sum(dim=-1)  # [B,H,Sq]
    return ptr_mass.mean().item()
