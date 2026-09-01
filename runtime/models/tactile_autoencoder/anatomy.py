from __future__ import annotations

import torch


ANATOMY_TOKEN_NAMES = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_little",
    "left_palm",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_little",
    "right_palm",
)

# Frozen canonical vocabularies inherited from the audited V6 data contract:
# hand: padding=0, left=1, right=2
# finger: padding=0, thumb=1, index=2, middle=3, ring=4, palm=5, little=6
_FINGER_TO_LOCAL_TOKEN = (0, 0, 1, 2, 3, 5, 4)


def anatomy_token_ids(
    hand_side_id: torch.Tensor,
    finger_id: torch.Tensor,
    region_mask: torch.Tensor,
) -> torch.Tensor:
    """Map every physical region to one of 12 fixed anatomy token identities.

    Invalid/padded regions return -1. The mapping depends only on audited anatomy
    metadata, never on storage slot or dataset/domain identity.
    """

    if hand_side_id.shape != finger_id.shape or hand_side_id.shape != region_mask.shape:
        raise ValueError("hand_side_id, finger_id and region_mask must have equal shapes")
    if ((hand_side_id < 0) | (hand_side_id > 2)).any():
        raise ValueError("hand_side_id must use padding/left/right ids 0/1/2")
    if ((finger_id < 0) | (finger_id > 6)).any():
        raise ValueError("finger_id must use the frozen canonical ids 0..6")

    lookup = finger_id.new_tensor(_FINGER_TO_LOCAL_TOKEN)
    local = lookup[finger_id]
    token = (hand_side_id - 1) * 6 + local
    valid_identity = region_mask & hand_side_id.ne(0) & finger_id.ne(0)
    return torch.where(valid_identity, token, token.new_full((), -1))


def anatomy_token_presence(region_token_id: torch.Tensor) -> torch.Tensor:
    token_ids = torch.arange(12, device=region_token_id.device)
    return (region_token_id[..., None] == token_ids).any(dim=1)
