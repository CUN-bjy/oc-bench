import torch

def adjusted_rand_index(true_mask, pred_mask, exclude_background=True):
    """
    compute the ARI for a single image. N.b. ARI 
    is invariant to permutations of the cluster IDs.
    See https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index.
    true_mask: LongTensor of shape [N, num_entities, 1, H, W]
        background == 0
        object 1 == 1
        object 2 == 2
        ...
    pred_mask: FloatTensor of shape [N, K, 1, H, W]  (mask probs)
    Returns: ari [N]
    """
    N, _, _, H, W = true_mask.shape
    max_num_entities = 20

    # take argmax across slots for true masks
    true_mask = true_mask.squeeze(2) # [N, K, H, W]
    true_groups = true_mask.shape[1]
    true_mask = torch.argmax(true_mask, dim=1) # [N, H, W]
    true_group_ids = true_mask.view(N, H * W).long()
    true_mask_oh = torch.nn.functional.one_hot(true_group_ids, true_groups).float()
    
    # exclude background
    if exclude_background:
        true_mask_oh[...,0] = 0
        
    # take argmax across slots for predicted masks
    pred_mask = pred_mask.squeeze(2)  # [N, K, H, W]
    pred_groups = pred_mask.shape[1]
    pred_mask = torch.argmax(pred_mask, dim=1)  # [N, H, W]
    pred_group_ids = pred_mask.view(N, H * W).long()
    pred_group_oh = torch.nn.functional.one_hot(pred_group_ids, pred_groups).float()
    
    n_points = H*W
    
    if n_points <= max_num_entities and n_points <= pred_groups:
        raise ValueError(
                "adjusted_rand_index requires n_groups < n_points. We don't handle "
                "the special cases that can occur when you have one cluster "
                "per datapoint")

    n_points = torch.sum(true_mask_oh, dim=[1,2])  # [N]
    nij = torch.einsum('bji,bjk->bki', pred_group_oh, true_mask_oh)
    a = torch.sum(nij, 1)
    b = torch.sum(nij, 2)

    rindex = torch.sum(nij * (nij - 1), dim=[1,2])
    aindex = torch.sum(a * (a - 1), 1)
    bindex = torch.sum(b * (b - 1), 1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    # check if both single cluster; nij matrix has only 1 nonzero entry
    check_single_cluster = torch.sum( (nij > 0).int(), dim=[1,2])  # [N]
    check_single_cluster = (1 == check_single_cluster).int()
    ari[ari != ari] = 0  # remove Nan
    ari = check_single_cluster * torch.ones_like(ari) + (1 - check_single_cluster) * ari

    return ari



def compute_ari(model, testloader):
    device = torch.device('cuda')
    model = model.to(device)
    val_dataloader = testloader
    ari_log = []
    for step, (imgs, gt_masks) in enumerate(val_dataloader):
        gt_masks = torch.squeeze(gt_masks, dim=0)
        gt_masks = torch.argmax(gt_masks, dim=1)
        recon_combined, recons, pred_masks, slots_all = model(imgs.to(device))
        pred_masks = torch.squeeze(pred_masks, dim=0)
        pred_masks = pred_masks.permute((1,0,2,3,4))
        ari = adjusted_rand_index(gt_masks.to(device), pred_masks.to(device))
        ari_log.append(torch.mean(ari))

    return torch.mean(torch.stack(ari_log))