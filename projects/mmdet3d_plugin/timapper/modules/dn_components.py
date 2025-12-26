# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
import torch.nn.functional as F
import numpy as np
import torch

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def update_noise(noise,noise_segment,polygon_indexs,polyline_indexs):
    
    noise[polyline_indexs,0] = noise_segment[polyline_indexs]
    noise[polyline_indexs:,1] = noise_segment[polyline_indexs,::-1]
    
    for ii in range(noise.shape[2]):
        noise[polygon_indexs,ii] = noise_segment[polygon_indexs].roll(ii,dim=1)
    return noise

def prepare_for_dn(dn_args, 
                    noise_fuse,
                    noise_branch,
                    embed_dims,
                    tgt_weight, 
                    embedweight, 
                    batch_size, 
                    training, 
                    num_segments, 
                    num_pts_per_vec, 
                    num_classes, 
                    hidden_dim, 
                    label_enc,
                    noise_type=0,
                    pc_range=[]):
    """
    The major difference from DN-DAB-DETR is that the author process pattern embedding pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, scalar, label_noise_scale, box_noise_scale, num_patterns
    :param tgt_weight: use learnbal tgt in dab deformable detr
    :param embedweight: positional anchor queries
    :param batch_size: bs
    :param training: if it is training or inference
    :param num_segments: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """

    if training:
        max_num_point = 10
        targets, scalar, label_noise_scale, box_noise_scale = dn_args
        targets_pts, targets_label = targets
        new_targets_pts, new_targets_label = [], []
        for ii in range(len(targets_pts)):
            target_pts, target_label = targets_pts[ii], targets_label[ii]
            length_instance = len(target_label)
            if length_instance > max_num_point:
                # ped_crossing_idx = torch.nonzero(target_label==1)
                # other_class_idx = torch.nonzero(target_label!=1)
                # if len(ped_crossing_idx)<max_num_point:
                #     selected_idx = other_class_idx[torch.randperm(other_class_idx.size(0))[:max_num_point-len(ped_crossing_idx)]]
                #     selected_idx = torch.cat([ped_crossing_idx,selected_idx],dim=0)
                # else:
                #     selected_idx = ped_crossing_idx[torch.randperm(ped_crossing_idx.size(0))[:max_num_point]]
                # selected_idx = selected_idx.squeeze(1)
                selected_idx = torch.randperm(length_instance)[:max_num_point]
                selected_pts, selected_label = target_pts.fixed_num_sampled_points[selected_idx], target_label[selected_idx]
            else:
                selected_pts, selected_label = target_pts.fixed_num_sampled_points, target_label
            
            selected_label = selected_label.repeat_interleave(num_pts_per_vec,dim=0)
            new_targets_label.append(selected_label)
            new_targets_pts.append(selected_pts)
        targets_label, targets_pts = new_targets_label, new_targets_pts

    
    num_queries = num_segments*num_pts_per_vec
    indicator0 = torch.zeros([num_queries, 1]).cuda()
    # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
    tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0]*torch.tensor(0).cuda()
    refpoint_emb = embedweight
    if training:
        known = [(torch.ones_like(t)).cuda() for t in targets_label]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]
        # you can uncomment this to use fix number of dn queries
        # if int(max(known_num))>0:
        #     scalar=scalar//int(max(known_num))

        # can be modified to selectively denosie some label or boxes; also known label prediction
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t for t in targets_label])
        boxes = torch.cat([t for t in targets_pts])
        boxes = normalize_2d_pts(boxes,pc_range=pc_range)
        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(targets_label)])
        # batch_idx = batch_idx.repeat_interleave(num_pts_per_vec,dim=0)

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # add noise
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_bboxs = boxes.repeat(scalar,1,1,1) # batch,19,20,2
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone().flatten(1,2)
        known_bboxs = known_bboxs.to(tgt_weight.device)
        known_labels = known_labels.long().to(tgt_weight.device)
        # noise on the label
        if label_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
            # known_labels_expaned = known_labels_expaned.repeat_interleave(num_pts_per_vec)
        # noise on the box
        if box_noise_scale > 0:
            noise = torch.zeros_like(known_bbox_expand)
            # add noise to the segment level line pts
            # # 1 as polygon, 0 as polyline
            # segment_type = [1 if x[-1,0,0]>-100 else 0 for x in known_bbox_expand]
            # polygon_indexs = [i for i,x in enumerate(segment_type) if x]
            # polyline_indexs = [i for i,x in enumerate(segment_type) if x!=0]
            # ===== 1. random noise on all points
            if noise_type == 0:
                known_bbox_feature = gen_sineembed_for_position(known_bbox_expand)
                instance_pts = known_bbox_feature.reshape(scalar,-1,num_pts_per_vec,embed_dims)
                instance_pts = instance_pts.flatten(0,1).permute(1,0,2).cuda()
                fused_instance_pts = noise_fuse(instance_pts).permute(1,0,2)
                noise_pts = noise_branch(fused_instance_pts).reshape(scalar,-1,num_pts_per_vec,1)
                known_bbox_expand = known_bbox_expand.cuda()

                noise_pts = noise_pts.flatten(1,2).repeat(1,1,2)
                noise_pts[:,:,-1] /= 2
                random_noise =  (torch.rand_like(known_bbox_expand) * 2 - 1.0).to(known_bbox_expand.device) * noise_pts * 10
                # random_noise[:,:,1] = random_noise[:,:,1]/2
                # noise = random_noise
                known_bbox_expand += random_noise

        known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
        known_bbox_expand = known_bbox_expand.flatten(0,1)
            # ===== 2. consistent noise on each segments
            # ===== 3. learned noise for each segments to simulate realistic lines
            

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        # add dn part indicator
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand).to(m.device)
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 2).cuda()
        refpoint_emb = refpoint_emb.cuda()
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
        input_query_bbox = torch.cat([padding_bbox.repeat(batch_size, 1, 1), refpoint_emb], dim=1)

        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs, known_bbox_expand.reshape(scalar,-1,num_pts_per_vec,2)),
            'know_idx': know_idx,
            'pad_size': pad_size
        }
    else:  # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    # input_query_label = input_query_label.transpose(0, 1)
    # input_query_bbox = input_query_bbox.transpose(0, 1)

    return input_query_label, input_query_bbox, attn_mask, mask_dict


def dn_post_process(outputs_class, outputs_bbox, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    
    if mask_dict and mask_dict['pad_size'] > 0:
        nlevel, bs, num_vec, num_pts_per_vec, _ = outputs_coord.shape
        outputs_class = outputs_class.repeat_interleave(num_pts_per_vec,dim=2)
        outputs_coord = outputs_coord.flatten(2,3)
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        
        mask_dict['output_known_lbs_bboxes']=(output_known_class,output_known_coord)
        outputs_coord = outputs_coord.reshape(nlevel,bs,-1,num_pts_per_vec,2)
        new_num_vec = outputs_coord.shape[2]
        outputs_class = outputs_class.reshape(nlevel,bs,new_num_vec,num_pts_per_vec,-1)[:,:,:,0,:]
        outputs_bbox = outputs_bbox[:, : ,num_vec-new_num_vec:,:]
    else:
        nlevel, bs, num_vec, num_pts_per_vec, _ = outputs_coord.shape
    return outputs_class, outputs_bbox, outputs_coord


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    Returns:

    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs, known_bboxs_expand = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice'].long()

    known_indice = mask_dict['known_indice'].long()

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice].long()
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel()
    return known_labels, known_bboxs, known_bboxs_expand, output_known_class, output_known_coord, num_tgt


def tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt,):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
       scr: scalar, batch*pts_per_vec, 2
       tgt: scalar, batch, pts_per_vec-1 ,pts_per_vec, 2
    """
    if len(tgt_boxes) == 0:
        return {
            'tgt_loss_bbox': torch.as_tensor(0.).to('cuda'),
        }

    loss_bbox = F.l1_loss(src_boxes, tgt_boxes.reshape(-1,2), reduction='none')


    losses = {}
    losses['tgt_loss_bbox'] = loss_bbox.sum() / num_tgt
    return losses

def tgt_loss_noises(src_boxes, tgt_boxes_expand, num_tgt,):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
       scr: scalar, batch*pts_per_vec, 2
       tgt: scalar, batch, pts_per_vec-1 ,pts_per_vec, 2
    """
    if len(tgt_boxes_expand) == 0:
        return {
            'tgt_loss_bbox': torch.as_tensor(0.).to('cuda'),
        }

    loss_bbox = F.l1_loss(tgt_boxes_expand.reshape(-1,2), src_boxes.detach(), reduction='none')


    losses = {}
    losses['tgt_loss_noise'] = loss_bbox.sum() / num_tgt
    return losses

def tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
            'tgt_class_error': torch.as_tensor(0.).to('cuda'),
        }

    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0).long()

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_ce': loss_ce}

    losses['tgt_class_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
    return losses


def compute_dn_loss(mask_dict, training, aux_num, focal_alpha):
    """
       compute dn loss in criterion
       Args:
           mask_dict: a dict for dn information
           training: training or inference flag
           aux_num: aux loss number
           focal_alpha:  for focal loss
       """
    losses = {}
    if training and 'output_known_lbs_bboxes' in mask_dict:
        known_labels, known_bboxs, known_bboxs_expand, output_known_class, output_known_coord, \
        num_tgt = prepare_for_loss(mask_dict)
        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
        losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
        losses.update(tgt_loss_noises(output_known_coord[-1], known_bboxs_expand, num_tgt))
    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')

    return losses

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    import math
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos