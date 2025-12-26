from mmcv.cnn.bricks.transformer import (BaseModule)
from mmcv.runner.base_module import ModuleList
from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import warnings


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)



@TRANSFORMER_LAYER.register_module()
class CustomDetrTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 num_heads=8,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.num_heads = num_heads

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn') + operation_order.count('ins_self_attn') + operation_order.count(
            'topo_self_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'ins_self_attn', 'topo_self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def masked_attn_mask(self,tensor_shape,masked_ratio=0.2):
        bs, num_vec, num_pts_per_vec = tensor_shape
        attn_size = num_pts_per_vec
        new_attn_masks = torch.ones(attn_size,attn_size).to('cuda') < 0
        # mask some grids
        if masked_ratio > 0:
            indexs = torch.meshgrid(
                            torch.arange(0,num_pts_per_vec),
                            torch.arange(0,num_pts_per_vec))
            index_x = indexs[0].flatten().type(torch.long)
            index_y = indexs[1].flatten().type(torch.long)
            num_index = num_pts_per_vec*num_pts_per_vec
            masked_indexs = torch.randperm(num_index)[:int(masked_ratio*num_index)]
            masked_index_x, masked_index_y = index_x[masked_indexs], index_y[masked_indexs]
            diag_index = torch.nonzero(masked_index_x!=masked_index_y)
            masked_index_x, masked_index_y = masked_index_x[diag_index], masked_index_y[diag_index]
            new_attn_masks[masked_index_x,masked_index_y] = True
        return new_attn_masks

    def topo_attn_mask(self,tensor_shape):
        
        bs, num_vec, num_pts_per_vec = tensor_shape
        attn_size = num_vec
        new_attn_masks = (1-torch.eye(attn_size)).to('cuda')
        # extend mask
        new_attn_masks = new_attn_masks.repeat_interleave(num_pts_per_vec,0).repeat_interleave(num_pts_per_vec,1)
        new_attn_masks += torch.eye(num_vec*num_pts_per_vec).to('cuda')
        new_attn_masks = new_attn_masks.repeat(bs,1,1)
        return new_attn_masks>0

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                tensor_shape=None,
                training=False,
                adjacent_matrix=None,
                masked_ratio=0.2,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            # warnings.warn(f'Use same attn_mask in all attentions in '
            #               f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                
                topo_attn_masks = self.topo_attn_mask(tensor_shape)
                topo_attn_masks = [
                        copy.deepcopy(topo_attn_masks) for _ in range(self.num_attn)
                    ]
                
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ins_self_attn':
                if training:
                    masked_attn_masks = self.masked_attn_mask(tensor_shape,masked_ratio)
                    masked_attn_masks = [
                            copy.deepcopy(masked_attn_masks) for _ in range(self.num_attn)
                        ]
                else:
                    masked_attn_masks = [None for _ in range(self.num_attn)]
                bs, num_vec, num_pts_per_vec = tensor_shape
                query_reshape = query.permute(1,0,2).reshape(bs*num_vec,num_pts_per_vec,-1).permute(1,0,2)
                query_pos_reshape = query_pos.permute(1,0,2).reshape(bs*num_vec,num_pts_per_vec,-1).permute(1,0,2)
                temp_key = temp_value = query_reshape
                query = self.attentions[attn_index](
                    query_reshape,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos_reshape,
                    key_pos=query_pos_reshape,
                    attn_mask=masked_attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                query = query.permute(1,0,2).reshape(bs,num_vec*num_pts_per_vec,-1).permute(1,0,2)
                identity = query
            elif layer == 'topo_self_attn':
            
                # if training:
                #     topo_attn_masks = self.topo_attn_mask(tensor_shape,adjacent_matrix)
                #     topo_attn_masks = [
                #             copy.deepcopy(topo_attn_masks) for _ in range(self.num_attn)
                #         ]
                # else:
                #     topo_attn_masks = [None for _ in range(self.num_attn)]
                bs, num_vec, num_pts_per_vec = tensor_shape
                
                query_reshape = query.permute(1,0,2).reshape(bs,num_vec,num_pts_per_vec,-1).permute(0,2,1,3).reshape(bs*num_pts_per_vec,num_vec,-1).permute(1,0,2)
                query_pos_reshape = query_pos.permute(1,0,2).reshape(bs,num_vec,num_pts_per_vec,-1).permute(0,2,1,3).reshape(bs*num_pts_per_vec,num_vec,-1).permute(1,0,2)
                temp_key = temp_value = query_reshape
                query = self.attentions[attn_index](
                    query_reshape,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos_reshape,
                    key_pos=query_pos_reshape,
                    attn_mask=None,
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                query = query.permute(1,0,2).reshape(bs,num_pts_per_vec,num_vec,-1).permute(0,2,1,3).reshape(bs,num_vec*num_pts_per_vec,-1).permute(1,0,2)
                
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query