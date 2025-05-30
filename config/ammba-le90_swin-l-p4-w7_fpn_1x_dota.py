_base_ = './ammba-le90_r50_fpn_1x_dota.py'
pretrained = '/checkpoint/swin_large_patch4_window7_224_22k.pth'
# download pretrained weights:https://github.com/microsoft/Swin-Transformer

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer', # install mmdetection
        embed_dims=192,
        depths=depths,
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4, 
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=[
        dict(
            type='mmdet.FPN',
            in_channels=[192, 384, 768, 1536],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        dict(
            type='mmrotate.BFPA',
            in_channels=256,
            out_channels=256,
            num_levels=5,
            refine_level=2, 
            refine_type='non_local')],
)

# train_dataloader = dict(batch_size=1, num_workers=1)

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
