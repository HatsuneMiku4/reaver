import gin
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.layers import (
    Input, Concatenate, Dense, Embedding,
    Conv2D, Flatten, Lambda,
    ConvLSTM2D, Conv2DTranspose,
)
from tensorflow.python.keras.layers.merge import add

from reaver.models.base.layers import (
    Squeeze, Split, Transpose, Log, Broadcast2D
)


class MHDPA:
    pass


@gin.configurable
def build_relational(obs_spec, act_spec, data_format='channels_first', broadcast_non_spatial=False, fc_dim=256):
    # https://github.com/deepmind/pysc2/blob/master/docs/environment.md#last-actions
    # obs_spec: screen, minimap, player (11,), last_actions (n,)
    # At each time step agents are presented with 4 sources of information:
    # minimap, screen, player, and previous-action.

    # TODO: set spatial_dim <- 64
    screen, screen_input = spatial_block('screen', obs_spec.spaces[0], conv_cfg(data_format, 'relu'))
    minimap, minimap_input = spatial_block('minimap', obs_spec.spaces[1], conv_cfg(data_format, 'relu'))

    # TODO: obs_spec[2:] <- ['available_actions', 'player', 'previous_action']
    # 'non-spatial': ['available_actions', 'player']
    available_actions = Input(obs_spec.spaces[2].shape)
    non_spatial_inputs_list = [Input(s.shape) for s in obs_spec.spaces[3:]]
    non_spatial_inputs = Concatenate(axis=1, name='non_spatial_inputs')(non_spatial_inputs_list)
    input_2d = _mlp2(Flatten()(non_spatial_inputs), units=[128, 64], cfg=dense_cfg('relu'))  # [64, ]

    # broadcast_non_spatial = False
    if broadcast_non_spatial:
        non_spatial, spatial_dim = non_spatial_inputs[1], obs_spec.spaces[0].shape[1]
        non_spatial = Log()(non_spatial)
        broadcasted_non_spatial = Broadcast2D(spatial_dim)(non_spatial)
        input_3d = Concatenate(axis=1, name="state_block")([screen, minimap, broadcasted_non_spatial])
    else:
        # concat along channel dim: [N, C, H, W]
        # input_3d: [8, 8, channel_1+channel_1]
        input_3d = Concatenate(axis=1, name="state_block")([screen, minimap])

    # TODO: state -> Conv2DLSTM -> output_3d: [8, 8, channel_2]
    # TODO: treat channel_x as parameters or read from configuration files
    # channel_2 = 96
    channel_2 = 96
    output_3d = ConvLSTM2D(filters=channel_2, kernel_size=3, **conv2dlstm_cfg())(input_3d)
    relational_spatial = _resnet12(output_3d, filters=[64, 48, 32, 32], cfg=conv_cfg(data_format, 'relu'))  # [8, 8, 32]
    channel_3 = 16
    relational_spatial = _deconv4x(relational_spatial, filters=[channel_3, channel_3],
                                   kernel_sizes=[4, 4], cfg=deconv_cfg(data_format))  # [32, 32, channel_3]

    # relational_spatial = Concatenate()(Broadcast2D())  # [32, 32, channel_3+embed_size]
    # TODO: tile action embedding, 1x1x1 conv
    # embed_sz = 16
    # [32, 32, channel_3+embed_sz]

    # TODO: check scale factor
    relational_nonspatial = _mlp2(Flatten()(output_3d), units=[512, 512], cfg=dense_cfg('relu'))  # [512, ]

    shared_features = Concatenate(axis=1, name='shared_features')([relational_nonspatial, input_2d])  # [512+64, ]
    value = _mlp2(shared_features, units=[256, 1], cfg=dense_cfg('relu', scale=0.1))  # TODO: check scale_factor
    policy_logits = _mlp2(shared_features, units=[256, len(act_spec)], cfg=dense_cfg('relu', scale=0.1))

    # fc = Flatten(name="state_flat")(state)
    # fc = Dense(fc_dim, **dense_cfg('relu'))(fc)
    #
    # value = Dense(1, name="value_out", **dense_cfg(scale=0.1))(fc)
    # value = Squeeze(axis=-1)(value)

    # TODO: calculate policy logits
    # logits = []
    # sensible action set for all minigames
    # action_ids = [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]

    # env.obs_spec()
    # Out[5]:
    # Spec: Observation
    # Space(screen
    # {player_relative, selected, visibility_map, unit_hit_points_ratio, unit_density}, (5, 16, 16), numpy.int32)
    # Space(minimap
    # {player_relative, selected, visibility_map, camera}, (4, 16, 16), numpy.int32)
    # Space(available_actions, (23,), numpy.int32)
    # Space(player, (11,), numpy.int32)

    # env.act_spec()
    # Out[6]:
    # Spec: Action
    # Space(function_id, (), cat: 23, numpy.int32)
    # Space(screen, (), cat: (16, 16), numpy.int32)
    # Space(minimap, (), cat: (16, 16), numpy.int32)
    # Space(screen2, (), cat: (16, 16), numpy.int32)
    # Space(queued, (), cat: 2, numpy.int32)
    # Space(control_group_act, (), cat: 5, numpy.int32)
    # Space(control_group_id, (), cat: 10, numpy.int32)
    # Space(select_add, (), cat: 2, numpy.int32)
    # Space(select_point_act, (), cat: 4, numpy.int32)
    # Space(select_unit_act, (), cat: 4, numpy.int32)
    # Space(select_worker, (), cat: 4, numpy.int32)
    # Space(build_queue_id, (), cat: 10, numpy.int32)

    # TODO: check 1x1x1 conv & scale_factor
    # conv_layer = Conv2D(filters=1, kernel_size=1, **conv_cfg(data_format, scale=0.1))
    # spatial_logits_part = Flatten()(conv_layer(relational_spatial))
    # for space in act_spec:
    #     if space.is_spatial():
    #         logits.append(spatial_logits_part)
    #     else:
    #         logits.append(Dense(space.size(), **dense_cfg(scale=0.1))(fc))

    # non_spatial_inputs[0]: available_actions
    mask_actions = Lambda(
        lambda x: tf.where(available_actions > 0, x, -1000 * tf.ones_like(x)),
        name="mask_unavailable_action_ids"
    )
    # function_id, shape = (23, ) (#action_ids)
    # logits[0] = mask_actions(logits[0])
    policy_logits = mask_actions(policy_logits)

    # TODO: action embedding, matmul -> transformed_logits

    return Model(
        inputs=[screen_input, minimap_input] + [available_actions] + non_spatial_inputs_list,
        outputs=[policy_logits, relational_spatial, value]
    )


def spatial_block(name, space, cfg):
    # TODO: tile spatial features with binary masks
    inpt = Input(space.shape, name=name + '_input')  # space.shape: [C, H, W], inpt.shape: [N, C, H, W]
    block = Split(space.shape[0], axis=1)(inpt)  # tf.split(x, num_splits, axis=axis) -> List

    # Input Preprocessing
    for i, (name, dim) in enumerate(zip(space.spatial_feats, space.spatial_dims)):
        if dim > 1:
            # categorical features are embedded into a continuous 10-dimensional space.
            block[i] = Squeeze(axis=1)(block[i])
            block[i] = Embedding(input_dim=dim, output_dim=10)(block[i])
            block[i] = Transpose([0, 3, 1, 2])(block[i])  # [N, H, W, C] -> [N, C, H, W]
        else:
            # numerical features are re-scaled with a logarithmic transformation
            block[i] = Log()(block[i])

    # State Encoding
    # TODO: determine channel_1
    channel_1 = len(block)
    block = Concatenate(axis=1)(block)  # concatenate along channel dimension -> [N, C, H, W]
    # TODO: adapt to different resolutions (64x64)
    block = _residual_block(block, filters=channel_1, cfg=cfg)  # 32x32 -> 16x16
    block = _residual_block(block, filters=channel_1, cfg=cfg)  # 16x16 -> 8x8

    # block: [8, 8, channel_1], inpt: [32, 32, x]
    return block, inpt


def _residual_block(x, filters, cfg):
    # TODO: check activation
    blk1 = Conv2D(filters=filters, kernel_size=4, strides=2, **cfg)(x)
    blk2 = Conv2D(filters=filters, kernel_size=3, strides=1, **cfg)(blk1)
    blk3 = Conv2D(filters=filters, kernel_size=3, strides=1, **cfg)(blk2)
    output = add([x, blk3])
    return output


# TODO: add typing
# TODO: cfg: ReLU activation
def _resnet12(x, filters, cfg):
    """
    Control Agent Architecture
    4 residual modules, [8, 8, 96] -> [8, 8, 32]
    NOT CONFIRMED: 96 -> 64 -> 48 -> 32 -> 32
    """
    for fltr in filters:
        x = _residual_block(x, fltr, cfg)  # ReLU activation
    return x


def _mlp2(x, units, cfg):
    """
    :param x: should be flattened
    :param units: (2, )
    :param cfg: dense_cfg
    :return: output tensor
    """
    assert len(x) == 2
    x = Dense(units[0], **cfg)(x)
    cfg['activation'] = None
    x = Dense(units[1], **cfg)(x)  # No activation
    return x


def _deconv4x(x, filters, kernel_sizes, cfg):
    assert len(filters) == 2
    assert len(kernel_sizes) == 2
    x = Conv2DTranspose(filters=filters[0], kernel_size=kernel_sizes[0], **cfg)(x)
    x = Conv2DTranspose(filters=filters[1], kernel_size=kernel_sizes[1], **cfg)(x)
    return x


def conv_cfg(data_format='channels_first', activation=None, scale=1.0):
    return dict(
        padding='same',
        activation=activation,
        data_format=data_format,
        kernel_initializer=VarianceScaling(scale=2.0 * scale)
    )


def dense_cfg(activation=None, scale=1.0):
    return dict(
        activation=activation,
        kernel_initializer=VarianceScaling(scale=2.0 * scale)
    )


def conv2dlstm_cfg(data_format='channels_first', scale=1.0):
    # TODO: check scale factor
    return dict(
        padding='same',
        data_format=data_format,
        kernel_initializer=VarianceScaling(scale=2.0 * scale)
    )


def deconv_cfg(data_format='channels_first', scale=1.0):
    return dict(
        padding='same',
        data_format=data_format,
        kernel_initializer=VarianceScaling(scale=2.0 * scale)
    )
