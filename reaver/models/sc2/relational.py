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


@gin.configurable
def build_relational(obs_spec, act_spec, data_format='channels_first', broadcast_non_spatial=False):
    # https://github.com/deepmind/pysc2/blob/master/docs/environment.md#last-actions
    # obs_spec: screen, minimap, player (11,), last_actions (n,)
    # At each time step agents are presented with 4 sources of information:
    # minimap, screen, player, and previous-action.
    assert broadcast_non_spatial is False, 'broadcast_non_spatial should be false for relational agents'

    batch_size = None
    channel_3 = 16
    channel_2 = 96

    # TODO: set spatial_dim <- 64
    screen, screen_input = spatial_block(
        'screen', obs_spec.spaces[0],
        conv_cfg(data_format, 'relu'),
        batch_size=batch_size
    )
    minimap, minimap_input = spatial_block(
        'minimap', obs_spec.spaces[1],
        conv_cfg(data_format, 'relu'),
        batch_size=batch_size
    )

    # TODO: obs_spec[2:] <- ['available_actions', 'player', 'last_actions']
    non_spatial_inputs_list = [
        Input(s.shape, batch_size=batch_size)
        for s in obs_spec.spaces[2:]
    ]
    available_actions = non_spatial_inputs_list[0]
    non_spatial_inputs = Concatenate(
        axis=1, name='non_spatial_inputs'
    )(non_spatial_inputs_list[1:])

    # input_2d: [30, 64], input_3d: [30, 9, 8, 8]
    input_2d = _mlp2(
        Flatten()(non_spatial_inputs),
        units=[128, 64], cfg=dense_cfg('relu')
    )
    input_3d = Concatenate(
        axis=1, name="state_block"
    )([screen, minimap])

    # TODO: treat channel_x as parameters or read from configuration files

    class ExpandDims(Lambda):
        def __init__(self, axis):
            Lambda.__init__(self, lambda x: tf.expand_dims(x, axis))

    # input_3d = ExpandDims(axis=1)(input_3d)

    # # output_3d: [30, 96, 8, 8]
    # # TODO: unroll length
    # output_3d = ConvLSTM2D(
    #     filters=channel_2,
    #     kernel_size=3,
    #     stateful=True,
    #     **conv2dlstm_cfg()
    # )(input_3d)

    output_3d = Conv2D(32, 3, **conv_cfg(data_format, 'relu'))(input_3d)
    output_3d = Conv2D(96, 3, **conv_cfg(data_format, 'relu'))(output_3d)

    # relational_spatial: [30, 32, 8, 8]
    relational_spatial = _resnet12(
        output_3d, filters=[64, 48, 32, 32],
        cfg=conv_cfg(data_format, 'relu')
    )
    # relational_spatial: [30, 16, 32, 32]
    relational_spatial = _deconv4x(
        relational_spatial,
        filters=[channel_3, channel_3],
        kernel_sizes=[4, 4],
        cfg=deconv_cfg(data_format, 'relu')
    )

    # TODO: check scale factor
    # relational_nonspatial: [30, 512]
    relational_nonspatial = _mlp2(
        Flatten()(output_3d),
        units=[512, 512],
        cfg=dense_cfg('relu')
    )

    # shared_features: [30, 512+64=576]
    shared_features = Concatenate(
        axis=1, name='shared_features'
    )([relational_nonspatial, input_2d])  # [512+64, ]

    # [30,]
    value = _mlp2(
        shared_features, units=[256, 1],
        cfg=dense_cfg('relu', scale=0.1)
    )
    value = Squeeze(axis=-1)(value)

    # [30, #actions=549]
    policy_logits = _mlp2(
        shared_features, units=[512, list(act_spec)[0].size()],
        cfg=dense_cfg('relu', scale=0.1)
    )

    mask_actions = Lambda(
        lambda x: tf.where(available_actions > 0, x, -1000 * tf.ones_like(x)),
        name="mask_unavailable_action_ids"
    )
    policy_logits = mask_actions(policy_logits)

    # TODO: check
    return Model(
        inputs=[screen_input, minimap_input] + non_spatial_inputs_list,
        outputs=[shared_features, policy_logits, relational_spatial, value]
    )


def spatial_block(name, space, cfg, batch_size):
    # TODO: tile spatial features with binary masks
    inpt = Input(space.shape, name=name + '_input', batch_size=batch_size)
    # tf.split(x, num_splits, axis=axis) -> List
    block = Split(space.shape[0], axis=1)(inpt)

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


def _residual_block(x, filters, cfg, stride=2):
    blk1 = Conv2D(filters=filters, kernel_size=4, strides=stride, **cfg)(x)
    blk2 = Conv2D(filters=filters, kernel_size=3, strides=1, **cfg)(blk1)
    blk3 = Conv2D(filters=filters, kernel_size=3, strides=1, **cfg)(blk2)
    output = add([blk1, blk3])
    return output


def _resnet12(x, filters, cfg):
    """
    Control Agent Architecture
    4 residual modules, [8, 8, 96] -> [8, 8, 32]
    NOT CONFIRMED: 96 -> 64 -> 48 -> 32 -> 32
    """
    for fltr in filters:
        x = _residual_block(x, fltr, cfg, stride=1)  # ReLU activation
    return x


def _mlp2(x, units, cfg):
    """
    :param x: should be flattened
    :param units: (2, )
    :param cfg: dense_cfg
    :return: output tensor
    """
    assert len(units) == 2
    x = Dense(units[0], **cfg)(x)
    cfg['activation'] = None
    x = Dense(units[1], **cfg)(x)  # No activation
    return x


def _deconv4x(x, filters, kernel_sizes, cfg):
    assert len(filters) == 2
    assert len(kernel_sizes) == 2
    x = Conv2DTranspose(filters=filters[0], kernel_size=kernel_sizes[0], strides=2, **cfg)(x)
    x = Conv2DTranspose(filters=filters[1], kernel_size=kernel_sizes[1], strides=2, **cfg)(x)
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


def deconv_cfg(data_format='channels_first', activation=None, scale=1.0):
    return dict(
        padding='same',
        activation=activation,
        data_format=data_format,
        kernel_initializer=VarianceScaling(scale=2.0 * scale)
    )