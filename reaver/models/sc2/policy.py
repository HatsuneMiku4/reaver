import gin
import tensorflow as tf
from reaver.models.base import MultiPolicy
from .relational import *


@gin.configurable
class SC2MultiPolicy(MultiPolicy):
    def __init__(self, act_spec, logits):
        super().__init__(act_spec, logits)

        args_mask = tf.constant(act_spec.spaces[0].args_mask, dtype=tf.float32)
        act_args_mask = tf.gather(args_mask, self.inputs[0])  # masked action_id
        act_args_mask = tf.transpose(act_args_mask, [1, 0])

        # self.dists = [self.make_dist(s, l) for s, l in zip(act_spec.spaces, logits)]
        self.logli = self.dists[0].log_prob(self.inputs[0])  # masked action_id
        for i in range(1, len(self.dists)):
            self.logli += act_args_mask[i-1] * self.dists[i].log_prob(self.inputs[i])


# self.policy = policy_cls(act_spec, self.model.outputs[:-1])
# return Model(
#     inputs=[screen_input, minimap_input] + non_spatial_inputs,
#     outputs=logits + [value]
# )

# TODO: implement Relational MultiPolicy
# see agents/a2c.py
# tf_inputs = self.model.inputs + self.policy.inputs + self.loss_inputs
# self.policy = policy_cls(act_spec, self.model.outputs[:-1])
# self.sess_mgr.run([self.policy.sample, self.value], self.model.inputs, obs)

class SC2RelationalMultiPolicy(MultiPolicy):

    # noinspection PyMissingConstructor
    def __init__(self, act_spec, logits):
        """
        policy_logits: masked, (len(act_spec), )
        relational_spatial: _deconv4x output, (32, 32, channel_3)
        :param act_spec:
        :param logits: [policy_logits, relational_spatial]
        """
        # super().__init__(act_spec, logits)

        shared_features, policy_logits, relational_spatial = logits
        policy_dist = self.make_dist(act_spec.spaces[0], policy_logits)
        policy_sample = policy_dist.sample()

        # action_embed: [30, 16]
        action_embed = Embedding(
            policy_logits.shape[-1], 16
        )(policy_sample)

        # action_logits: [30, 16+16, 32, 32]
        action_logits = Concatenate(axis=1)([
            relational_spatial,
            Broadcast2D(size=relational_spatial.shape[-1])(action_embed)
        ])

        # conditioned_shared_feature: [30, 576+16=592]
        conditioned_shared_feature = Concatenate(axis=-1)(
            [shared_features, action_embed]
        )

        self.logits = [policy_logits, ]
        data_format = 'channel_first'     # TODO: refactor me
        for space in list(act_spec)[1:]:  # [0]: function_id
            if space.is_spatial():
                conv_layer = Conv2D(1, 1, **conv_cfg(data_format, scale=0.1))
                logits.append(conv_layer(action_logits))
                logits[-1] = Flatten()(logits[-1])
            else:
                fc_layer = Dense(space.size(), **dense_cfg(scale=0.1))
                logits.append(fc_layer(conditioned_shared_feature))

        self.dists = [self.make_dist(s, l) for s, l in zip(act_spec.spaces, self.logits)]
        self.entropy = sum([dist.entropy() for dist in self.dists])
        self.sample = [policy_sample, ] + [dist.sample() for dist in self.dists[1:]]
        args_mask = tf.constant(act_spec.spaces[0].args_mask, dtype=tf.float32)
        self.inputs = [Input([*s.shape], dtype=s.dtype) for s in act_spec]
        act_args_mask = tf.gather(args_mask, self.inputs[0])  # masked action_id
        act_args_mask = tf.transpose(act_args_mask, [1, 0])
        self.logli = self.dists[0].log_prob(self.inputs[0])
        for i in range(1, len(self.dists)):
            self.logli += act_args_mask[i - 1] * self.dists[i].log_prob(self.inputs[i])


