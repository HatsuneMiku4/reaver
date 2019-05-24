import gin
import tensorflow as tf
from reaver.models.base import MultiPolicy


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

    # Override MultiPolicy methods
    def __init__(self, act_spec, logits):
        """
        policy_logits: masked, (len(act_spec), )
        relational_spatial: _deconv4x output, (32, 32, channel_3)
        :param act_spec:
        :param logits: [policy_logits, relational_spatial]
        """
        super().__init__(act_spec, logits)

        # MultiPolicy.__init__:
        # self.logits = logits
        # self.inputs: given action calculate probs
        # self.inputs = [tf.placeholder(s.dtype, [None, *s.shape]) for s in act_spec]
        #
        # self.dists = [self.make_dist(s, l) for s, l in zip(act_spec.spaces, logits)]
        #
        # self.entropy = sum([dist.entropy() for dist in self.dists])
        # log_prob: Log probability density/mass function, action -> prob
        # self.logli = sum([dist.log_prob(act) for dist, act in zip(self.dists, self.inputs)])
        #
        # self.sample = [dist.sample() for dist in self.dists]

        # TODO: calculate logits for arguments based on self.sample[0]

        args_mask = tf.constant(act_spec.spaces[0].args_mask, dtype=tf.float32)
        act_args_mask = tf.gather(args_mask, self.inputs[0])  # masked action_id
        act_args_mask = tf.transpose(act_args_mask, [1, 0])

        # self.dists = [self.make_dist(s, l) for s, l in zip(act_spec.spaces, logits)]
        self.logli = self.dists[0].log_prob(self.inputs[0])  # masked action_id
        for i in range(1, len(self.dists)):
            self.logli += act_args_mask[i - 1] * self.dists[i].log_prob(self.inputs[i])

    @staticmethod
    def make_dist(space, logits):
        # # tfp is really heavy on init, better to lazy load
        # import tensorflow_probability as tfp
        #
        # if space.is_continuous():
        #     mu, logstd = tf.split(logits, 2, axis=-1)
        #     return tfp.distributions.MultivariateNormalDiag(mu, tf.exp(logstd))
        # else:
        #     return tfp.distributions.Categorical(logits)
        return super().make_dist(space, logits)
