import tensorflow as tf
from keras.losses import mean_squared_error
import tensorflow_addons as tfa

def weighted_mse(yTrue,yPred):

    ones = K.ones_like(yTrue[0,:]) #a simple vector with ones shaped as (60,)
    idx = K.cumsum(ones) #similar to a 'range(1,61)'


    return K.mean((1/idx)*K.square(yTrue-yPred))



def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))


############# PROPOSE CONTRASTIVE LOSS ##############

class SupervisedContrastiveLoss(keras.losses.Loss):

    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    #     def pdist_euclidean(A):
    #         # Euclidean pdist
    #         # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    #         r = tf.reduce_sum(A*A, 1)
    #         # turn r into column vector
    #         r = tf.reshape(r, [-1, 1])
    #         D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    #         return tf.sqrt(D)

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        z = feature_vectors
        y = labels
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
        # Compute logits between the feature vectors ###########
        logits1 = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        ############ To compute ordinal distance between the labels #############
        r = tf.reduce_sum(labels * labels, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = tf.sqrt(r - 2 * tf.matmul(labels, tf.transpose(labels)) + tf.transpose(r))
        D_y = D * 0.0002
        logits2 = tf.clip_by_value(D_y, 0, 2)
        logits = 150 * (logits1 + logits2)
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


########### ORIGINAL CONTRASTIVE LOSS (SUPERVISED CONTRSTIVE LEARNING KHOSLA et. al)
class SupervisedContrastiveLoss1(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss1, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)