#encoding=utf-8
#__author__ = veinpy
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models.matrix_capsule import capnet

class mcapsule_test(tf.test.TestCase):

    def test(self):
        hparams = capnet.capsule_img_base()
        vocab_size = 10
        p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size)
        p_hparams.input_modality["inputs"] = (registry.Modalities.IMAGE, None)
        img_shape = (5, 28,28,1)
        image = np.random.random_integers(
            0, high=255, size=img_shape)
        label = np.random.randint(0,9,size= [img_shape[0],1,1,1])
        with self.test_session() as session:
            features = {
                "inputs": tf.constant(image, dtype=tf.float32),
                "targets": tf.constant(label,dtype=tf.int32)
            }
            model = capnet.Capsule_Img(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
            logits, _ = model(features)
            session.run(tf.global_variables_initializer())
            losses = session.run(logits)


if __name__ == "__main__":
  tf.test.main()