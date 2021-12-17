import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import facenet

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

class LoadRecogModel():
    """  Importing and running isolated TF graph """
    def __init__(self):
        # Create local graph and use it in the session
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))

            with self.sess.as_default():
                self.modeldir = my_absolute_dirpath+"/models/facenet/20170512-110547.pb"
                facenet.load_model(self.modeldir)

                self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

    def embedding_tensor(self):
        return(self.embedding_size)

    def embed(self, data, emb_array):
        """ Running the activation operation previously imported """

        feed_dict = {self.images_placeholder: data, self.phase_train_placeholder: False}
        emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return emb_array
