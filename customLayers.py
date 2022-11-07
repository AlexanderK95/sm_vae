import tensorflow as tf
from tensorflow.keras import backend as K

class SampleLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(SampleLayer, self).__init__(name=name)
    
    def call(self, inputs):
        mu, log_variance = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        sampled_point = mu + K.exp(log_variance / 2) * epsilon
        return tf.convert_to_tensor(sampled_point)

class CustVariationalLayer (tf.keras.layers.Layer):
    
    def vae_loss(self, x_inp_img, z_reco_img):
        # The references to the layers are resolved outside the function 
        x = K.flatten(x_inp_img)   # B: tensorflow.keras.backend
        z = K.flatten(z_reco_img)
        
        # reconstruction loss per sample 
        # Note: that this is averaged over all features (e.g.. 784 for MNIST) 
        reco_loss = tf.keras.metrics.binary_crossentropy(x, z)
        
        # KL loss per sample - we reduce it by a factor of 1.e-3 
        # to make it comparable to the reco_loss  
        kln_loss  = -0.5e-4 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=1) 
        # mean per batch (axis = 0 is automatically assumed) 
        return K.mean(reco_loss + kln_loss), K.mean(reco_loss), K.mean(kln_loss) 
           
    def call(self, inputs):
        inp_img = inputs[0]
        out_img = inputs[1]
        total_loss, reco_loss, kln_loss = self.vae_loss(inp_img, out_img)
        self.add_loss(total_loss, inputs=inputs)
        self.add_metric(total_loss, name='total_loss', aggregation='mean')
        self.add_metric(reco_loss, name='reco_loss', aggregation='mean')
        self.add_metric(kln_loss, name='kl_loss', aggregation='mean')
        
        return out_img  #not really used in this approach  

class KL_Layer(tf.keras.layers.Layer):
    '''
    @note: Returns the input layers ! Required to allow for z-point calculation
           in a final Lambda layer of the Encoder model    
    '''
    # Standard initialization of layers 
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KL_Layer, self).__init__(*args, **kwargs)

    # The implementation interface of the Layer
    def call(self, inputs, fact = 4.5e-4):
        mu      = inputs[0]
        log_var = inputs[1]
        # Note: from other analysis we know that the backend applies tf.math.functions 
        # "fact" must be adjusted - for MNIST reasonable values are in the range of 0.65e-4 to 6.5e-4
        kl_mean_batch = -fact * K.mean(1 + log_var - K.square(mu) - K.exp(log_var))
        # We add the loss via the layer's add_loss() - it will be added up to other losses of the model     
        self.add_loss(kl_mean_batch, inputs=inputs)
        # We add the loss information to the metrics displayed during training 
        self.add_metric(kl_mean_batch, name='kl_loss', aggregation='mean')
        return inputs