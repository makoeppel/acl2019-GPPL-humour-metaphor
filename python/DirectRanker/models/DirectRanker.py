import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.base import BaseEstimator
import gc

class RBFKernelFn(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__()
        
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

    def call(x):
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=tf.nn.softplus(0.1 * self._amplitude), length_scale=tf.nn.softplus(5. * self._length_scale))


class DirectRanker(BaseEstimator):
    """
    TODO
    """

    def __init__(self,
                 # DirectRanker HPs
                 hidden_layers_dr=[256, 128, 64, 20],
                 feature_activation_dr='tanh',
                 ranking_activation_dr='tanh',
                 feature_bias_dr=True,
                 kernel_initializer_dr=tf.random_normal_initializer,
                 kernel_regularizer_dr=0.0,
                 drop_out=0.5,
                 GaussianNoise=0,
                 batch_norm=True,
                 gp_inducing_points=10,
                 # Common HPs
                 scale_factor_train_sample=5,
                 batch_size=200,
                 loss=tf.keras.losses.MeanSquaredError(),
                 learning_rate=0.001,
                 learning_rate_decay_rate=1,
                 learning_rate_decay_steps=1000,
                 optimizer=tf.keras.optimizers.Adam,
                 epoch=10,
                 # other variables
                 verbose=0,
                 validation_size=0.0,
                 num_features=0,
                 random_seed=42,
                 name="DirectRanker",
                 dtype=tf.float32,
                 print_summary=False,
                 ):

        # DirectRanker HPs
        self.hidden_layers_dr = hidden_layers_dr
        self.feature_activation_dr = feature_activation_dr
        self.ranking_activation_dr = ranking_activation_dr
        self.feature_bias_dr = feature_bias_dr
        self.kernel_initializer_dr = kernel_initializer_dr
        self.kernel_regularizer_dr = kernel_regularizer_dr
        self.drop_out = drop_out
        self.GaussianNoise = GaussianNoise
        self.batch_norm = batch_norm
        self.gp_inducing_points = gp_inducing_points
        # Common HPs
        self.scale_factor_train_sample = scale_factor_train_sample
        self.batch_size = batch_size
        self.loss = loss
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.optimizer = optimizer
        self.epoch = epoch
        # other variables
        self.verbose = verbose
        self.validation_size = validation_size
        self.num_features = num_features
        self.random_seed = random_seed
        self.name = name
        self.dtype = dtype
        self.print_summary = print_summary

    def _build_model(self):
        """
        TODO
        """
        # Placeholders for the inputs
        self.x0 = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="x0"
        )

        self.x1 = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="x1"
        )

        input_layer = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="input"
        )

        nn = tf.keras.layers.Dense(
            units=self.hidden_layers_dr[0],
            activation=self.feature_activation_dr,
            use_bias=self.feature_bias_dr,
            kernel_initializer=self.kernel_initializer_dr(seed=self.random_seed),
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            bias_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            name="nn_hidden_0"
        )(input_layer)
 
        if self.GaussianNoise > 0:
            nn = tf.keras.layers.GaussianNoise(self.GaussianNoise)(nn)

        if self.drop_out > 0:
            nn = tf.keras.layers.Dropout(self.drop_out)(nn)
            
        if self.batch_norm:
            nn = tf.keras.layers.BatchNormalization()(nn)

        for i in range(1, len(self.hidden_layers_dr)):
            nn = tf.keras.layers.Dense(
                units=self.hidden_layers_dr[i],
                activation=self.feature_activation_dr,
                use_bias=self.feature_bias_dr,
                kernel_initializer=self.kernel_initializer_dr(seed=self.random_seed),
                kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                bias_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                name="nn_hidden_" + str(i)
            )(nn)
            
            if self.GaussianNoise > 0:
                nn = tf.keras.layers.GaussianNoise(self.GaussianNoise)(nn)

            if self.drop_out > 0:
                nn = tf.keras.layers.Dropout(self.drop_out)(nn)
                
            if self.batch_norm:
                nn = tf.keras.layers.BatchNormalization()(nn)

        feature_part = tf.keras.models.Model(input_layer, nn, name='feature_part')

        if self.print_summary:
            feature_part.summary()

        nn0 = feature_part(self.x0)
        nn1 = feature_part(self.x1)

        subtracted = tf.keras.layers.Subtract()([nn0, nn1])
        
        if self.gp_inducing_points > 0:
            subtracted = tfp.layers.VariationalGaussianProcess(self.gp_inducing_points, RBFKernelFn())(subtracted)

        out = tf.keras.layers.Dense(
            units=1,
            activation=self.ranking_activation_dr,
            use_bias=False,
            kernel_initializer=self.kernel_initializer_dr(seed=self.random_seed),
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            name="ranking_part"
        )(subtracted)

        self.model = tf.keras.models.Model(
            inputs=[self.x0, self.x1],
            outputs=out,
            name='DR'
        )

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        self.model.compile(
            optimizer=self.optimizer(lr_schedule),
            loss=self.loss,
            metrics=['acc']
        )

        if self.print_summary:
            self.model.summary()

    def fit(self, x, y, **fit_params):
        """
        TODO
        """
        self._build_model()
        
        x = np.array(x)

        for i in range(self.epoch):
            
            idx0, idx1, _ = list(zip(*[y[i] for i in np.random.randint(0, len(y), self.scale_factor_train_sample*len(y))]))
            idx0 = np.array(idx0)
            idx1 = np.array(idx1)
            x0_cur = x[idx0]
            x1_cur = x[idx1]

            print('Epoch {}/{}'.format(i + 1, self.epoch))

            self.model.fit(
                x=[x0_cur, x1_cur],
                y=np.ones(self.scale_factor_train_sample*len(y)),
                batch_size=self.scale_factor_train_sample*self.batch_size,
                epochs=1,
                verbose=self.verbose,
                shuffle=True,
                validation_split=self.validation_size
            )
            # https://github.com/tensorflow/tensorflow/issues/14181
            # https://github.com/tensorflow/tensorflow/issues/30324
            gc.collect()
            
    def fit_gppl(self, a1_train, a2_train, items_feat, prefs_train, **fit_params):
        self._build_model()
        
        x0 = np.array([items_feat[i] for i in a1_train])
        x1 = np.array([items_feat[i] for i in a2_train])
        
        self.model.fit(
            x=[x0, x1],
            y=prefs_train,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=self.verbose,
            shuffle=True,
            validation_split=self.validation_size
        )
        
        gc.collect()

    def predict_proba(self, features):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features, np.zeros(np.shape(features))], batch_size=self.batch_size, verbose=self.verbose)

        return res
    
    def predict_ranking(self, x0, x1):
        return self.model.predict([x0, x1], batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, features, threshold):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        features_conv = np.expand_dims(features, axis=2)

        res = self.model.predict([features, np.zeros(np.shape(features))], batch_size=self.batch_size, verbose=self.verbose)

        return [1 if r > threshold else 0 for r in res]
