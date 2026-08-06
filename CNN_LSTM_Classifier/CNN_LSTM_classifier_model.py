import tensorflow as tf
import tensorflow_probability as tfp
import keras

@keras.saving.register_keras_serializable(package="DetectionLoss")
class DetectionLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        margin_mean=3.0,
        lambda_mean=0.0,
        margin_tail=1.0,
        lambda_tail=0.0,
        q_signal=0.05,
        q_noise=0.95,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin_mean = margin_mean
        self.lambda_mean = lambda_mean
        self.margin_tail = margin_tail
        self.lambda_tail = lambda_tail
        self.q_signal = q_signal
        self.q_noise = q_noise
        
        self.bce = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=0.25,
            gamma=4.0,
            from_logits=True  # IMPORTANT: False, if you use sigmoid in last layer
        )
        
        """
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        """
        
    def call(self, y_true, y_pred):
        labels = y_true

        bce_loss = self.bce(labels, y_pred)

        labels = tf.cast(tf.reshape(labels, [-1]), tf.bool)
        logits = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        # Split signal / noise
        signal_scores = tf.boolean_mask(logits, labels)
        noise_scores  = tf.boolean_mask(logits, ~labels)

        # --- Mean–mean margin --- #
        def mean_margin():
            mu_s = tf.reduce_mean(signal_scores)
            mu_n = tf.reduce_mean(noise_scores)
            return tf.maximum(0.0, self.margin_mean - (mu_s - mu_n))

        mean_term = tf.cond(
            tf.logical_and(tf.size(signal_scores) > 0,
                        tf.size(noise_scores) > 0),
            mean_margin,
            lambda: 0.0
        )

        # --- Quantile–Quantile tail margin --- #
        def qq_margin():
            q_s = tfp.stats.percentile(
                signal_scores, 100.0 * self.q_signal
            )
            q_n = tfp.stats.percentile(
                noise_scores, 100.0 * self.q_noise
            )
            gap = q_s - q_n
            return tf.maximum(0.0, self.margin_tail - gap)

        qq_term = tf.cond(
            tf.logical_and(tf.size(signal_scores) > 0,
                        tf.size(noise_scores) > 0),
            qq_margin,
            lambda: 0.0
        )

        tf.debugging.assert_greater_equal(
            bce_loss + self.lambda_mean * mean_term,
            0.0,
            message="DetectionLoss went negative (should be impossible)"
        )
        
        return (
            bce_loss
            + self.lambda_mean * mean_term
            + self.lambda_tail * qq_term
        )
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "margin_mean": self.margin_mean,
            "lambda_mean": self.lambda_mean,
            "margin_tail": self.margin_tail,
            "lambda_tail": self.lambda_tail,
            "q_signal": self.q_signal,
            "q_noise": self.q_noise,
        })
        return config
        
@keras.saving.register_keras_serializable(package="CNN_LSTM_classifier_model") # Only add if you want to create a custom object and tell Keras how to handle it
class CNN_LSTM_classifier_model:
    """
    Wrapper class for a Many-to-One CNN LSTM classifier model.
    """
    
    def __init__(self,
                input_shape, # (window_size, channels)
                cnn_filters,
                cnn_kernel_sizes,
                cnn_strides,
                cnn_spatial_dropout_rates,
                lstm_units,
                dropout_rate,
                activation,
                learning_rate,
                weight_decay,
                loss_config,
                recurrent_dropout_rate=0.0):
        """
        Initialize the model configuration.
        """
        self.input_shape = input_shape
        self.cnn_filters = cnn_filters
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.cnn_strides = cnn_strides
        self.cnn_spatial_dropout_rates = cnn_spatial_dropout_rates
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.activation = activation,
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_config = loss_config
        self.act = tf.keras.layers.Activation(activation)
        
        if not (len(cnn_filters) == len(cnn_kernel_sizes) == len(cnn_strides) == len(cnn_spatial_dropout_rates)):
            raise ValueError("cnn_filters, cnn_kernel_sizes, cnn_strides, and cnn_spatial_dropout_rates must have the same length")
        
        # Build and Compile immediately
        self.model = self._build_and_compile_model()
      
    def get_config(self):
        
        #REQUIRED: Returns the config dictionary for Keras serialization.
        
        return {
            "input_shape": self.input_shape,
            "cnn_filters": self.cnn_filters,
            "cnn_kernel_sizes": self.cnn_kernel_sizes,
            "cnn_strides": self.cnn_strides,
            "cnn_spatial_dropout_rates": self.cnn_spatial_dropout_rates,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "recurrent_dropout_rate": self.recurrent_dropout_rate,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "loss_config": self.loss_config,
        }

    @classmethod
    def from_config(cls, config):
        
        #REQUIRED: Recreates the object from the config dictionary.
        
        return cls(**config)
    
    def _build_and_compile_model(self):
        """CNN LSTM Architecture """
        
        x = tf.keras.layers.Input(shape=self.input_shape)
        inputs = x
        
        # --- CNN as feature extractors --- #
        for filters, kernel_size, stride, sd_rate in zip(
            self.cnn_filters,
            self.cnn_kernel_sizes,
            self.cnn_strides,
            self.cnn_spatial_dropout_rates,
        ):
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding="same",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)
            
            x = self.act(x)

            if sd_rate > 0.0:
                x = tf.keras.layers.SpatialDropout1D(sd_rate)(x)
        
  
        # --- LSTM --- #
        x = tf.keras.layers.LSTM(
            units=self.lstm_units,
            return_sequences=False,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout_rate,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
        )(x)


        # --- Head: Read-out --- #
        x = tf.keras.layers.Dense(32, activation='silu', kernel_initializer=tf.keras.initializers.HeNormal())(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        outputs = tf.keras.layers.Dense(1, activation=None, kernel_initializer="glorot_uniform")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Optimizer and Compile
        optimizer = tf.keras.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        
        # ----- Model compilation ----- #
        loss_fn = DetectionLoss(**self.loss_config)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(
                    threshold=0.0, name="accuracy"
                ),
                tf.keras.metrics.AUC(
                    from_logits=True, name="auc"
                ),
            ]
        )
            
        print(f"--- CNN LSTM Model Built ---")
        return model
        
    def train(self, train_ds, val_ds, epochs=20, callbacks=None):
        """
        Executes the training loop.
        """
        print(f"Starting training for {epochs} epochs...")
    
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history
    
    def save(self, filepath):
        """Helper to save the internal Keras model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
