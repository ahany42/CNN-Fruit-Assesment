import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection     = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches":    self.num_patches,
            "projection_dim": self.projection.units,
        })
        return config

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, transformer_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.dropout_rate = dropout_rate

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.drop1 = layers.Dropout(dropout_rate)
        self.add1 = layers.Add()

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(transformer_units[0], activation=tf.nn.gelu)
        self.drop2 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(transformer_units[1], activation=tf.nn.gelu)
        self.drop3 = layers.Dropout(dropout_rate)
        self.dense3 = layers.Dense(projection_dim)
        self.add2 = layers.Add()

    def call(self, inputs, training=False):
        x1 = self.norm1(inputs)
        attn = self.att(x1, x1, training=training)
        attn = self.drop1(attn, training=training)
        x2 = self.add1([attn, inputs])
        x3 = self.norm2(x2)
        x3 = self.dense1(x3)
        x3 = self.drop2(x3, training=training)
        x3 = self.dense2(x3)
        x3 = self.drop3(x3, training=training)
        x3 = self.dense3(x3)
        return self.add2([x3, x2])

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "num_heads": self.num_heads,
            "transformer_units": self.transformer_units,
            "dropout_rate": self.dropout_rate,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + tf.cos(
            np.pi * (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        ))
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }
    
def test(model, model_path):
    img_size = 224
    test_folder = 'C:\\Users\\Admin\\Downloads\\NN Test Scripts\\Test'
    output_csv = f'C:\\Users\\Admin\\Downloads\\NN Test Scripts\\output\\{model}.csv'

    expected_classes = {
        0: 'banana_overripe',
        1: 'banana_ripe',
        2: 'banana_rotten',
        3: 'banana_unripe',
        4: 'tomato_fully_ripened',
        5: 'tomato_green',
        6: 'tomato_half_ripened'
    }

    print(f"Loading {model} model...")

    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        return

    try:
        if model == 'vit':
            custom_objects = {
                "TransformerBlock": TransformerBlock,
                "Patches": Patches,
                "PatchEncoder": PatchEncoder,
                "WarmUpCosine": WarmUpCosine
            }
            model = tf.keras.models.load_model(model_path,custom_objects=custom_objects)
        else:
            model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except ValueError as e:
        print(f"Error loading model: {e}")
        return

    def preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img).astype('float32')
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)

    results = []

    print(f"Predicting for images in: {test_folder}")
    for fname in sorted(os.listdir(test_folder)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(test_folder, fname)
        img_tensor = preprocess_image(img_path)
        preds = model.predict(img_tensor, verbose=0)
        pred_class = int(np.argmax(preds[0]))

        results.append((fname, pred_class))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageID', 'Class'])
        writer.writerows(results)

    print(f"\nPredictions saved to '{output_csv}'")


# Run tests
test('efficientnet', 'C:\\Users\\Admin\\Downloads\\NN Test Scripts\\models\\efficientnet_model.h5')
test('googlenet', 'C:\\Users\\Admin\\Downloads\\NN Test Scripts\\models\\googlenet.keras')
test('vit', 'C:\\Users\\Admin\\Downloads\\NN Test Scripts\\models\\vit_model.h5')