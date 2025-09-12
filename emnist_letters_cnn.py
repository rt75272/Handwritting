import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings/info
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
import tensorflow_probability as tfp

def load_emnist_letters(batch_size=128):
    """Load and preprocess EMNIST Letters dataset with data augmentation."""
    print("Loading EMNIST dataset...")
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/letters',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )
    def normalize_img(image, label):
        """Normalize images to [0, 1] and adjust labels to [0-25]."""
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32) - 1
        return image, label
    # Data augmentation for training
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1)
    ])
    def augment(image, label):
        """Apply data augmentation."""
        image = data_augmentation(image)
        return image, label
    ds_train = ds_train.map(normalize_img).map(augment).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(normalize_img).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test

# --- Mixup Augmentation ---
def mixup(batch_images, batch_labels, alpha=0.2):
    batch_size = tf.shape(batch_images)[0]
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * batch_images + (1 - lam) * tf.gather(batch_images, index)
    mixed_labels = lam * batch_labels + (1 - lam) * tf.gather(batch_labels, index)
    return mixed_images, mixed_labels

def mixup_tf(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tfp.distributions.Beta(alpha, alpha).sample([batch_size])
    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam_x * images + (1 - lam_x) * tf.gather(images, index)
    mixed_labels = lam_y * labels + (1 - lam_y) * tf.gather(labels, index)
    return mixed_images, mixed_labels

def mixup_dataset(ds, alpha=0.2):
    def _mixup(images, labels):
        return mixup_tf(images, labels, alpha)
    return ds.map(_mixup, num_parallel_calls=tf.data.AUTOTUNE)

# --- Squeeze-and-Excitation Block ---
def se_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = keras.layers.Dense(filters, activation='sigmoid')(se)
    se = keras.layers.Reshape((1, 1, filters))(se)
    return keras.layers.multiply([input_tensor, se])

def build_emnist_model():
    """Advanced CNN with SE blocks, batch norm, residuals, label smoothing, AdamW."""
    inputs = keras.Input(shape=(28,28,1))
    x = keras.layers.Conv2D(32, (3,3), padding='same', activation=None)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(32, (3,3), padding='same', activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = se_block(x)
    x = keras.layers.MaxPooling2D(2,2)(x)
    x = keras.layers.Dropout(0.25)(x)
    res = keras.layers.Conv2D(64, (1,1), padding='same')(x)
    y = keras.layers.Conv2D(64, (3,3), padding='same', activation=None)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Conv2D(64, (3,3), padding='same', activation=None)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = se_block(y)
    y = keras.layers.Add()([y, res])
    y = keras.layers.MaxPooling2D(2,2)(y)
    y = keras.layers.Dropout(0.25)(y)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(256, activation='relu')(y)
    y = keras.layers.Dropout(0.5)(y)
    outputs = keras.layers.Dense(26, activation='softmax')(y)
    model = keras.Model(inputs, outputs)
    # Compile with AdamW optimizer and label smoothing
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

# --- Cosine Annealing Learning Rate ---
def cosine_annealing(epoch, lr):
    T_max = 30  # Total epochs
    eta_min = 1e-5
    return eta_min + (0.001 - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2

def train_and_evaluate(model, ds_train, ds_test, epochs=30):
    """Train the model with advanced callbacks and evaluate on test set."""
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.LearningRateScheduler(cosine_annealing)
    ]
    print("Training model...")
    # Convert labels to one-hot for label smoothing
    def one_hot_labels(images, labels):
        labels = tf.one_hot(labels, 26)
        return images, labels
    ds_train_onehot = ds_train.map(one_hot_labels)
    ds_test_onehot = ds_test.map(one_hot_labels)
    ds_train_mixup = mixup_dataset(ds_train_onehot)
    history = model.fit(ds_train_mixup, epochs=epochs, validation_data=ds_test_onehot, callbacks=callbacks)
    test_loss, test_acc = model.evaluate(ds_test_onehot)
    print(f"\n✅ Test Accuracy: {test_acc:.2f}")
    model.save("emnist_letters_model.h5")
    print("\nModel saved as emnist_letters_model.h5")
    return history

def show_predictions(model, ds_test, num_samples=5):
    """Show predictions for a few test samples."""
    class_names = [chr(i) for i in range(65, 91)]
    print("\nMaking predictions on test samples...")
    for images, labels in ds_test.take(1):
        preds = model.predict(images)
        for i in range(num_samples):
            plt.imshow(images[i].numpy().reshape(28,28), cmap="gray")
            true_label = chr(labels[i].numpy() + 65)
            pred_label = chr(np.argmax(preds[i]) + 65)
            plt.title(f"True: {true_label}, Pred: {pred_label}")
            plt.axis("off")
            plt.show()

def predict_image(image_path, model_path="emnist_letters_model.h5"):
    """Load trained model and predict the letter from a user-supplied image file."""
    model = keras.models.load_model(model_path)
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    pred = model.predict(img)
    pred_label = chr(np.argmax(pred[0]) + 65)
    print(f"Predicted letter: {pred_label}")
    plt.imshow(img[0].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {pred_label}")
    plt.axis("off")
    plt.show()
    return pred_label

def main():
    """Main function to run training, evaluation, and sample predictions."""
    num_epochs = 50
    ds_train, ds_test = load_emnist_letters()
    model = build_emnist_model()
    train_and_evaluate(model, ds_train, ds_test, epochs=num_epochs)
    show_predictions(model, ds_test, num_samples=5)
    # Example usage for prediction(s):
    predict_image("letter_J.jpg")
    predict_image("letter_R.jpg")

if __name__ == "__main__":
    """The big red action button."""
    main()
