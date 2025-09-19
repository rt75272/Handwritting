import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings/info
import tensorflow as tf
from tensorflow import keras
import numpy as np
# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
import tensorflow_probability as tfp
import multiprocessing as mp

# GPU Configuration
def configure_gpu():
    """Configure GPU for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Enable mixed precision for better performance on RTX cards
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"✅ GPU configured successfully!")
            print(f"Available GPUs: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU {i}: {details.get('device_name', 'Unknown')}")
            print(f"Mixed precision enabled: {policy.name}")
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    else:
        print("❌ No GPU found, using CPU")
    return len(gpus) > 0

def configure_threading():
    """Configure TensorFlow threading for optimal CPU utilization."""
    # Get number of CPU cores
    num_cores = mp.cpu_count()
    print(f"🧵 Configuring threading for {num_cores} CPU cores")
    # Configure TensorFlow threading (must be done before any other TF operations)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    print(f"✅ Threading configured:")
    print(f"   Inter-op parallelism: {num_cores}")
    print(f"   Intra-op parallelism: {num_cores}")
    return num_cores

# Configure threading FIRST (before any TensorFlow operations)
num_cpu_cores = mp.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(num_cpu_cores)
tf.config.threading.set_intra_op_parallelism_threads(num_cpu_cores)
# Configure GPU and threading at startup
configure_gpu()
num_cpu_cores = configure_threading()

def monitor_gpu_usage():
    """Monitor and display GPU memory usage."""
    if tf.config.list_physical_devices('GPU'):
        try:
            # Get GPU memory info
            gpu_details = tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0])
            print(f"🖥️  GPU: {gpu_details.get('device_name', 'Unknown')}")
            # Try to get memory info (may not be available on all systems)
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                current_mb = memory_info['current'] / (1024**2)
                peak_mb = memory_info['peak'] / (1024**2)
                print(f"🔢 GPU Memory - Current: {current_mb:.1f} MB, Peak: {peak_mb:.1f} MB")
            except:
                print("🔢 GPU Memory info not available on this system")  
        except Exception as e:
            print(f"⚠️ Could not get GPU info: {e}")
    else:
        print("❌ No GPU available for monitoring")

def load_emnist_letters(batch_size=128):
    """Load and preprocess EMNIST Letters dataset with maximum parallelization."""
    print("Loading EMNIST dataset with parallel processing...")
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
    # GPU-optimized data augmentation
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
    # Maximum parallelization settings
    AUTOTUNE = tf.data.AUTOTUNE
    buffer_size = 100000  # Increased buffer for better shuffling
    prefetch_buffer = AUTOTUNE  # Aggressive prefetching
    num_parallel_calls = num_cpu_cores * 2  # Use more parallel calls than cores
    print(f"🚀 Data pipeline optimization:")
    print(f"   Parallel calls: {num_parallel_calls}")
    print(f"   Shuffle buffer: {buffer_size:,}")
    print(f"   Prefetch buffer: AUTOTUNE")
    # Highly optimized training pipeline
    ds_train = (ds_train
                .map(normalize_img, num_parallel_calls=num_parallel_calls)
                .cache()  # Cache normalized data in memory
                .shuffle(buffer_size, reshuffle_each_iteration=True)
                .map(augment, num_parallel_calls=num_parallel_calls)
                .batch(batch_size, drop_remainder=True)  # Drop remainder for consistent batch sizes
                .prefetch(prefetch_buffer))
    # Optimized test pipeline (no augmentation needed)
    ds_test = (ds_test
               .map(normalize_img, num_parallel_calls=num_parallel_calls)
               .cache()  # Cache test data
               .batch(batch_size, drop_remainder=False)
               .prefetch(prefetch_buffer))
    print(f"✅ Dataset loaded with maximum parallelization")
    return ds_train, ds_test

# --- Optimized Mixup Augmentation ---
def mixup(batch_images, batch_labels, alpha=0.2):
    """Vectorized mixup implementation for better performance."""
    batch_size = tf.shape(batch_images)[0]
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha, size=batch_size)
    lam = tf.constant(lam, dtype=batch_images.dtype)
    # Reshape for broadcasting
    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam_x * batch_images + (1 - lam_x) * tf.gather(batch_images, index)
    mixed_labels = lam_y * batch_labels + (1 - lam_y) * tf.gather(batch_labels, index)
    return mixed_images, mixed_labels

def mixup_tf(images, labels, alpha=0.2):
    """TensorFlow mixup with optimized parallel execution."""
    batch_size = tf.shape(images)[0]
    # Use TensorFlow probability for better GPU utilization
    lam = tfp.distributions.Beta(alpha, alpha).sample([batch_size])
    # Ensure lambda values match the dtype of images for mixed precision compatibility
    lam = tf.cast(lam, images.dtype)
    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])
    # Cast lambda values to match labels dtype
    lam_y = tf.cast(lam_y, labels.dtype)
    # Parallel shuffling and gathering
    index = tf.random.shuffle(tf.range(batch_size))
    # Vectorized operations for better performance
    mixed_images = tf.add(
        tf.multiply(lam_x, images),
        tf.multiply(tf.subtract(1.0, lam_x), tf.gather(images, index))
    )
    mixed_labels = tf.add(
        tf.multiply(lam_y, labels),
        tf.multiply(tf.subtract(1.0, lam_y), tf.gather(labels, index))
    )
    return mixed_images, mixed_labels

def mixup_dataset(ds, alpha=0.2):
    """Apply mixup to dataset with parallel processing."""
    def _mixup(images, labels):
        return mixup_tf(images, labels, alpha)
    # Use maximum parallelism for mixup
    return ds.map(_mixup, num_parallel_calls=tf.data.AUTOTUNE,
                  deterministic=False)  # Allow non-deterministic for better performance

# --- Squeeze-and-Excitation Block ---
def se_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = keras.layers.Dense(filters, activation='sigmoid')(se)
    se = keras.layers.Reshape((1, 1, filters))(se)
    return keras.layers.multiply([input_tensor, se])

def build_emnist_model():
    """Advanced CNN with SE blocks, batch norm, residuals, mixed precision support."""
    inputs = keras.Input(shape=(28,28,1))
    # First block
    x = keras.layers.Conv2D(32, (3,3), padding='same', activation=None)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(32, (3,3), padding='same', activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = se_block(x)
    x = keras.layers.MaxPooling2D(2,2)(x)
    x = keras.layers.Dropout(0.25)(x)
    # Residual block
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
    # Dense layers
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(256, activation='relu')(y)
    y = keras.layers.Dropout(0.5)(y)
    # Output layer - use float32 for mixed precision compatibility
    outputs = keras.layers.Dense(26, activation='softmax', dtype='float32')(y)
    model = keras.Model(inputs, outputs)
    # Compile with optimized settings for GPU
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-4,
        clipnorm=1.0  # Gradient clipping for mixed precision stability
    )
    model.compile(
        optimizer=optimizer,
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
    """Train the model with maximum parallelization and GPU optimization."""
    print(f"Model parameters: {model.count_params():,}")
    # Display system info
    if tf.config.list_physical_devices('GPU'):
        print("🔥 Training on GPU with mixed precision")
        print(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}")
    else:
        print("⚠️ Training on CPU")
    print(f"🧵 Using {num_cpu_cores} CPU cores for parallel processing")
    model.summary()
    # Enhanced callbacks for optimized training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.LearningRateScheduler(cosine_annealing, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    print("Training model with maximum parallelization...")
    # Parallel one-hot encoding
    def one_hot_labels(images, labels):
        labels = tf.one_hot(labels, 26)
        # Ensure consistent dtypes for mixed precision
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return images, labels
    # Apply parallel processing to datasets
    print("🔄 Applying parallel data transformations...")
    ds_train_onehot = ds_train.map(
        one_hot_labels, 
        num_parallel_calls=num_cpu_cores * 2,
        deterministic=False  # Allow non-deterministic for better performance
    )
    ds_test_onehot = ds_test.map(
        one_hot_labels, 
        num_parallel_calls=num_cpu_cores * 2,
        deterministic=False
    )
    # Apply mixup augmentation with parallel processing
    print("🔄 Applying mixup augmentation...")
    ds_train_mixup = mixup_dataset(ds_train_onehot)
    # Modern TensorFlow data optimization
    options = tf.data.Options()
    # Set autotune options for better performance
    options.autotune.enabled = True
    options.autotune.cpu_budget = num_cpu_cores
    ds_train_mixup = ds_train_mixup.with_options(options)
    ds_test_onehot = ds_test_onehot.with_options(options)
    # Train with explicit device placement and parallelization
    print("🚀 Starting optimized training...")
    with tf.device('/GPU:0') if tf.config.list_physical_devices('GPU') else tf.device('/CPU:0'):
        history = model.fit(
            ds_train_mixup,
            epochs=epochs,
            validation_data=ds_test_onehot,
            callbacks=callbacks,
            verbose=1
        )
    # Evaluate on test set
    print("📊 Evaluating final performance...")
    test_loss, test_acc = model.evaluate(ds_test_onehot, verbose=0)
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"✅ Final Test Loss: {test_loss:.4f}")
    # Save model in modern Keras format
    model.save("emnist_letters_model.keras")
    print("\n💾 Model saved as emnist_letters_model.keras")
    return history

def show_predictions(model, ds_test, num_samples=5):
    """Show predictions for a few test samples and save them as images."""
    class_names = [chr(i) for i in range(65, 91)]
    print("\nMaking predictions on test samples...")
    for images, labels in ds_test.take(1):
        preds = model.predict(images)
        # Create a figure for all predictions
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]
        for i in range(num_samples):
            axes[i].imshow(images[i].numpy().reshape(28,28), cmap="gray")
            true_label = chr(labels[i].numpy() + 65)
            pred_label = chr(np.argmax(preds[i]) + 65)
            axes[i].set_title(f"True: {true_label}, Pred: {pred_label}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig("sample_predictions.png", dpi=150, bbox_inches='tight')
        print("✅ Sample predictions saved as 'sample_predictions.png'")
        plt.close()
        break

def predict_image(image_path, model_path="emnist_letters_model.keras"):
    """Load trained model and predict the letter from a user-supplied image file."""
    model = keras.models.load_model(model_path)
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    pred = model.predict(img, verbose=0)
    pred_label = chr(np.argmax(pred[0]) + 65)
    print(f"Predicted letter: {pred_label}")
    # Save prediction plot
    plt.figure(figsize=(4, 4))
    plt.imshow(img[0].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {pred_label}")
    plt.axis("off")
    # Create filename based on input image
    output_filename = f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Prediction saved as '{output_filename}'")
    plt.close()
    return pred_label

def main():
    """Main function to run highly parallelized GPU-optimized training."""
    print("🚀 Starting maximum performance EMNIST training...")
    monitor_gpu_usage()
    num_epochs = 50
    # Optimize batch size for better GPU utilization and parallel processing
    batch_size = 512
    print(f"\n📊 Optimized Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   CPU Cores: {num_cpu_cores}")
    print(f"   Mixed Precision: {tf.keras.mixed_precision.global_policy().name}")
    print(f"   Parallel Threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
    print("\n🔄 Loading dataset with maximum parallelization...")
    ds_train, ds_test = load_emnist_letters(batch_size=batch_size)
    print("\n🏗️ Building optimized model...")
    model = build_emnist_model()
    # Monitor GPU before training
    print("\n🔍 System status before training:")
    monitor_gpu_usage()
    # Start training with timing
    import time
    start_time = time.time()
    print("\n🚀 Starting accelerated training...")
    history = train_and_evaluate(model, ds_train, ds_test, epochs=num_epochs)
    end_time = time.time()
    training_time = end_time - start_time
    # Monitor GPU after training
    print("\n🔍 System status after training:")
    monitor_gpu_usage()
    # Performance summary
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"📈 Average time per epoch: {training_time/len(history.history['loss']):.2f} seconds")
    print("\n🎯 Showing sample predictions...")
    show_predictions(model, ds_test, num_samples=5)
    # Example usage for predictions:
    if os.path.exists("letter_J.jpg"):
        predict_image("letter_J.jpg")
    if os.path.exists("letter_R.jpg"):
        predict_image("letter_R.jpg")

if __name__ == "__main__":
    """The big red action button."""
    main()
