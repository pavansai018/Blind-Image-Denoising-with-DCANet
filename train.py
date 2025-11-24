import tensorflow as tf
import os
import datetime
from typing import Optional, Tuple
from losses import dummy_loss

from DCANet import dca_net
from losses import charbonnier_loss, edge_loss
from scheduler import WarmupThenCosine
import dataset_generation
from model_saving_callback import EpochModelCheckpoint


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class DCTrainer:
    """
    Comprehensive trainer for DCANet with early stopping, model checkpointing,
    and training resumption capabilities.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (256, 256, 3),
                 filters: int = 64,
                 kernel_size: int = 3,
                 out_channels: int = 3,
                 base_lr: float = 1e-4,
                 lr_min: float = 1e-6,
                 warmup_epochs: int = 3,
                 extra_epochs: int = 40,
                 checkpoint_dir: str = "checkpoints",
                 logs_dir: str = "logs"):

        self.schedule = None
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.base_lr = base_lr
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.extra_epochs = extra_epochs

        # Create directories
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Initialize model and training state
        self.model = None
        self.initial_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'loss': [], 'val_loss': [],
            'output_loss': [], 'val_output_loss': [],
            'conv_tail_loss': [], 'val_conv_tail_loss': [],
            'noise_loss': [], 'val_noise_loss': [],
            'lr': []
        }

    def build_model(self) -> tf.keras.Model:
        """Build the DCANet model with multi-output structure"""
        model = dca_net(
            input_shape=self.input_shape,
            filters=self.filters,
            kernel_size=self.kernel_size,
            out_channels=self.out_channels
        )
        return model

    def custom_loss(self, y_true, y_pred):
        """
        Simplified loss matching the PyTorch implementation
        y_pred contains: [final_output, conv_tail, noise_estimation]
        y_true is the ground truth clean image
        """
        # Unpack outputs - we only use final_output and noise_estimation
        final_output = y_pred[0]  # Final denoised image (equivalent to 'restored' in PyTorch)
        noise_est = y_pred[2]  # noise estimation (equivalent to 'noise_level' in PyTorch)

        # Charbonnier loss for the final output
        charbonnier_loss_val = charbonnier_loss(y_true, final_output)

        # Edge loss for the final output
        edge_loss_val = edge_loss(y_true, final_output)

        # TV regularization for noise estimation (matches PyTorch calculation)
        tv_loss = self.tv_regularization(noise_est)

        # Combine losses with same weights as PyTorch
        total_loss = charbonnier_loss_val + 0.1 * edge_loss_val + 0.05 * tv_loss

        return total_loss

    def tv_regularization(self, noise_map):
        """Total Variation regularization matching PyTorch calculation"""
        if len(noise_map.shape) == 3:
            noise_map = tf.expand_dims(noise_map, axis=0)
        # Calculate horizontal differences
        h_tv = tf.reduce_mean(tf.square(noise_map[:, 1:, :, :] - noise_map[:, :-1, :, :]))

        # Calculate vertical differences
        w_tv = tf.reduce_mean(tf.square(noise_map[:, :, 1:, :] - noise_map[:, :, :-1, :]))

        return h_tv + w_tv

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint file if it exists"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir)
                       if f.startswith('dcanet_epoch_') and f.endswith('.h5')]
        if not checkpoints:
            return None

        # Extract epoch numbers and find the latest
        def extract_epoch(filename):
            return int(filename.split('_')[2].split('.')[0])

        latest_checkpoint = max(checkpoints, key=extract_epoch)
        return os.path.join(self.checkpoint_dir, latest_checkpoint)

    def load_model_and_weights(self, checkpoint_path: Optional[str] = None) -> Tuple[tf.keras.Model, int]:
        """Load model and weights, handle training resumption"""

        # Build the model
        model = self.build_model()

        # If no specific checkpoint provided, try to find the latest
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading weights from: {checkpoint_path}")

            # Custom object scope for loading
            custom_objects = {
                'WarmupThenCosine': WarmupThenCosine,
                'custom_loss': self.custom_loss,
                'charbonnier_loss': charbonnier_loss,
                'edge_loss': edge_loss
            }

            with tf.keras.utils.custom_object_scope(custom_objects):
                model.load_weights(checkpoint_path)

            # Extract epoch number from filename
            epoch_num = int(os.path.basename(checkpoint_path).split('_')[2].split('.')[0])
            self.initial_epoch = epoch_num + 1
            print(f"Resuming training from epoch {self.initial_epoch}")

            # Load training history if exists
            history_path = os.path.join(self.checkpoint_dir, 'training_history.npy')
            if os.path.exists(history_path):
                import numpy as np
                self.history = np.load(history_path, allow_pickle=True).item()
                self.best_val_loss = min(self.history['val_loss']) if self.history['val_loss'] else float('inf')

        else:
            print("No checkpoint found. Starting training from scratch.")
            self.initial_epoch = 0

        return model, self.initial_epoch

    def create_callbacks(self, steps_per_epoch: int) -> list:
        """Create training callbacks including early stopping and model checkpointing"""

        # Model checkpoint - save after every epoch
        # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(self.checkpoint_dir, 'dcanet_epoch_{epoch:03d}.h5'),
        #     save_weights_only=False,
        #     save_freq='epoch',
        #     verbose=1
        # )
        checkpoint_callback = EpochModelCheckpoint(self.checkpoint_dir)

        # Best model checkpoint
        best_model_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_dir, 'dcanet_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        # TensorBoard
        log_dir = os.path.join(self.logs_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )

        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: self.lr_schedule(epoch, lr, steps_per_epoch),
            verbose=1
        )

        # Custom callback to save history
        class HistorySaver(tf.keras.callbacks.Callback):
            def __init__(self, trainer):
                self.trainer = trainer

            def on_epoch_end(self, epoch, logs=None):
                # Save history after each epoch
                import numpy as np
                history_path = os.path.join(self.trainer.checkpoint_dir, 'training_history.npy')
                np.save(history_path, self.trainer.history)

        history_saver = HistorySaver(self)

        return [
            checkpoint_callback,
            best_model_callback,
            early_stopping,
            tensorboard_callback,
            lr_scheduler,
            history_saver
        ]

    def lr_schedule(self, epoch, lr, steps_per_epoch):
        """Learning rate schedule compatible with Keras callbacks"""
        self.schedule = WarmupThenCosine(
            base_lr=self.base_lr,
            lr_min=self.lr_min,
            num_epochs=100,  # You can adjust this
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=self.warmup_epochs,
            extra_epochs=self.extra_epochs
        )

        # Calculate current step
        current_step = (epoch + self.initial_epoch) * steps_per_epoch
        return self.schedule(current_step).numpy()

    def update_history(self, logs):
        """Update training history with current epoch metrics"""
        for key, value in logs.items():
            if key in self.history:
                self.history[key].append(value)

        # Always track learning rate
        if hasattr(self.model.optimizer, 'lr'):
            self.history['lr'].append(self.model.optimizer.lr.numpy())

    def train(self,
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              epochs: int = 100,
              steps_per_epoch: Optional[int] = None,
              validation_steps: Optional[int] = None,
              resume_from_checkpoint: Optional[str] = None):
        """
        Main training function with all required functionalities
        """

        # Load model and setup training resumption
        self.model, initial_epoch = self.load_model_and_weights(resume_from_checkpoint)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = train_dataset.cardinality().numpy()
            if steps_per_epoch == tf.data.UNKNOWN_CARDINALITY:
                steps_per_epoch = 620

        if validation_steps is None:

            validation_steps = val_dataset.cardinality().numpy()
            if validation_steps == tf.data.UNKNOWN_CARDINALITY:
                validation_steps = 150

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_lr),
            loss={
                'add_12': self.custom_loss,  # Real loss for final output
                'conv2d_43': dummy_loss,  # Dummy loss
                'activation': dummy_loss  # Dummy loss
            },
            loss_weights={
                'add_12': 1.0,
                'conv2d_43': 0.0,  # Zero weight so it doesn't affect training
                'activation': 0.0
            }
        )

        print(f"Starting training for {epochs} epochs")
        print(f"Initial epoch: {initial_epoch}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")

        # Create callbacks
        # callbacks = self.create_callbacks(steps_per_epoch)

        # Custom training loop to handle history tracking
        class CustomTrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self, trainer):
                self.trainer = trainer

            def on_epoch_end(self, epoch, logs=None):
                self.trainer.update_history(logs)

        # callbacks.append(CustomTrainingCallback(self))

        # Start training
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            # steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,

            # validation_steps=validation_steps,
            # callbacks=callbacks,
            verbose=1
        )

        return history


# Usage example
if __name__ == "__main__":
    # Initialize trainer
    trainer = DCTrainer(
        input_shape=(256, 256, 3),
        base_lr=1e-4,
        lr_min=1e-6,
        warmup_epochs=3,
        checkpoint_dir="training_checkpoints",
        logs_dir="training_logs"
    )

    # Train the model
    hist = trainer.train(
        train_dataset=dataset_generation.get_train_dataset(),
        val_dataset=dataset_generation.get_val_dataset(),
        epochs=100,
        resume_from_checkpoint=None  # Set to path if resuming
    )