import tensorflow as tf
import os

class EpochModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        filename = f'dcanet_epoch_{epoch + 1:03d}.h5'
        filepath = os.path.join(self.checkpoint_dir, filename)
        self.model.save(filepath)
        print(f"Saved model to {filepath}")


