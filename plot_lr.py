import matplotlib.pyplot as plt
import numpy as np
from scheduler import WarmupThenCosine
import tensorflow as tf

from dataset_generation import get_train_dataset, get_val_dataset
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device=device, enable=True)


steps_per_epoch = get_train_dataset().cardinality().numpy()
print(steps_per_epoch)
# Create the scheduler with your actual parameters
lr_schedule = WarmupThenCosine(
    base_lr=1e-4,
    lr_min=1e-6,
    num_epochs=100,  # Your total epochs
    steps_per_epoch=steps_per_epoch,  # Use your actual steps_per_epoch here
    warmup_epochs=3,
    extra_epochs=40
)

# Generate for all epochs (not dependent on dataset)
total_steps = 100 * steps_per_epoch  # epochs * steps_per_epoch
steps = np.arange(total_steps)
learning_rates = [lr_schedule(step).numpy() for step in steps]

# Convert steps to epochs
epochs = steps / steps_per_epoch  # Divide by your actual steps_per_epoch

# Plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, learning_rates)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()