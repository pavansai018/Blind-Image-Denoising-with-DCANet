import tensorflow as tf

train_clean = r'/home/pavan/Downloads/SUTD/Project/DCANet/dataset/train/clean/'
train_noisy = r'/home/pavan/Downloads/SUTD/Project/DCANet/dataset/train/dirty/'

val_clean = r'/home/pavan/Downloads/SUTD/Project/DCANet/dataset/val/clean/'
val_noisy = r'/home/pavan/Downloads/SUTD/Project/DCANet/dataset/val/dirty/'


def prepare_dataset_for_multi_output(dataset):
    """Convert dataset to provide 3 outputs matching the model"""
    def map_fn(noisy, clean):
        # For 3-output model: return (input, [target, target, target])
        return noisy, (clean, clean, clean)
    return dataset.map(map_fn)

def get_train_dataset() -> tf.data.Dataset:
    train_clean_generator = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_clean,
        labels=None,
        shuffle=False,
        batch_size=32,

    )

    train_noisy_generator = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_noisy,
        labels=None,
        shuffle=False,
        batch_size=32
    )

    train_dataset = tf.data.Dataset.zip((train_noisy_generator, train_clean_generator))
    train_dataset = prepare_dataset_for_multi_output(train_dataset)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset

def get_val_dataset() -> tf.data.Dataset:
    val_clean_generator = tf.keras.preprocessing.image_dataset_from_directory(
        directory=val_clean,
        labels=None,
        shuffle=False,
        batch_size=32,

    )

    val_noisy_generator = tf.keras.preprocessing.image_dataset_from_directory(
        directory=val_noisy,
        labels=None,
        shuffle=False,
        batch_size=32
    )

    val_dataset = tf.data.Dataset.zip((val_noisy_generator, val_clean_generator))
    val_dataset = prepare_dataset_for_multi_output(val_dataset)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    return val_dataset



if __name__ == '__main__':
    get_train_dataset()