import tensorflow as tf
import math


class WarmupThenCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    EXACT REPLICATION of:
      GradualWarmupScheduler(multiplier=1, total_epoch=warmup_epochs)
      followed by CosineAnnealingLR(T_max = num_epochs - warmup_epochs + 40)

    Warmup is epoch-based.
    Cosine schedule is epoch-based.
    """
    def __init__(self,
                 base_lr,
                 lr_min,
                 num_epochs,
                 steps_per_epoch,
                 warmup_epochs=3,
                 extra_epochs=40):
        super().__init__()
        self.base_lr = float(base_lr)
        self.lr_min = float(lr_min)
        self.num_epochs = int(num_epochs)
        self.steps_per_epoch = float(steps_per_epoch)
        self.warmup_epochs = float(warmup_epochs)
        self.extra_epochs = float(extra_epochs)

        # EXACT PyTorch: T_max = (NUM_EPOCHS - warmup_epochs + 40)
        self.T_max = float(self.num_epochs - self.warmup_epochs + self.extra_epochs)

        # Convert to steps
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.total_steps = self.num_epochs * self.steps_per_epoch
        self.T_max_steps = self.T_max * self.steps_per_epoch

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # --------------------------
        # 1. Warmup schedule
        # --------------------------
        def warmup_phase():
            progress = step / self.warmup_steps
            progress = tf.clip_by_value(progress, 0.0, 1.0)
            return self.base_lr * progress  # multiplier = 1

        # --------------------------
        # 2. Cosine annealing
        # --------------------------
        def cosine_phase():
            t = step - self.warmup_steps
            t = tf.clip_by_value(t, 0.0, self.T_max_steps)

            cosine = 0.5 * (1 + tf.cos(math.pi * t / self.T_max_steps))
            return self.lr_min + (self.base_lr - self.lr_min) * cosine

        # Decide warmup or cosine
        return tf.cond(step < self.warmup_steps, warmup_phase, cosine_phase)


    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "lr_min": self.lr_min,
            "num_epochs": self.num_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "warmup_epochs": self.warmup_epochs,
            "extra_epochs": self.extra_epochs
        }
