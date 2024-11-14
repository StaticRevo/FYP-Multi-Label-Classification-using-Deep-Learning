import os
import torch
from pytorch_lightning.callbacks import Callback

# Custom callback for additional logging
class BigEarthNetSubsetCallback(Callback):
    def __init__(self, checkpoint_path, model_name):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name

    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        checkpoint_path = os.path.join(self.checkpoint_path, f"epoch={epoch}_{self.model_name}.ckpt")
        torch.save(pl_module.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    def on_train_end(self, trainer, pl_module):
        base_path = os.path.join(self.checkpoint_path, f"final_{self.model_name}")
        final_model_path = f"{base_path}.ckpt"
        counter = 1

        # Check if the path exists and increment the counter until a unique path is found
        while os.path.exists(final_model_path):
            final_model_path = f"{base_path}_{counter}.ckpt"
            counter += 1

        torch.save(pl_module.state_dict(), final_model_path)
        print(f"Final model saved at: {final_model_path}")

