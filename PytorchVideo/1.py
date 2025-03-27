import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

class KineticsDataModule(pytorch_lightning.LightningDataModule):

  # Dataset configuration
  _DATA_PATH = <path_to_kinetics_data_dir>
  _CLIP_DURATION = 2  # Duration of sampled clip for each video
  _BATCH_SIZE = 8
  _NUM_WORKERS = 8  # Number of parallel processes fetching data

  def train_dataloader(self):
    """
    Create the Kinetics train partition from the list of video labels
    in {self._DATA_PATH}/train
    """
    train_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._DATA_PATH, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
        decode_audio=False,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )

  def val_dataloader(self):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/val
    """
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._DATA_PATH, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
        decode_audio=False,
    )
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )