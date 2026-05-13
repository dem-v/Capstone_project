from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


class SyntheticLesionDataset(Dataset):
    """Synthetic binary classification data with maskable circular lesions.

    This is only for pipeline verification before real medical data is available.
    Positive images contain a bright circular lesion; negative images contain noise only.
    """

    def __init__(
        self,
        n_samples: int = 256,
        image_size: int = 96,
        positive_fraction: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.image_size = image_size
        self.positive_fraction = positive_fraction
        self.generator = torch.Generator().manual_seed(seed)
        self.images, self.labels, self.masks = self._generate()

    def _generate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = torch.randn((self.n_samples, 1, self.image_size,
                             self.image_size), generator=self.generator) * 0.12
        labels = torch.zeros((self.n_samples,), dtype=torch.long)
        masks = torch.zeros((self.n_samples, self.image_size,
                            self.image_size), dtype=torch.bool)

        yy, xx = torch.meshgrid(
            torch.arange(self.image_size),
            torch.arange(self.image_size),
            indexing="ij",
        )

        n_positive = int(self.n_samples * self.positive_fraction)
        positive_indices = torch.randperm(
            self.n_samples, generator=self.generator)[:n_positive]

        for idx in positive_indices:
            radius = int(torch.randint(
                6, 13, (1,), generator=self.generator).item())
            margin = radius + 5
            cx = int(torch.randint(margin, self.image_size -
                     margin, (1,), generator=self.generator).item())
            cy = int(torch.randint(margin, self.image_size -
                     margin, (1,), generator=self.generator).item())
            lesion = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
            labels[idx] = 1
            masks[idx] = lesion
            images[idx, 0][lesion] += 1.15

            # Add a faint surrounding gradient so localization is not perfectly trivial.
            distance = torch.sqrt((xx - cx).float() **
                                  2 + (yy - cy).float() ** 2)
            halo = torch.exp(-distance / max(radius, 1)) * 0.15
            images[idx, 0] += halo

        # Add low-amplitude anatomy-like background bands.
        y_float = torch.arange(self.image_size).float() / self.image_size
        band = 0.08 * torch.sin(2 * math.pi *
                                y_float).view(1, 1, self.image_size, 1)
        images += band
        images = images.clamp(-1.0, 1.5)
        images = (images - images.min()) / (images.max() - images.min())
        return images, labels, masks

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index], self.masks[index]
