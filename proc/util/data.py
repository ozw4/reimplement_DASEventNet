"""Data loading and dataset splitting utilities."""

from pathlib import Path

import numpy as np


def load_event_noise(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load event and noise arrays from ``data_dir``.

        Parameters
        ----------
        data_dir : Path
                Directory containing ``extract_event_2s.npy`` and
                ``extract_noise_2s.npy``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
                Loaded ``(event_data, noise_data)`` arrays with newest events first.

        """
        event = np.load(data_dir / 'extract_event_2s.npy')
        noise = np.load(data_dir / 'extract_noise_2s.npy')

        # Reverse so that the newest events come first
        event = event[::-1]
        noise = noise[::-1]
        return event, noise


def split_data(
        event: np.ndarray,
        noise: np.ndarray,
        test_frac: float = 0.1,
        val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split event and noise arrays into train/val/test sets."""
        if len(event) != len(noise):
                msg = 'event and noise arrays must have the same length'
                raise ValueError(msg)

        n_total = len(event)
        n_test = int(n_total * test_frac)
        n_val = int(n_total * val_frac)

        test_range = (0, n_test)
        val_range = (n_test, n_test + n_val)
        train_range = (n_test + n_val, n_total)

        x_train = np.vstack(
                (
                        event[train_range[0] : train_range[1]],
                        noise[train_range[0] : train_range[1]],
                )
        )
        x_val = np.vstack(
                (
                        event[val_range[0] : val_range[1]],
                        noise[val_range[0] : val_range[1]],
                )
        )
        x_test = np.vstack(
                (
                        event[test_range[0] : test_range[1]],
                        noise[test_range[0] : test_range[1]],
                )
        )

        n_train = train_range[1] - train_range[0]
        n_val_len = val_range[1] - val_range[0]
        n_test_len = test_range[1] - test_range[0]

        y_train = np.concatenate((np.ones(n_train), np.zeros(n_train)))
        y_val = np.concatenate((np.ones(n_val_len), np.zeros(n_val_len)))
        y_test = np.concatenate((np.ones(n_test_len), np.zeros(n_test_len)))

        return x_train, y_train, x_val, y_val, x_test, y_test

