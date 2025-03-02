"""This module is for cleaning up temporary files."""

import os
import fcntl
import time
from tempfile import TemporaryDirectory
from psutil import Process


class TempFileCleaner:
    """Cleaning up old, unused temp files.

        Note
        ----
        Only in use if scenario.cleanup is True.

        Parameters
        ----------
        logger : logging.Logger
            Initialized logging object.
        temp_dir : str
            Path to the temporary directory.
        age_limit : int
            Number of seconds a file can exist unused before removal.
    """

    def __init__(self, logger, temp_dir=None, age_limit=600):
        self.temp_dir = temp_dir or TemporaryDirectory().name
        self.tracked_files = set()
        self.age_limit = age_limit
        self.logger = logger
        self.logger.info('\n\nCleaner initialized!\n\n')
        self.logger.info(f'\n\nTemporaryDirectory: {self.temp_dir}\n\n')

    def track_file(self, file_path):
        """
        Adds a file to the list.

        Parameters
        ----------
        file_path : str
            Path to the file to look after.
        """
        if os.path.isfile(file_path):
            self.tracked_files.add(file_path)

    def file_in_use(self, file_path):
        """
        Check if the file is locked or in use.

        Parameters
        ----------
        file_path : str
            Path to the file to look after.

        Returns
        -------
        bool
            Whether the file is in use.
        """
        try:
            with open(file_path, 'rb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return False  # File not in use
        except (IOError, OSError):
            return True  # File in use

    def list_open_files(self):
        """
        Get list of open files for the current process.

        Returns
        -------
        list
            List of the open temporary files.
        """
        return [f.path for f in Process().open_files()]

    def clean_up(self):
        """Clean up unused and inactive files."""
        now = time.time()
        open_files = self.list_open_files()

        for file_path in list(self.tracked_files):
            try:
                # Skip if file no longer exists
                if not os.path.exists(file_path):
                    self.tracked_files.remove(file_path)
                    continue

                # Skip if the file is in use or open
                if self.file_in_use(file_path) or file_path in open_files:
                    continue

                # Skip if the file was accessed recently
                last_access_time = os.path.getatime(file_path)
                if now - last_access_time < self.age_limit:
                    continue

                # Clean up the file
                self.logger.info(f'Cleaning up file: {file_path}')
                os.remove(file_path)
                self.tracked_files.remove(file_path)

            except Exception as e:
                self.logger.info(f'Error while cleaning file {file_path}: {e}')
