from pathlib import Path
from typing import List, Any
from nltk.downloader import Downloader
import logging


class NltkDownload:
    """
    A class to handle the downloading of NLTK data resources.
    """

    def __init__(self, list_of_resources=None, download_dir=None):
        """
        Initializes the NltkDownload object with a list of resources and a download directory.

        :param list_of_resources: List of NLTK resource IDs to be downloaded. Defaults to ['stopwords', 'punkt', 'punkt_tab'].
        :param download_dir: Directory where the NLTK data will be stored. Defaults to './data/nltk/'.
        """
        if list_of_resources is None:
            list_of_resources = [
                "stopwords",
                "punkt",
                "punkt_tab"
            ]

        self.download_dir = download_dir
        if download_dir is None:
            self.download_dir = Path("./data/nltk/")

        self.download_nltk_data(list_of_resources, self.download_dir)

    def check_package_exists(
        self,
        package_id: Any,
        download_dir: Path,
    ) -> bool:
        """
        Checks if a specific NLTK package exists in the given directory.

        :param package_id: ID of the NLTK package to check.
        :param download_dir: Directory where the NLTK data is stored.
        :return: True if the package exists, False otherwise.
        """
        downloader = Downloader(download_dir=str(download_dir))
        return downloader.is_installed(package_id)

    def download_nltk_data(
        self,
        list_of_resources: List[str],
        download_dir: Path,
    ) -> None:
        """
        Downloads NLTK data resources if they do not already exist in the specified directory.

        :param list_of_resources: List of NLTK resource IDs to be downloaded.
        :param download_dir: Directory where the NLTK data will be stored.
        """
        download_dir.mkdir(parents=True, exist_ok=True)
        downloader = Downloader(download_dir=str(download_dir))
        for resource in list_of_resources:
            if not self.check_package_exists(resource, download_dir):
                logging.debug(f"Downloading {resource} to {download_dir}")
                downloader.download(info_or_id=resource, quiet=True)
            else:
                logging.debug(f"{resource} already exists in {download_dir}")


# try:
#     # Check if 'stopwords' is already downloaded
#     nltk.data.find('corpora/stopwords')
#     print("Stopwords are already downloaded.")
# except LookupError:
#     # Download 'stopwords' if not found
#     print("Stopwords not found. Downloading stopwords now...")
#     # Download NLTK stopwords once
#     nltk.download("stopwords")

# try:
#     nltk.data.find('tokenizers/punkt')
#     print("Tokenizers are already downloaded.")
# except LookupError:
#     print("Tokenizers not found. Downloading punkt now...")
#     nltk.download("punkt")
