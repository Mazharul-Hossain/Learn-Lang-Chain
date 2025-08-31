from pathlib import Path
from nltk import download as nltk_download
from typing import List, Any
from nltk.downloader import Downloader
import logging

class NltkDownload:
    def __init__(self, list_of_resources=None, download_dir=None):
        if list_of_resources is None:
            list_of_resources = [
                'stopwords',
                'punkt',
            ]
        
        if download_dir is None:
            download_dir = Path('./data/nltk/')

        self.download_nltk_data(list_of_resources, download_dir)


    def check_package_exists(self, 
        package_id: Any,
        download_dir: Path,
    ) -> bool:
        downloader = Downloader(download_dir=str(download_dir))
        return downloader.is_installed(package_id)

    def download_nltk_data(self,
        list_of_resources: List[str],
        download_dir: Path,
    ) -> None:
        download_dir.mkdir(parents=True, exist_ok=True)
        downloader = Downloader(download_dir=str(download_dir))
        for resource in list_of_resources:
            if not self.check_package_exists(resource, download_dir):
                logging.debug(f'Downloading {resource} to {download_dir}')
                downloader.download(info_or_id=resource, quiet=True)
            else:
                logging.debug(f'{resource} already exists in {download_dir}')

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


