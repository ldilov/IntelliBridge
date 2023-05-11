import multiprocessing
import os
import tarfile
import requests
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

from tqdm import tqdm


class CorpusDownloader:
    def __init__(self, download_links, download_directory=None):
        self.download_links = download_links
        if download_directory is None:
            download_directory = Path.home() / ".cache"

        self.download_directory = download_directory
        os.makedirs(self.download_directory, exist_ok=True)

    def download_and_extract(self, url):
        filename = self.download_directory / os.path.basename(url)
        print(f"Downloading {url} to {filename}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()

        print(f"Downloaded {filename}. Extracting...")

        if tarfile.is_tarfile(filename):
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(self.download_directory)

        print(f"Extraction complete for {filename}")

    def download_corpora(self):
        n_workers = multiprocessing.cpu_count * 3
        thread_prefix = "[corpus/download]"

        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix=thread_prefix) as executor:
            executor.map(self.download_and_extract, self.download_links)


if __name__ == "__main__":
    enron_mail_link = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
    downloader = CorpusDownloader([enron_mail_link])
    downloader.download_corpora()
