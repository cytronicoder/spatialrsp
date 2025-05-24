import os
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# global session with retry + pooling
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
session.mount("https://", adapter)
session.mount("http://", adapter)

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _download_file(
    url: str,
    dest_path: str,
    verbose: bool = False,
    desc: str = "Downloading",
) -> str:
    """
    Download a file from a URL to a specified destination path.

    Args:
        url (str): The URL to download the file from.
        dest_path (str): The local path where the file will be saved.
        verbose (bool): If True, prints download progress and status.
        desc (str): Description for the tqdm progress bar.

    Returns:
        str: The path to the downloaded file.

    Raises:
        requests.HTTPError: If the download fails.
        FileNotFoundError: If the destination directory does not exist.
    """
    Path(dest_path).parent.mkdir(exist_ok=True, parents=True)
    if os.path.exists(dest_path):
        if verbose:
            print(f"[✓] Exists: {dest_path}")
        return dest_path

    with session.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total,
            unit="iB",
            unit_scale=True,
            desc=desc,
            leave=False,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    if verbose:
        print(f"[✓] Downloaded: {dest_path}")
    return dest_path


def download_hcl(verbose: bool = False) -> str:
    """
    Download the Human Cell Landscape (HCL) dataset.
    This dataset contains single-cell RNA-seq data from various human tissues.

    Args:
        verbose (bool): If True, prints download progress and status.

    Returns:
        str: The path to the downloaded HCL dataset file.
    """
    return _download_file(
        "https://datasets.cellxgene.cziscience.com/ae0c62a1-a30f-4033-97d2-0edb2e146c53.h5ad",
        "data/hcl.h5ad",
        verbose,
        "HCL",
    )


def download_kpmp(data_type: str = "sn", verbose: bool = False) -> str:
    """
    Download the Kidney Precision Medicine Project (KPMP) dataset.
    This dataset contains spatial transcriptomics data from kidney biopsies.

    Args:
        data_type (str): Type of data to download, either 'sn' for single-nucleus
                         or 'sc' for single-cell.
        verbose (bool): If True, prints download progress and status.

    Returns:
        str: The path to the downloaded KPMP dataset file.

    Raises:
        ValueError: If an invalid data_type is provided.
    """
    urls = {
        "sn": "https://datasets.cellxgene.cziscience.com/7d8af09a-2f96-49f9-a473-f561a332f25d.h5ad",
        "sc": "https://datasets.cellxgene.cziscience.com/f5b6d620-76df-45c5-9524-e5631be0e44a.h5ad",
    }
    if data_type not in urls:
        raise ValueError("Invalid data_type: must be 'sn' or 'sc'.")
    return _download_file(
        urls[data_type],
        f"data/kpmp_{data_type}.h5ad",
        verbose,
        f"KPMP-{data_type}",
    )


def download_mca(verbose: bool = False) -> List[str]:
    """
    Download the Mouse Cell Atlas (MCA) dataset.
    This dataset contains single-cell RNA-seq data from mouse tissues.

    Args:
        verbose (bool): If True, prints download progress and status.

    Returns:
        List[str]: A list containing the paths to the downloaded MCA files.
    """
    files = [
        ("mca.h5ad", "https://figshare.com/ndownloader/files/37560595?…"),
        ("mca_cell_info.csv", "https://figshare.com/ndownloader/files/36222822?…"),
    ]
    # parallel download
    with ThreadPoolExecutor(max_workers=len(files)) as exe:
        futures = {
            exe.submit(
                _download_file,
                url,
                os.path.join("data", fn),
                verbose,
                fn,
            ): fn
            for fn, url in files
        }
        return [f.result() for f in as_completed(futures)]
