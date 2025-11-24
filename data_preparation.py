# This script downloads and parses all the files necessary to run the project. Run this script on first setup
import json
import os
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

### parameters
FORM_990_EXTRACT_PATH = "./data/form_990/"
FORM_990_ZIP_PATH = "./load/form_990/"
FORM_990_PROCESSED_PATH = "./processed/form_990/"
FORM_990_BASE_URL = "https://apps.irs.gov/pub/epostcard/990/xml"

CMS_YEAR = 2022
CMS_DATASET_ID = f"hospitals_10_2022"
CMS_URL = f"https://data.cms.gov/provider-data/sites/default/files/archive/Hospitals/{CMS_YEAR}/{CMS_DATASET_ID}.zip"
CMS_ZIP_PATH = f"./load/hospitals/"
CMS_EXTRACT_PATH = f"data/hospitals/"


COM_INSIGHT_URL = "https://www.communitybenefitinsight.org/api/get_hospitals.php"
COST_REPORT_PATH = "./data/cost_report/"


def generate_form_990_urls():
    """Generates a list of URLs to download to get form 990 data"""
    form_990_urls = []
    # WARNING: This will take 10s of GB

    form_990_urls.append(
        os.path.join(FORM_990_BASE_URL, "2022", "2022_TEOS_XML_01A.zip")
    )
    for i in range(1, 13):
        if i < 10:
            idx_name = f"0{i}"
        else:
            idx_name = str(i)
        form_990_urls.append(
            os.path.join(FORM_990_BASE_URL, "2023", f"2023_TEOS_XML_{idx_name}A.zip")
        )
        form_990_urls.append(
            os.path.join(FORM_990_BASE_URL, "2024", f"2024_TEOS_XML_{idx_name}A.zip")
        )
    return form_990_urls


def download(url, filename, chunk_size=8192):
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as f, tqdm(
        total=total, unit_scale=True, unit_divisor=chunk_size
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)


def download_990s(bulk_urls, zip_path, extract_path):
    """Download and extract form 990s from the irs"""
    if not os.path.exists(zip_path):
        os.makedirs(zip_path)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    for url in bulk_urls:
        try:
            year = url.split("/")[7]
            index = url.split("/")[8].split("_")[3][:2]
            download_path = os.path.join(zip_path, year)
            save_path = os.path.join(extract_path, year)
            if not os.path.exists(download_path):
                os.mkdir(download_path)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            download_path = os.path.join(download_path, f"{index}.zip")
            save_path = os.path.join(save_path, f"{index}")

            if not os.path.exists(save_path):
                print(f"downloading {url}")
                download(url, download_path)

                print("extracting...")
                with zipfile.ZipFile(download_path, "r") as z:
                    for member in tqdm(z.namelist()):
                        z.extract(member, save_path)

            else:
                print(f"{url} already saved at {save_path}. Skipping...")

        except Exception as e:
            print(f"ERROR: {e}; url = {url}")


def get_cms_data(cms_url, zip_path, extract_path, cms_dataset_id, force=False):

    if not os.path.exists(zip_path):
        os.makedirs(zip_path)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    download_dest = os.path.join(zip_path, cms_dataset_id + ".zip")
    if not os.path.exists(download_dest) or force:
        print(f"downloading {cms_url} to {download_dest}...")
        download(cms_url, download_dest)
    else:
        print(f"{download_dest} already exists. Skipping download")

    if not os.path.exists(extract_path) or force:
        print(f"unzipping {download_dest} to {extract_path}...")
        with zipfile.ZipFile(download_dest, "r") as z:
            for member in tqdm(z.namelist()):
                z.extract(member, extract_path)
    else:
        print(f"{extract_path} already exists. Skipping Extraction")

    return os.path.join(extract_path, cms_dataset_id)


def download_bridge_file():
    if os.path.exists("./data/bridge.csv"):
        return pd.read_csv("./data/bridge.csv")

    download(
        "https://www.communitybenefitinsight.org/api/get_hospitals.php",
        "./data/bridge.json",
    )

    with open("./data/bridge.json", "r") as f:
        data = json.load(f)

    bridge = pd.DataFrame(data)
    bridge.to_csv("./data/bridge.csv")
    os.remove("./data/bridge.json")


def download_cost_report(cost_report_path):
    if not os.path.exists(cost_report_path):
        os.makedirs(cost_report_path)

    if os.path.exists(os.path.join(cost_report_path, "cost_report_proc.csv")):
        print("Cost report already downloaded")
        return

    chunk_size = 1000
    offset = 0
    num_records = chunk_size
    index = 0
    max_iter = 100
    while num_records == chunk_size:
        chunk_path = f"{cost_report_path}/{index}.json"

        if not os.path.exists(chunk_path):
            download(
                f"https://data.cms.gov/data-api/v1/dataset/44060663-47d8-4ced-a115-b53b4c270acb/data?size={chunk_size}&offset={offset}",
                chunk_path,
            )

        with open(chunk_path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df.to_csv(f"{cost_report_path}/{index}.csv")

        num_records = len(df)
        index += 1
        offset += chunk_size

        if index == max_iter:
            raise RuntimeError("max iter reached")

    dfs = []
    for f in os.listdir(cost_report_path):
        if ".csv" in f:
            dfs.append(pd.read_csv(os.path.join(cost_report_path, f)))

    df_proc = pd.concat(dfs)
    df_proc.to_csv(os.path.join(cost_report_path, "cost_report_proc.csv"))


def main():
    form_990_urls = generate_form_990_urls()

    print("Downloading form 990s...")
    download_990s(form_990_urls, FORM_990_ZIP_PATH, FORM_990_EXTRACT_PATH)

    print("Downloading cms data...")
    get_cms_data(CMS_URL, CMS_ZIP_PATH, CMS_EXTRACT_PATH, CMS_DATASET_ID)

    print("Downloading bridge file...")
    download_bridge_file()

    print("Downloading cost report...")
    download_cost_report(COST_REPORT_PATH)


if __name__ == "__main__":
    main()
