import numpy as np
import xml.etree.ElementTree as ET
import zipfile
import requests
import os
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Optional
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from xml.etree.ElementTree import iterparse
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy import stats
import json


IRS_NS = {"irs": "http://www.irs.gov/efile"}

def download(url, filename, chunk_size=8192):
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as f, tqdm(total=total, unit_scale=True, unit_divisor=chunk_size) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)


def _text_to_int(x: Optional[str]) -> int:
    """Convert numeric text to int; treat None/'' as 0."""
    if not x:
        return 0
    return int(re.sub(r"[^\d\-]", "", x))

    
def list_xml_files(dir_path: str):
    # Faster than os.listdir + joins; also filter *_public.xml
    with os.scandir(dir_path) as it:
        for e in it:
            if e.is_file() and e.name.endswith("_public.xml"):
                yield e.path


def download_990s(bulk_urls, zip_path, extract_path):
    if not os.path.exists(zip_path):
        os.makedirs(zip_path)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
        
    for url in bulk_urls:
        try:
            year = url.split('/')[7]
            index = url.split('/')[8].split('_')[3][:2]
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
                with zipfile.ZipFile(download_path, 'r') as z:
                    for member in tqdm(z.namelist()):
                        z.extract(member, save_path)

            else:
                print(f"{url} already saved at {save_path}. Skipping...")
                        
        except Exception as e:
            print(f"ERROR: {e}; url = {url}")

                
def get_cms_data(cms_url, 
                 zip_path, 
                 extract_path, 
                 cms_dataset_id,
                 force = False):

    if not os.path.exists(zip_path):
        os.makedirs(zip_path)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    download_dest = os.path.join(zip_path, cms_dataset_id + ".zip")
    if not os.path.exists(download_dest) or force:
        print(f"downloading {cms_url} to {download_dest}...")
        download(
            cms_url,
            download_dest
        )
    else:
        print(f"{download_dest} already exists. Skipping download")

    # if not os.path.exists(extract_path) or force:
    print(f"unzipping {download_dest} to {extract_path}...")
    with zipfile.ZipFile(download_dest, "r") as z:
        for member in tqdm(z.namelist()):
            z.extract(member, extract_path)

    # else:
    #     print(f"{extract_path} already exists. Skipping Extraction")

    return os.path.join(extract_path, cms_dataset_id)

def get_bridge_file():
    if os.path.exists("bridge.csv"):
        return pd.read_csv("bridge.csv")
        
    download(
        'https://www.communitybenefitinsight.org/api/get_hospitals.php',
        'test.json'
    )
    
    with open('test.json', 'r') as f:
        data = json.load(f)
    
    bridge = pd.DataFrame(data)
    bridge.to_csv("bridge.csv")
    return bridge
    

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
        chunk_path = f'{cost_report_path}/{index}.json'

        if not os.path.exists(chunk_path):
            download(
                f'https://data.cms.gov/data-api/v1/dataset/44060663-47d8-4ced-a115-b53b4c270acb/data?size={chunk_size}&offset={offset}',
                chunk_path
            )

        with open(chunk_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df.to_csv(f'{cost_report_path}/{index}.csv')

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


    
def strip_namespaces(root):
    """Remove namespace URIs from all tags in an ElementTree root."""
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    return root
    
def fast_parse_top_level(xml_path: str, wanted) -> dict:
    
    wanted_paths = {tuple(k.split('/')): v for k, v in wanted.items()}
    found = {v: None for v in wanted.values()}
    remaining = set(wanted_paths.keys())
    stack = []

    for event, elem in iterparse(xml_path, events=("start", "end")):
        tag = elem.tag.split("}", 1)[1] if "}" in elem.tag else elem.tag

        if event == "start":
            stack.append(tag)
        else:  # end
            # suffix-match: ... Return › ReturnHeader › Filer › EIN  endswith  ReturnHeader › Filer › EIN
            for wanted_path in list(remaining):
                if len(stack) >= len(wanted_path) and tuple(stack[-len(wanted_path):]) == wanted_path:
                    key = wanted_paths[wanted_path]
                    text = (elem.text or "").strip()
                    found[key] = text if text != "" else None
                    remaining.remove(wanted_path)
            stack.pop()
            elem.clear()
            if not remaining:
                break

    # Normalize types
    rev = found.get("revenue")
    if rev:
        try:
            found["revenue"] = int(rev.replace(",", ""))
        except ValueError:
            found["revenue"] = 0
    else:
        found["revenue"] = 0
    return found


def parse_schedule_j_person_rows(xml_path: str) -> List[Dict]:
    """
    Return a list of dict rows for Schedule J people. Each row contains
    name, title, and all comp components for filing org and related orgs.
    Handles both OfficerTrstKeyEmplGrp and RltdOrgOfficerTrstKeyEmplGrp.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []

    # Two possible groups in Schedule J:
    groups = []
    groups += root.findall(".//irs:IRS990ScheduleJ/irs:OfficerTrstKeyEmplGrp", IRS_NS)
    groups += root.findall(".//irs:IRS990ScheduleJ/irs:RltdOrgOfficerTrstKeyEmplGrp", IRS_NS)

    for g in groups:
        get = lambda tag: g.find(f"irs:{tag}", IRS_NS)

        row = {
            "PersonNm":           (get("PersonNm").text if get("PersonNm") is not None else None),
            "TitleTxt":           (get("TitleTxt").text if get("TitleTxt") is not None else None),

            # Base
            "BaseComp_Filer":     _text_to_int(get("BaseCompensationFilingOrgAmt").text if get("BaseCompensationFilingOrgAmt") is not None else None),
            "BaseComp_Related":   _text_to_int(get("CompensationBasedOnRltdOrgsAmt").text if get("CompensationBasedOnRltdOrgsAmt") is not None else None),

            # Bonus (covers “bonus & incentive” in IRS parlance)
            "Bonus_Filer":        _text_to_int(get("BonusFilingOrganizationAmount").text if get("BonusFilingOrganizationAmount") is not None else None),
            "Bonus_Related":      _text_to_int(get("BonusRelatedOrganizationsAmt").text if get("BonusRelatedOrganizationsAmt") is not None else None),

            # Other comp
            "Other_Filer":        _text_to_int(get("OtherCompensationFilingOrgAmt").text if get("OtherCompensationFilingOrgAmt") is not None else None),
            "Other_Related":      _text_to_int(get("OtherCompensationRltdOrgsAmt").text if get("OtherCompensationRltdOrgsAmt") is not None else None),

            # Deferred
            "Deferred_Filer":     _text_to_int(get("DeferredCompensationFlngOrgAmt").text if get("DeferredCompensationFlngOrgAmt") is not None else None),
            "Deferred_Related":   _text_to_int(get("DeferredCompRltdOrgsAmt").text if get("DeferredCompRltdOrgsAmt") is not None else None),

            # Nontaxable benefits
            "Nontax_Filer":       _text_to_int(get("NontaxableBenefitsFilingOrgAmt").text if get("NontaxableBenefitsFilingOrgAmt") is not None else None),
            "Nontax_Related":     _text_to_int(get("NontaxableBenefitsRltdOrgsAmt").text if get("NontaxableBenefitsRltdOrgsAmt") is not None else None),

            # Totals (preferred if present)
            "TotalComp_Filer":    _text_to_int(get("TotalCompensationFilingOrgAmt").text if get("TotalCompensationFilingOrgAmt") is not None else None),
            "TotalComp_Related":  _text_to_int(get("TotalCompensationRltdOrgsAmt").text if get("TotalCompensationRltdOrgsAmt") is not None else None),
        }

        # Derived fields
        row["Bonus_Total"] = row["Bonus_Filer"] + row["Bonus_Related"]

        # Prefer IRS reported totals; if missing, fall back to summing components
        total_reported = row["TotalComp_Filer"] + row["TotalComp_Related"]
        if total_reported > 0:
            row["TotalComp_AllIn"] = total_reported
        else:
            row["TotalComp_AllIn"] = (
                row["BaseComp_Filer"] + row["BaseComp_Related"] +
                row["Bonus_Filer"] + row["Bonus_Related"] +
                row["Other_Filer"] + row["Other_Related"] +
                row["Deferred_Filer"] + row["Deferred_Related"] +
                row["Nontax_Filer"] + row["Nontax_Related"]
            )

        row["IncentivePct"] = (
            row["Bonus_Total"] / row["TotalComp_AllIn"]
            if row["TotalComp_AllIn"] > 0 else None
        )

        rows.append(row)

    return rows

def parse_xml(path, wanted):

    try:
        row = fast_parse_top_level(path, wanted)
        comp_data = parse_schedule_j_person_rows(path)
    
        for c in comp_data:
            for r in row:
                c[r] = row[r]
    
        return comp_data

    except Exception as e:
        msg = f"Error: {e}; path = {path}"
        print(msg)
        # with open("log.txt", "a") as f:
        #     f.write(msg)

        return []


def parse_xml_dir(dir_path,
                  save_path,
                  wanted,
                  workers=os.cpu_count(),
                  limit = None):
    
    files = list(list_xml_files(dir_path))
    rows = []
    print(f"running parse all with {workers} workers")

    if workers == 1:
        for i, f in enumerate(tqdm(files)):
            data = parse_xml(f, wanted)
            if len(data) > 0:
                rows += data
                
            if limit is not None and i >= limit: 
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(parse_xml, f, wanted): f for f in files}
            for fut in tqdm(as_completed(futures), total=len(files)):
                rows += fut.result()
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df    

def parse_all_dirs(src_dir, dest_dir, wanted):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for year in os.listdir(src_dir):
        for idx in os.listdir(os.path.join(src_dir, year)):
            for folder in os.listdir(os.path.join(src_dir, year, idx)):
                path = os.path.join(src_dir, year, idx, folder)

                save_path = os.path.join(dest_dir, 
                                         f"{year}_{idx}_{folder}_processed.csv")
                msg_id = f"{year} {idx} {folder}"
                if not os.path.exists(save_path):
                    try:
                        parse_xml_dir(dir_path = path, save_path = save_path, wanted = wanted)
                        msg = f"Successfully parsed {msg_id}"
                        print(msg)
                    except Exception as e:
                        msg = f"ERROR: {e}; {msg_id}"
                        print(msg)
                else:
                    print(f"{save_path} already exists. Skipping...")

def collect_dfs(processed_dir, save_dir):
    fs = os.listdir(processed_dir)
    dfs = []
    for f in fs:
        if ".csv" not in f: 
            continue
        dfs.append(pd.read_csv(os.path.join(processed_dir, f)))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(save_dir)


    
def clean_990(df_990, year, min_rev, max_rev):
    for col in ['revenue', 'expenses', 'TotalComp_AllIn', 'Bonus_Total']:
        if col in df_990.columns:
            df_990[col] = pd.to_numeric(df_990[col], errors='coerce').fillna(0).astype(float)
    df_990 = df_990[df_990['tax_year'] == year]
    df_990 = df_990.groupby([
        'name',
        'ein',
        'tax_year',
        'state',
        'city'
    ], dropna=False).sum(numeric_only = True).reset_index()[[
        'name',
        'ein',
        'tax_year',
        'state',
        'city',
        'revenue',
        'expenses',
        'TotalComp_AllIn',
        'Bonus_Total'
    ]]

    df_990['incentive_perc'] = df_990.apply(lambda r: r['TotalComp_AllIn'] / r['Bonus_Total'] if r['Bonus_Total'] > 0 else 0, axis = 1)
    df_990['op_margin'] = df_990.apply(lambda r: (r['revenue'] - r['expenses']) / r['revenue'] if r['revenue'] > 0 else 0, axis = 1)
    df_990 = df_990[(df_990['revenue'] >= min_rev) & (df_990['revenue'] <= max_rev)]
    return df_990

def get_hcahps(cms_root_path):
    hcahps = pd.read_csv(os.path.join(cms_root_path, 'HCAHPS-Hospital.csv'))
    hcahps = hcahps.pivot_table(
        index = [
            "Facility ID",
            "Facility Name",
            "City",
            "State",
            "ZIP Code"
        ],
        columns = [
            "HCAHPS Question"
        ],
        values = [
            "HCAHPS Answer Percent",
            "Patient Survey Star Rating",
            "HCAHPS Linear Mean Value"
        ],
        aggfunc = "first"
    ).reset_index()
    hcahps.columns = [
        f"{measure}: {val}" if len(measure) > 0 else val
        for val, measure in hcahps.columns
    ]

    hcahps = hcahps.drop(
        columns = [c for c in hcahps.columns if (hcahps[c] == 'Not Applicable').all()]
    )

    hcahps = hcahps.replace('Not Available', np.nan)
    for c in hcahps.columns:
        if ':' in c:
            hcahps[c] = hcahps[c].astype(float)
            
    return hcahps

def get_comp_mort(cms_root_path):
    comp_mort = pd.read_csv(os.path.join(cms_root_path, 'Complications_and_Deaths-Hospital.csv'))
    comp_mort = comp_mort.pivot_table(
        index = [
            "Facility ID",
            "Facility Name",
            "City",
            "State",
            "ZIP Code"
        ],
        columns = [
            "Measure Name"
        ],
        values = [
            "Denominator",
            "Score",
            "Lower Estimate",
            "Higher Estimate"
        ],
        aggfunc = "first"
    ).reset_index()
    comp_mort.columns = [
        f"{measure}: {val}" if len(measure) > 0 else val
        for val, measure in comp_mort.columns
    ]

    comp_mort = comp_mort.drop(
        columns = [c for c in comp_mort.columns if (comp_mort[c] == 'Not Applicable').all() or (comp_mort[c] == 'Not Available').all()]
    )

    comp_mort = comp_mort.replace('Not Available', np.nan)


    for c in comp_mort.columns:
        if ':' in c:
            comp_mort[c] = comp_mort[c].astype(float)
    return comp_mort

def get_hosp_gen_info(cms_root_path):
    hosp_info = pd.read_csv(os.path.join(cms_root_path, "Hospital_General_Information.csv"))
    return hosp_info

def get_hvbp_clinical_outcomes(cms_root_path):
    hvbp_clin_out = pd.read_csv(os.path.join(cms_root_path, "hvbp_clinical_outcomes.csv"))
    return hvbp_clin_out 
    
    
def get_cost_report(cost_report_path):
    return pd.read_csv(os.path.join(cost_report_path, "cost_report_proc.csv"))

def get_bridge_file():
    if os.path.exists("bridge.csv"):
        return pd.read_csv("bridge.csv")
        
    download(
        'https://www.communitybenefitinsight.org/api/get_hospitals.php',
        'test.json'
    )
    
    with open('test.json', 'r') as f:
        data = json.load(f)
    
    bridge = pd.DataFrame(data)
    bridge.to_csv("bridge.csv")
    return bridge

def get_healthcare_system():
    return pd.read_csv("./data/health_system.csv", encoding="cp1252")
