from pathlib import Path
from bs4 import BeautifulSoup, Tag
import pandas as pd
from operator import itemgetter
import os


ANNOT_DIR = Path("/users/spraak/qmeeus/data/patience/data/oracle_action_frames")
OUTPUT_FILE = Path("data/target.csv")

os.makedirs(OUTPUT_FILE.parent, exist_ok=True)


def read_text(filename):
    with open(filename) as txtfile:
        return txtfile.read().strip()

def xml2uttid(filename):
    filename = Path(filename)
    speaker = filename.parents[1].name
    uttid = filename.name.replace("_oracleframe.xml", "")
    return f"{speaker}_{uttid}"

def extract(filename):
    with open(filename) as xmlf:
        soup = BeautifulSoup(xmlf, "html.parser")
    action = soup.find("thisframe").get_text().strip()
    attributes = {
        child.name: child.get_text().strip()
        for child in soup.find("data").children
        if isinstance(child, Tag)
    }

    return dict(action=action, **attributes)


def main():
    xml_files = list(ANNOT_DIR.glob("**/*.pjt/*_oracleframe.xml"))
    data = list(map(extract, xml_files))
    dataset = pd.DataFrame()
    dataset["uttid"] = list(map(xml2uttid, xml_files))
    dataset["filename"] = xml_files
    dataset["action"] = list(map(itemgetter("action"), data))
    dataset["attributes"] = data
    dataset["attributes"].map(lambda d: d.pop("action"))
    attributes = pd.concat(
        dataset["attributes"].map(lambda d: pd.DataFrame.from_dict(d, orient="index").T).tolist(),
        sort=False
    )
    
    dataset = dataset.assign(**{col: "" for col in attributes.columns})
    dataset.loc[dataset["attributes"] != {}, attributes.columns] = attributes.values
    dataset = dataset.drop("attributes", axis=1)
    dataset.to_csv(OUTPUT_FILE, index=False)


if __name__ == '__main__':
    main()
