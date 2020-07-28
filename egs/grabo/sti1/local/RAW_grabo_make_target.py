from pathlib import Path
from bs4 import BeautifulSoup, Tag
import pandas as pd
from operator import itemgetter
import os


ANNOT_DIR = Path("/users/spraak/qmeeus/data/grabo/speakers")
OUTPUT_FILE = Path("data/grabo/target.csv")

os.makedirs(OUTPUT_FILE.parent, exist_ok=True)


def read_text(filename):
    with open(filename) as txtfile:
        return txtfile.read().strip()

def xml2wav(filename):
    return Path(str(filename).replace("framedir", "spchdatadir").replace(".xml", ".wav"))

def xml2txt(filename):
    return Path(str(filename).replace("framedir", "transcriptions").replace(".xml", ".txt"))

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
    xml_files = list(ANNOT_DIR.glob("**/framedir/recording*/*.xml"))
    data = list(map(extract, xml_files))
    dataset = pd.DataFrame()
    dataset["filename"] = xml_files
    dataset["audiofile"] = dataset["filename"].map(xml2wav)
    assert dataset["audiofile"].map(Path.exists).all()
    dataset["action"] = list(map(itemgetter("action"), data))
    dataset["attributes"] = data
    dataset["attributes"].map(lambda d: d.pop("action"))
    dataset = (
        dataset.join(pd.concat(
            dataset["attributes"].map(lambda d: pd.DataFrame.from_dict(d, orient="index").T).tolist(),
            sort=False
        ).set_index(dataset.index)).drop("attributes", axis=1)
    )

    dataset["action_string"] = (dataset.iloc[:, 2:]
                                .apply(lambda row: "_".join(filter(pd.notnull, row)), axis=1))

    dataset["text"] = dataset["filename"].map(xml2txt).map(read_text)

    dataset.to_csv("data/grabo/target.csv", index=False)


if __name__ == '__main__':
    main()
