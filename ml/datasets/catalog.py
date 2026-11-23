"""
The catalogue of public datasets: which are integrated, where they come from, and which are not
and why. The reasons matter as much as the loaders: a dataset that cannot support the behavioural
features is not integrated just because it is popular.

Downloading is deliberately NOT done here. The Kaggle datasets need the user's own login, and
unofficial mirrors cannot be verified as unmodified. `find_files` tells you exactly what to fetch
and where to put it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ml.datasets.loaders import load_ieee, load_paysim, load_sparkov, load_tabformer
from ml.datasets.schema import DatasetError


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    source: str                      # where to get it
    download: str                    # a command the user runs themselves
    primary: tuple                   # glob patterns for THE data file: the first pattern that matches is used
    loader: Callable                 # (paths, max_customers, seed) -> (raw, notes)
    optional: tuple = ()             # extra files that are loaded if present (for example a test split)
    notes: str = ""
    size: str = ""

    @property
    def files(self) -> tuple:
        return self.primary + self.optional


DATASETS = {
    "sparkov": Dataset(
        name="sparkov", title="Sparkov simulated credit-card transactions (2019 to 2020)",
        source="https://www.kaggle.com/datasets/kartik2112/fraud-detection",
        download="kaggle datasets download -d kartik2112/fraud-detection -p data/raw/sparkov --unzip",
        primary=("fraudTrain.csv",), optional=("fraudTest.csv",), size="about 500 MB unzipped, 1.85 million rows",
        loader=lambda paths, k, seed: load_sparkov(paths, k, seed),
        notes="Best fit: about 1,000 cards with long histories and a real fraud label. No device information."),
    "tabformer": Dataset(
        name="tabformer", title="IBM TabFormer synthetic credit-card transactions",
        source="https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions",
        download="kaggle datasets download -d ealtman2019/credit-card-transactions -p data/raw/tabformer --unzip",
        primary=("credit_card_transactions-ibm_v*.csv", "card_transaction.v1.csv"),
        size="about 2.3 GB, 24 million rows (use --max-customers)",
        loader=lambda paths, k, seed: load_tabformer(paths[0], k, seed),
        notes="History is deep (2,000 users, years of data) but, measured on the real file, fraud is only weakly visible "
              "to this pipeline's 15 features (ROC-AUC 0.69 to 0.76): fraud rows are 61% online against 11% of "
              "normal rows, and payment channel and merchant category are not features. Very large, so sample customers. "
              "The download also holds a one-user sample and two files with names, addresses and card numbers; "
              "only the transactions file is ever read."),
    "paysim": Dataset(
        name="paysim", title="PaySim synthetic mobile-money transfers",
        source="https://www.kaggle.com/datasets/ealaxi/paysim1",
        download="kaggle datasets download -d ealaxi/paysim1 -p data/raw/paysim --unzip",
        primary=("PS_*_log.csv", "paysim*.csv"), size="about 470 MB, 6.3 million rows",
        loader=lambda paths, k, seed: load_paysim(paths[0], k),
        notes="Poor fit: nearly every origin account appears once, so there is no customer history. "
              "Loadable for completeness; prepare refuses it unless --force."),
    "ieee": Dataset(
        name="ieee", title="IEEE-CIS Fraud Detection (Vesta e-commerce chargebacks)",
        source="https://www.kaggle.com/c/ieee-fraud-detection",
        download="kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee   # accept the rules on the site first",
        primary=("train_transaction.csv",), optional=("train_identity.csv",), size="about 600 MB, 590,000 rows",
        loader=lambda paths, k, seed: load_ieee(paths[0].parent),
        notes="Real chargeback data. Customer identity is a card proxy, so several features are degraded (see the IEEE adapter)."),
}

NOT_INTEGRATED = {
    "baf": ("Bank Account Fraud (NeurIPS 2022) is account-opening fraud: each row is one application with no "
            "customer identity, timestamp or payment amount, only a month number. MerchantShield's features are "
            "built from a customer's earlier transactions, so there is nothing to compute. It could serve as a "
            "drift benchmark (it is built with month-to-month shift) but not as transaction training data."),
    "ulb": ("The ULB credit-card fraud set (2013) has only a time offset, an amount and 28 anonymised PCA "
            "components, with no customer or card identifier, so per-customer history features cannot be built. "
            "A score on it says little about how this system behaves."),
}


def find_files(name: str, path) -> list:
    """The files for `name` under `path` (a folder or a single file), or a DatasetError saying how to get them."""
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        why = f" ({NOT_INTEGRATED[name]})" if name in NOT_INTEGRATED else ""
        raise DatasetError(f"Unknown dataset {name!r}. Integrated datasets: {known}.{why}")
    ds, base = DATASETS[name], Path(path)
    if base.is_file():
        return [base]
    primary = next((sorted(base.glob(pattern)) for pattern in ds.primary if sorted(base.glob(pattern))), None)
    if not primary:
        raise DatasetError(
            f"{' or '.join(ds.primary)} not found in {base}. Download it yourself (it needs your Kaggle login, and I do "
            f"not use unofficial mirrors):\n  {ds.download}\nSource: {ds.source}")
    extras = [p for pattern in ds.optional for p in sorted(base.glob(pattern))]
    return list(dict.fromkeys(primary[:1] + extras if name in ("tabformer", "paysim") else primary + extras))


def main():
    print("Integrated datasets (you download them; this project never fetches them):\n")
    for d in DATASETS.values():
        print(f"{d.name:10s} {d.title}\n{'':10s} {d.notes}\n{'':10s} size: {d.size}\n{'':10s} get it: {d.download}\n")
    print("Not integrated, and why:\n")
    for k, why in NOT_INTEGRATED.items():
        print(f"{k:10s} {why}\n")


if __name__ == "__main__":
    main()
