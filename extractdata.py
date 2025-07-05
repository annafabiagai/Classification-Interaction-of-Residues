import os
import argparse
import subprocess
import urllib.request
import pandas as pd
from joblib import load

import numpy as np

from arrays import columns, full_dummies, newOrder, ordered

from mappings import mapping


def download_mmCIF(pdb_id: str, output_dir: str) -> str:
    """
    Download the mmCIF file given a PDB ID to save it locally.

    Args:
        pdb_id (str): The PDB identifier of the structure.
        output_dir (str): The direcotry where the output file will be saved.

    Returns:
        str: The full path to the downloaded mmCIF.
    """

    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    out_path = os.path.join(output_dir, f"{pdb_id}.cif")

    # If the file does not already exist, download it!
    if not os.path.exists(out_path):
        print(f"Downloading {pdb_id}.cif ...")
        urllib.request.urlretrieve(url, out_path)
        print("Download complete.")
    else:
        print(f"{pdb_id}.cif already exists.")
    return out_path


def run_3di_extraction(cif_path: str, out_dir: str, script_path: str) -> None:
    """
    Run the 3Di extractor on a given mmCIF.

    Args:
        cif_path (str): Path to the input mmCIF file.
        out_dir (str): Directory where extraction results will be written.
        script_path (str): Path to the 3Di extraction script.
    """

    command = ["python", script_path, cif_path, "-out_dir", out_dir]
    print("Running 3di extraction...")
    subprocess.run(command, check=True)
    print("3Di extraction done.")


def run_features_extraction(
    cif_path: str, out_dir: str, script_path: str, config_path: str = None
) -> None:
    """
    Run the feature extraction script on a given mmCIF file, optionally using a config file.

    Args:
        cif_path (str): Path to the input mmCIF file.
        out_dir (str): Directory where extraction results will be written.
        script_path (str): Path to the feature extraction script.
        config_path (Optional[str]): Path to the config file.
    """

    command = ["python", script_path, cif_path, "-out_dir", out_dir]

    if config_path:
        command.extend(["-conf_file", config_path])

    print("Running features extraction...")
    subprocess.run(command, check=True)
    print("Features extraction done.")


def main():
    """
    Run the full extraction and prediction pipeline for a given PDB ID.
    """

    # region ArgParsing
    parser = argparse.ArgumentParser(
        description="Run 3di and features extraction from PDB ID."
    )
    parser.add_argument("pdb_id", help="PDB ID (e.g., 1ABC)")
    parser.add_argument("-out_dir", help="Output directory", default="outputs")
    parser.add_argument(
        "-script", help="Path to 3di extraction script", default="calc_3di.py"
    )
    parser.add_argument(
        "-config",
        help="Path to configuration.json for features extraction",
        default=None,
    )
    parser.add_argument(
        "-features_script",
        help="Path to features extraction script",
        default="calc_features.py",
    )
    args = parser.parse_args()
    # endregion

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: Download mmCIF
    cif_path = download_mmCIF(args.pdb_id, args.out_dir)

    # Step 2: Run 3Di extraction script
    run_3di_extraction(cif_path, args.out_dir, args.script)

    # Step 3: Run features extraction
    run_features_extraction(cif_path, args.out_dir, args.features_script, args.config)

    # region Step 4: Merge the two output files together and build calculated features
    df_3di = pd.read_csv(f"{args.out_dir}/{args.pdb_id}_3di.tsv", sep="\t")
    df_features = pd.read_csv(f"{args.out_dir}/{args.pdb_id}_features.tsv", sep="\t")

    df_features = df_features.merge(
        df_3di.rename(
            columns={
                "ch": "s_ch",
                "resi": "s_resi",
                "ins": "s_ins",
                "resn": "s_resn",
                "3di_state": "s_3di_state",
            }
        )[["pdb_id", "s_ch", "s_resi", "s_ins", "s_resn", "s_3di_state"]],
        on=["pdb_id", "s_ch", "s_resi", "s_ins", "s_resn"],
        how="left",
    )

    df_features = df_features.merge(
        df_3di.rename(
            columns={
                "ch": "t_ch",
                "resi": "t_resi",
                "ins": "t_ins",
                "resn": "t_resn",
                "3di_state": "t_3di_state",
            }
        )[["pdb_id", "t_ch", "t_resi", "t_ins", "t_resn", "t_3di_state"]],
        on=["pdb_id", "t_ch", "t_resi", "t_ins", "t_resn"],
        how="left",
    )

    # df_features["c_dist"] = (df_features["s_resi"] - df_features["t_resi"]).abs()
    df_features["c_dist"] = np.where(
        df_features["s_ch"] == df_features["t_ch"],
        (df_features["s_resi"] - df_features["t_resi"]).abs(),
        -1,
    )
    df_features["c_is_same_chain"] = df_features["s_ch"] == df_features["t_ch"]

    cols_to_drop = ["s_up", "s_down", "s_ss3", "t_up", "t_down", "t_ss3"]
    df_final = df_features.drop(columns=cols_to_drop)

    df_final = df_final[ordered]

    os.remove(f"outputs/{args.pdb_id}_3di.tsv")
    os.remove(f"outputs/{args.pdb_id}_features.tsv")
    os.remove(f"outputs/{args.pdb_id}.cif")

    # Save merged file
    df_final.to_csv(f"{args.out_dir}/{args.pdb_id}.tsv", sep="\t", index=False)
    print("Saved features and 3di states.")

    # endregion

    # region Load models
    ml_model = load("models/multi_label_weights.joblib")
    mc_model = load("models/multi_class_weights.joblib")
    scaler = load("models/scaler.pkl")
    # endregion

    # region Data Preprocessing
    X = df_final[columns]
    cat_cols = [c for c, t in X.dtypes.items() if t == "object" or t.name == "category"]
    X = pd.get_dummies(X, columns=cat_cols, prefix=cat_cols, drop_first=False)

    df_dropped = df_final[~df_final.index.isin(df_final.dropna().index)].copy()
    df_dropped.loc[:, "Interaction"] = "??"
    df_dropped["Interactions"] = "??"

    df_final = df_final.dropna()

    for c in full_dummies:
        if c not in X.columns:
            X[c] = False

    X = X.dropna()
    X = X[newOrder]
    X = scaler.transform(X)
    print("Table prepared for the prediction.")
    # endregion

    # region Running prediction:
    # MultiClass
    df_final["Interaction"] = mc_model.predict(X)
    df_final["Score"] = np.array([np.max(row) for row in mc_model.predict_proba(X)])

    # MultiLabel
    predictions = ml_model.predict(X)
    df_final["Interactions"] = [
        ", ".join(mapping[i] for i, val in enumerate(row) if val == 1.0)
        for row in predictions
    ]
    # endregion

    # Joining unpredicted values
    df_final = pd.concat([df_final, df_dropped]).sort_index()
    df_final["Interaction"] = df_final["Interaction"].map(mapping)

    # Export results
    df_final.to_csv(
        f"{args.out_dir}/{args.pdb_id}_predicted.tsv", sep="\t", index=False
    )

    print("Prediction completed and esported.")


if __name__ == "__main__":
    main()
