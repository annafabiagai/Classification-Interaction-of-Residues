import numpy as np
import logging
from pathlib import Path, PurePath
import argparse
import os
import pandas as pd

import torch
from Bio.SeqUtils import seq1

from Bio.PDB import FastMMCIFParser, is_aa
import foldseek_extract_pdb_features


# 50 letters (X/x are missing)
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz'

# WARNING change accordingly depending on the execution path
model_dir = '3di_model'


def encoder_features(residues, virt_cb=(270, 0, 2), full_backbone=True):
    """
    Calculate 3D descriptors for each residue of a PDB file.
    """
    coords, valid_mask = foldseek_extract_pdb_features.get_atom_coordinates(residues, full_backbone=full_backbone)

    coords = foldseek_extract_pdb_features.move_CB(coords, virt_cb=virt_cb)

    partner_idx = foldseek_extract_pdb_features.find_nearest_residues(coords, valid_mask)
    features, valid_mask2 = foldseek_extract_pdb_features.calc_angles_forloop(coords, partner_idx, valid_mask)

    seq_dist = (partner_idx - np.arange(len(partner_idx)))[:, np.newaxis]
    log_dist = np.sign(seq_dist) * np.log(np.abs(seq_dist) + 1)

    vae_features = np.hstack([features, log_dist])

    return vae_features, valid_mask2

def discretize(centroids, x):
    return np.argmin(foldseek_extract_pdb_features.distance_matrix(x, centroids), axis=1)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_file', help='mmCIF or PDB file')
    parser.add_argument('-out_dir', help='Output directory', default='.')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    # Set the logger
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Load the config file
    # If not provided, set the path to "configuration.json", which is in the same folder of this Python file
    src_dir = str(PurePath(os.path.realpath(__file__)).parent)

    # Start
    pdb_id = Path(args.pdb_file).stem
    logging.info("{} processing".format(pdb_id))

    # Load model parameters
    encoder = torch.load('{}/encoder.pt'.format(model_dir), weights_only=False)
    centroids = np.loadtxt('{}/states.txt'.format(model_dir))
    encoder.eval()

    pdb_id = Path(args.pdb_file).stem
    logging.info("{} processing".format(pdb_id))

    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure('None', args.pdb_file)

    data = []
    for chain in structure[0]:
        residues = list(chain.get_residues())

        feat, mask = encoder_features(residues)
        res_features = feat[mask]

        with torch.no_grad():
            res = encoder(torch.tensor(res_features, dtype=torch.float32)).detach().numpy()

        valid_states = discretize(centroids, res)

        states = np.full(len(mask), -1)
        states[mask] = valid_states

        for i, state in enumerate(states):
            if state != -1:
                data.append((pdb_id, chain.id, *residues[i].id[1:], seq1(residues[i].get_resname()), state, LETTERS[state],
                      *feat[i]))

    # Create a DataFrame and save to file
    df = pd.DataFrame(data, columns=['pdb_id', 'ch', 'resi', 'ins', 'resn', '3di_state', '3di_letter',
                                     'cos_phi_12', 'cos_phi_34', 'cos_phi_15', 'cos_phi_35',
                                     'cos_phi_14', 'cos_phi_23', 'cos_phi_13', 'd', 'seq_dist', 'log_dist'])

    df.to_csv("{}/{}_3di.tsv".format(args.out_dir, pdb_id), sep="\t", index=False)