
# Copied from Midi.
# Do not move these imports, the order seems to matter
from rdkit import Chem
import torch
import pytorch_lightning as pl
import torch_geometric

import hydra
import omegaconf
import numpy as np
import sys, os
sys.path.append(os.path.abspath('.'))
from qm9.rdkit_functions import build_molecule, build_xae_molecule
sys.path.append("eval_src/midi_metrics")

from midi.datasets import qm9_dataset, geom_dataset
from midi.utils import setup_wandb
from midi.analysis.rdkit_functions import Molecule
from midi.metrics.molecular_metrics import SamplingMetrics
from midi.analysis.baselines_evaluation import atom_decoder_dict, atom_encoder_dict
from pathlib import Path
import subprocess
import tempfile
import logging

logger = logging.getLogger(name=__name__)

# def calculate_charges(pos, atom_types, dataset_info_dict):
#     # Build X, A, E using the given build_xae_molecule function
#     X, A, E = build_xae_molecule(pos, atom_types, dataset_info_dict)
    
#     mol = Chem.RWMol()
#     for atom in X:
#         a = Chem.Atom(atom_decoder[atom.item()])
#         mol.AddAtom(a)

#     all_bonds = torch.nonzero(A)

#     for i, bond in enumerate(all_bonds):
#         if bond[0].item() != bond[1].item():
#             mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
#             # add formal charge to atom: e.g. [O+], [N+], [S+]
#             # not support [O-], [N-], [S-], [NH+] etc.
#             flag, atomid_valence = check_valency(mol)
#             if flag:
#                 continue
#             else:
#                 assert len(atomid_valence) == 2
#                 idx = atomid_valence[0]
#                 v = atomid_valence[1]
#                 an = mol.GetAtomWithIdx(idx).GetAtomicNum()
#                 if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
#                     mol.GetAtomWithIdx(idx).SetFormalCharge(1)
#                     # print("Formal charge added")
    
#     # Sanitize the molecule (important to assign bond orders and charge corrections)
#     Chem.SanitizeMol(mol)
    
#     # Compute the formal charges
#     formal_charges = []
#     for atom in mol.GetAtoms():
#         formal_charges.append(atom.GetFormalCharge())

#     return formal_charges

def write_xyz_file(atom_numbers, positions, temp_dir, filename):
    """
    Writes a single molecule (atom_numbers and positions) to an XYZ file.
    
    :param atom_numbers: Tensor of atom numbers for a single molecule (N,)
    :param positions: Tensor of positions for a single molecule (N, 3)
    :param temp_dir: Directory where the file will be saved
    :param filename: Name of the XYZ file
    """
    num_atoms = len(atom_numbers)
    xyz_file_path = os.path.join(temp_dir, filename)

    with open(xyz_file_path, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write("Generated by Open Babel\n")
        for atom_number, pos in zip(atom_numbers, positions):
            f.write(f"{atom_number} {pos[0]} {pos[1]} {pos[2]}\n")

    return xyz_file_path

def convert_xyz_to_sdf(xyz_file, sdf_file):
    """
    Uses subprocess to call Open Babel (obabel) to convert XYZ file to SDF format.
    
    :param xyz_file: Path to the XYZ file
    :param sdf_file: Path to the output SDF file
    """
    subprocess.run(['obabel', xyz_file, '-O', sdf_file, '--addtotitle', 'end'], check=True)

def open_babel_preprocess(file, name ):
    """
    :param file: str - File path
    :param name: str - 'qm9_with_h', 'qm9_no_h', 'geom_with_h', 'geom_no_h'
    :param atom_encoder_dict: dict - Encoding of atoms by name
    :param atom_decoder_dict: dict - Decoding map for atoms
    :return: List of Molecule objects
    """
    # Fetch the atom encoders and decoders
    atom_encoder = atom_encoder_dict.get(name)
    atom_decoder = atom_decoder_dict.get(name)

    if atom_encoder is None or atom_decoder is None:
        raise ValueError(f"Atom encoder/decoder for {name} not found.")

    with open(file, "r") as f:
        lines = f.readlines()[3:]  # Skipping the first 3 header lines

    result = []
    temp = []

    # Process the file line by line
    for line in lines:
        line = line.strip()

        # Use the standard SDF delimiter $$$$ to mark the end of a molecule
        if line == "$$$$":
            if temp:
                result.append(temp)  # Save the molecule data
            temp = []  # Reset temp for the next molecule
        elif "M" in line or "$" in line or "OpenBabel" in line:
            continue  # Skip metadata lines
        else:
            vec = line.split()
            temp.append(vec)

    all_mols = []

    # Process each molecule in the 'result' array
    for array in result:
        if len(array) == 0:
            continue  # Skip if empty molecule block

        atom_temp = []
        pos_temp = []

        num_atoms = int(array[0][0])  # The first entry in each molecule block is the number of atoms
        for i in range(num_atoms):
            element = array[i + 1][3]  # Atom element type (from 4th position)
            x = atom_encoder.get(element, None)
            if x is None:
                # Handle elements not in the map
                print(f'Element {element} is not handled in the current mapping')
                continue
            atom_temp.append(x)

            # Atom coordinates
            x_pos = float(array[i + 1][0])
            y_pos = float(array[i + 1][1])
            z_pos = float(array[i + 1][2])
            pos_temp.append([x_pos, y_pos, z_pos])

        # Process bond information, starting after atom data
        bonds_start_idx = num_atoms + 1
        bond_matrix_size = num_atoms
        bond_matrix = [[0] * bond_matrix_size for _ in range(bond_matrix_size)]

        for j in range(bonds_start_idx, len(array)):
            bond_data = array[j]
            atom_a = int(bond_data[0]) - 1  # Atom indices (1-based in file, 0-based in code)
            atom_b = int(bond_data[1]) - 1
            bond_type = int(bond_data[2])  # Bond type

            bond_matrix[atom_a][atom_b] = bond_type
            bond_matrix[atom_b][atom_a] = bond_type

        # Convert collected data to tensors
        X = torch.tensor(atom_temp)
        charges = torch.zeros(X.shape)  # No charge info, default to zeros
        E = torch.tensor(bond_matrix)
        posis = torch.tensor(pos_temp)

        # Create a Molecule object and store it
        molecule = Molecule(
            atom_types=X, bond_types=E, positions=posis, charges=charges, atom_decoder=atom_decoder
        )
        molecule.build_molecule(atom_decoder=atom_decoder)
        all_mols.append(molecule)

    return all_mols
from tqdm import tqdm
def process_edm_molecule_lists_with_openbabel(atom_number_list, pos_list, mol_num_limit):
    try:
        result = subprocess.run(['obabel', '-h'], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("obabel is not working correctly.")
    except FileNotFoundError:
        print("Error: obabel is not installed or not available in the system PATH.")
        return
    
    all_molecules = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (atom_numbers, positions) in tqdm(enumerate(zip(atom_number_list, pos_list)), total=len(atom_number_list)):
            if mol_num_limit is not None:
                if i > mol_num_limit:
                    break
            # Write each molecule to an XYZ file
            xyz_filename = f"molecule_{i}.xyz"
            xyz_file_path = write_xyz_file(atom_numbers, positions, temp_dir, xyz_filename)

            # Define the output SDF file path
            sdf_filename = f"molecule_{i}.sdf"
            sdf_file_path = os.path.join(temp_dir, sdf_filename)

            # Convert XYZ to SDF using Open Babel
            convert_xyz_to_sdf(xyz_file_path, sdf_file_path)

            # Read the generated SDF file with pybel (optional)
            mol = open_babel_preprocess(sdf_file_path, "qm9_with_h")

            # Append the molecule to the result list
            all_molecules.extend(mol)
            
    
        return all_molecules  # or process it as required

def edmMolecules2MiDiMolecules(edm_molecule_dict, dataset_infos, use_openbabel=True, mol_num_limit=None):
    mol_num = edm_molecule_dict['x'].shape[0]
    atom_decoder = dataset_infos.atom_decoder
    dataset_info_dict = {
        "atom_decoder": dataset_infos.atom_decoder,
        "name": dataset_infos.name
    }
    assert dataset_infos.name == "qm9", "Only implemented for qm9 now"
    atomtype2atomnumber = {
        0: 1,
        1: 6,
        2: 7,
        3: 8,
        4: 9,
    }
    
    if use_openbabel:
        atom_number_list = []
        pos_list = []
    else:
        midi_molecule_list = []
    for i in range(mol_num):
        if mol_num_limit is not None:
            assert type(mol_num_limit) is int, "Please specify an int number for molecule limit."
            if i > mol_num_limit:
                break
        
        mol_x = edm_molecule_dict['x'][i].cpu()
        mol_node_mask = edm_molecule_dict['node_mask'][i].to(torch.bool).squeeze().cpu()
        mol_one_hot = edm_molecule_dict['one_hot'][i].cpu()
        
        # preprocess
        atom_types = torch.argmax(mol_one_hot[mol_node_mask], dim=-1) # This is encoded type
        pos = mol_x[mol_node_mask]
        
        
        # Now, choose from two ways to get bonds
        if not use_openbabel:
            X, A, E = build_xae_molecule(pos, atom_types, dataset_info_dict)
            # X: N         (int)
            # A: N x N     (bool)                  (binary adjacency matrix)
            # E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
            # TODO: Calculate charge of each atom using rdkit
            charges = torch.zeros(atom_types.shape, dtype=torch.long, device=atom_types.device)
            assert torch.all(torch.triu(E, diagonal=1) == 0), "Matrix E is not lower triangular"
            E = E + E.T - torch.diag(E.diag())
            midi_molecule = Molecule(
                atom_types=X,
                bond_types=E.to(torch.long),
                charges=charges, #TODO: calculate charges using rdkit.
                positions=pos,
                atom_decoder=dataset_infos.atom_decoder
            )
            
            midi_molecule_list.append(midi_molecule)
        else:
            atom_numbers =  np.vectorize(lambda x: atomtype2atomnumber[x])(atom_types.cpu().numpy())
            atom_numbers = torch.tensor(atom_numbers, dtype=torch.long)
            atom_number_list.append(atom_numbers)
            pos_list.append(pos)
            
            
            
    if use_openbabel:
        # Given: atom_number_list (contains all atom_numbers, type is tensor, shape is (N)) and pos_list (contains all pos, type is tensor, shape is (N, 3)) 
        midi_molecule_list = process_edm_molecule_lists_with_openbabel(atom_number_list, pos_list, mol_num_limit)
            
    return midi_molecule_list
        
        
        
        
def main_midi(cfg, edm_molecules):
    assert isinstance(edm_molecules, dict) and set(edm_molecules.keys()) == set(["x", "node_mask", "one_hot"]), "edm molecules should have format in dict and with keys [x, node_mask, one_hot]."
    
    
    dataset_config = cfg.dataset
    pl.seed_everything(cfg.train.seed)

    assert cfg.train.batch_size == 1

    if dataset_config.name in ['qm9', "geom"]:
        if dataset_config.name == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

        else:
            datamodule = geom_dataset.GeomDataModule(cfg)
            dataset_infos = geom_dataset.GeomInfos(datamodule=datamodule, cfg=cfg)

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    train_smiles = list(datamodule.train_dataloader().dataset.smiles)
    sampling_metrics = SamplingMetrics(train_smiles=train_smiles, dataset_infos=dataset_infos, test=True)
    
    
    midi_molecules = edmMolecules2MiDiMolecules(edm_molecules, dataset_infos, use_openbabel=cfg.use_openbabel, mol_num_limit=cfg.mol_num_limit)
    
    to_log = sampling_metrics(molecules=midi_molecules, name='train_set', current_epoch=-1, local_rank=0)
    return to_log
        

