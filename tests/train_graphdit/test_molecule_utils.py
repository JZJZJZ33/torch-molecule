import torch
from rdkit import Chem

from train_graphdit.llamole_graphdit import molecule_utils


def test_connect_fragments_updates_unsanitized_property_cache():
    molecule = Chem.MolFromSmiles("CC.CC", sanitize=False)

    connected = molecule_utils.connect_fragments(molecule)
    smiles = molecule_utils.mol2smiles(connected)

    assert smiles is not None
    assert "." not in smiles


def test_graph_to_smiles_rejects_invalid_rdkit_fallback(monkeypatch):
    def fail_correction(*_args, **_kwargs):
        raise RuntimeError("forced correction failure")

    monkeypatch.setattr(molecule_utils, "correct_mol", fail_correction)
    atom_types = torch.zeros(6, dtype=torch.long)
    edge_types = torch.zeros((6, 6), dtype=torch.long)
    edge_types[0, 1:] = 1
    edge_types[1:, 0] = 1

    assert molecule_utils.graph_to_smiles(
        [(atom_types, edge_types)], ["C"]
    ) == [None]
