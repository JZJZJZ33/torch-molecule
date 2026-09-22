import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from torch_molecule.visualization import (
    PolymerConformerGenerator,
    expand_psmiles,
    generate_uff_conformer,
)


def test_expand_psmiles_builds_ten_unit_chain_with_capped_ends():
    smiles = expand_psmiles("[*:1]CC[*:2]")
    molecule = Chem.MolFromSmiles(smiles)

    assert molecule.GetNumAtoms() == 20
    assert all(atom.GetAtomicNum() != 0 for atom in molecule.GetAtoms())
    assert Chem.MolToSmiles(molecule) == "C" * 20


def test_expand_psmiles_respects_repeat_units():
    molecule = Chem.MolFromSmiles(expand_psmiles("*CO*", repeat_units=3))

    assert molecule.GetNumAtoms() == 6
    assert sum(atom.GetSymbol() == "C" for atom in molecule.GetAtoms()) == 3
    assert sum(atom.GetSymbol() == "O" for atom in molecule.GetAtoms()) == 3


@pytest.mark.parametrize("psmiles", ["CC", "*CC", "*C(*)*"])
def test_expand_psmiles_rejects_invalid_attachment_count(psmiles):
    with pytest.raises(ValueError, match="exactly two"):
        expand_psmiles(psmiles)


def test_generate_uff_conformer_returns_3d_mol_block():
    result = generate_uff_conformer("*CC*", repeat_units=2, random_seed=7)

    assert result.repeat_units == 2
    assert result.molecule.GetNumConformers() == 1
    assert result.molecule.GetConformer().Is3D()
    assert "V2000" in result.mol_block or "V3000" in result.mol_block
    assert isinstance(result.energy, float)


def test_generator_defaults_to_ten_repeat_units():
    generator = PolymerConformerGenerator()
    molecule = Chem.MolFromSmiles(generator.expand("*C*"))

    assert molecule.GetNumAtoms() == 10
