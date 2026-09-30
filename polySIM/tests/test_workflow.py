"""Fast direct checks for the modular polymer-building example."""

from collections import Counter

import numpy as np
from rdkit.Chem import rdMolDescriptors
from chain_builder import build_chain
from conf_builder import build_mmff_conformer
from system_builder import pack_chains
from xtb_calc import _energy_from_output


def test_unmapped_pvdf_repeat_count_and_hydrogen_termination() -> None:
    expected = {1: "C2H4F2", 2: "C4H6F4", 10: "C20H22F20"}
    for repeat_units, formula in expected.items():
        chain = build_chain("[*]CC(F)(F)[*]", repeat_units)
        assert rdMolDescriptors.CalcMolFormula(chain) == formula
        assert chain.GetNumAtoms() == 4 * repeat_units
        assert all(atom.GetAtomicNum() != 0 for atom in chain.GetAtoms())


def test_mismatched_attachment_bonds_require_explicit_policy() -> None:
    psmiles = "*OC(=O)CCCCC(=O)Nc1ccc(CCc2ccc(C=C=*)cc2)cc1"
    try:
        build_chain(psmiles, 2)
    except ValueError as error:
        assert "same bond type" in str(error)
    else:
        raise AssertionError("strict policy accepted mismatched attachment bonds")
    chain = build_chain(psmiles, 2, mismatched_bond_policy="single")
    assert chain.GetNumAtoms() > 0


def test_wildcard_stereochemistry_is_cleaned_before_embedding() -> None:
    psmiles = "*/C=C(/*)c1cc(Br)ccc1OC(=O)c1ccc(OCC(C)CC)cc1"
    chain = build_chain(psmiles, 2)
    result = build_mmff_conformer(chain, random_seed=42, max_iterations=1000)
    assert result.converged


def test_small_pvdf_workflow() -> None:
    chain = build_chain("[*]CC(F)(F)[*]", 2)
    assert rdMolDescriptors.CalcMolFormula(chain) == "C4H6F4"

    conformer = build_mmff_conformer(chain, random_seed=7, max_iterations=1000)
    symbols = [atom.GetSymbol() for atom in conformer.molecule.GetAtoms()]
    coordinates = conformer.molecule.GetConformer().GetPositions()
    assert Counter(symbols) == Counter(C=4, H=6, F=4)

    packed = pack_chains(
        symbols,
        coordinates,
        chains=3,
        density_g_cm3=0.1,
        minimum_distance=2.0,
        random_seed=7,
    )
    assert len(packed.symbols) == 42
    # Intrachain distances can be below 2 A, so only compare different blocks.
    atoms = packed.atoms_per_chain
    for left in range(3):
        for right in range(left + 1, 3):
            delta = packed.coordinates[left * atoms : (left + 1) * atoms, None, :] - packed.coordinates[
                None, right * atoms : (right + 1) * atoms, :
            ]
            delta -= packed.box_length_angstrom * np.round(
                delta / packed.box_length_angstrom
            )
            assert np.linalg.norm(delta, axis=2).min() >= 2.0


def test_xtb_text_energy_fallback_uses_final_energy() -> None:
    output = """
    :: total energy             -44.000000000000 Eh    ::
    | TOTAL ENERGY             -44.125000000000 Eh   |
    """
    assert _energy_from_output(output) == -44.125
