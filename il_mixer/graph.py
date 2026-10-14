"""Disconnected molecular graphs: retain every fragment, charge and multiplicity."""
from functools import lru_cache
import torch
from rdkit import Chem


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=20000)
def graph(smiles, require_ions=True):
    mol = Chem.MolFromSmiles(canonicalize(smiles))
    fragments = Chem.GetMolFrags(mol)
    charges = [sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in frag) for frag in fragments]
    if require_ions and (len(fragments) < 2 or not any(c > 0 for c in charges) or not any(c < 0 for c in charges)):
        raise ValueError(f"Expected separate cation and anion fragments: {smiles}")
    if require_ions and sum(charges) != 0:
        raise ValueError(f"Non-neutral formula unit: {smiles}")
    fragment_id = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
    for i, frag in enumerate(fragments):
        fragment_id[list(frag)] = i
    # Include charge, hydrogens and stereochemical tags; no artificial covalent bonds.
    x = torch.tensor([[a.GetAtomicNum(), a.GetFormalCharge() + 8,
                       min(a.GetTotalDegree(), 8), min(a.GetTotalNumHs(), 8),
                       int(a.GetIsAromatic()), int(a.GetChiralTag()), min(a.GetIsotope(), 300)]
                      for a in mol.GetAtoms()], dtype=torch.long)
    if (x[:, 1] < 0).any() or (x[:, 1] > 16).any():
        raise ValueError("Formal charge outside supported [-8, 8] range")
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feature = [b.GetBondTypeAsDouble() / 3, float(b.GetIsAromatic()),
                   float(b.IsInRing()), float(b.GetIsConjugated()), int(b.GetStereo()) / 6]
        edges.extend([(i, j), (j, i)])
        attrs.extend([feature, feature])
    return dict(x=x, edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T,
                edge_attr=torch.tensor(attrs, dtype=torch.float32).reshape(-1, 5), fragment=fragment_id)


def pack(smiles, device="cpu", require_ions=True):
    graphs = [graph(s, require_ions=require_ions) for s in smiles]
    xs, es, attrs, batches, fragments = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g['x']); es.append(g['edge_index'] + offset)
        attrs.append(g['edge_attr']); fragments.append(g['fragment'])
        batches.append(torch.full((len(g['x']),), i, dtype=torch.long))
        offset += len(g['x'])
    return {k: v.to(device) for k, v in dict(x=torch.cat(xs), edge_index=torch.cat(es, 1),
            edge_attr=torch.cat(attrs), batch=torch.cat(batches), fragment=torch.cat(fragments)).items()}
