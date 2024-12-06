#! /usr/bin/env python

import torch
import torch_geometric as tg
from torch_geometric.data import HeteroData

import ROOT
import h5py
from tqdm import tqdm
import argparse
import sys
import os
import numpy as np
import json


def GetEdgeInfo(eta, phi, max_distance):
    pi = ROOT.TMath.Pi()

    phi_diff = abs(phi.unsqueeze(1) - phi.unsqueeze(0))
    mask = phi_diff > pi
    delta_phi = phi_diff - mask * 2 * pi

    eta_diff = eta.unsqueeze(1) - eta.unsqueeze(0)
    distances = torch.sqrt(eta_diff ** 2 + delta_phi ** 2)

    torch.diagonal(distances).fill_(1000.0)
    distances[torch.tril(torch.ones(distances.shape, dtype=torch.bool))] = 1000.0

    mask = distances < max_distance
    indices = mask.nonzero(as_tuple=False).t()
    edge_index_PP = indices.flip(0)  # (target, source)
    edge_attr_PP = distances[mask]

    return edge_index_PP, edge_attr_PP


def GetDzAssociatedPVInfo(particle_vertex_keys, dzAssociatedPV):
    num_particles = len(particle_vertex_keys)
    edge_index_V = torch.stack((particle_vertex_keys, torch.arange(num_particles)), dim=0)
    edge_attr_V = dzAssociatedPV
    return edge_index_V, edge_attr_V


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input', help='The input ROOT file name', type=str)
    parser.add_argument('--maxDR', dest='maxDR', help='Max DR to construct the graph', type=float, default=0.5)
    opt = parser.parse_args()

    file = ROOT.TFile(opt.input)
    tree = file.Get("analyzer1/Trees/Events")

    out_file_name, _ = os.path.splitext(os.path.basename(opt.input))
    hdf5_file = h5py.File(f'./{out_file_name}.h5', "w")

    pu_groups = {}
    pu_nevents = {}

    num_events = tree.GetEntries()
    for e in tqdm(range(num_events)):
        tree.GetEntry(e)

        phi = torch.tensor(tree.Phi, dtype=torch.float32)
        energy = torch.tensor(tree.Energy, dtype=torch.float32)
        p = torch.tensor(tree.P, dtype=torch.float32)
        pt = torch.tensor(tree.Pt, dtype=torch.float32)
        particleDz_ = torch.tensor(tree.dz, dtype=torch.float32)
        particleDxy_ = torch.tensor(tree.dxy, dtype=torch.float32)
        eta = torch.tensor(tree.Eta, dtype=torch.float32)
        Type = torch.tensor(tree.Type, dtype=torch.int32)
        charge = torch.tensor(tree.Charge, dtype=torch.int32)
        particle_vertex_keys = torch.tensor(tree.vertexKey, dtype=torch.int32)
        dzAssociatedPV = torch.tensor(tree.dzAssociatedPV, dtype=torch.float32)

        Particle_node_features = torch.stack((phi, charge, energy, p, pt, particleDz_, particleDxy_, eta, Type), dim=1)
        Vertex_node_features = torch.stack((dzAssociatedPV, particle_vertex_keys), dim=1)

        edge_index_PP, edge_attr_PP = GetEdgeInfo(eta, phi, opt.maxDR)
        edge_index_V, edge_attr_V = GetDzAssociatedPVInfo(particle_vertex_keys, dzAssociatedPV)

        data = HeteroData()
        data['Particles'].x = Particle_node_features
        data['Vertices'].x = Vertex_node_features
        data['Vertices', 'connects_V', 'Particles'].edge_index = edge_index_V
        data['Vertices', 'connects_V', 'Particles'].edge_attr = edge_attr_V
        data['Particles', 'connects_PP', 'Particles'].edge_index = edge_index_PP
        data['Particles', 'connects_PP', 'Particles'].edge_attr = edge_attr_PP

        graph_label = tree.nInt
        data.y = torch.tensor(graph_label, dtype=torch.int64)

        if graph_label not in pu_groups:
            label = f"PU{graph_label}"
            pu_groups[graph_label] = hdf5_file.create_group(label)
            pu_nevents[graph_label] = 0

        group = pu_groups[graph_label]
        event_id = pu_nevents[graph_label]

        current_type = np.dtype([
            ("node_features_Particles", np.dtype(f"({data['Particles'].x.shape[0]},{data['Particles'].x.shape[1]})f")),
            ("node_features_Vertices", np.dtype(f"({data['Vertices'].x.shape[0]},{data['Vertices'].x.shape[1]})f")),
            ("edge_attributes_PP", np.dtype(f"({len(data['Particles', 'connects_PP', 'Particles'].edge_attr)},)f")),
            ("edge_indices_PP", np.dtype(f"(2,{data['Particles', 'connects_PP', 'Particles'].edge_index.shape[1]})i")),
            ("edge_attributes_VP", np.dtype(f"({len(data['Vertices', 'connects_V', 'Particles'].edge_attr)},)f")),
            ("edge_indices_VP", np.dtype(f"(2,{data['Vertices', 'connects_V', 'Particles'].edge_index.shape[1]})i")),
            ("graph_labels", np.dtype("(1,)i"))
        ])

        arr = np.empty(shape=(1,), dtype=current_type)
        arr["node_features_Particles"] = data['Particles'].x.numpy()
        arr["node_features_Vertices"] = data['Vertices'].x.numpy()
        arr["edge_attributes_PP"] = data['Particles', 'connects_PP', 'Particles'].edge_attr.numpy()
        arr["edge_indices_PP"] = data['Particles', 'connects_PP', 'Particles'].edge_index.numpy()
        arr["edge_attributes_VP"] = data['Vertices', 'connects_V', 'Particles'].edge_attr.numpy()
        arr["edge_indices_VP"] = data['Vertices', 'connects_V', 'Particles'].edge_index.numpy()
        arr["graph_labels"] = data.y.numpy()

        group.create_dataset(
            f'E{event_id}',
            shape=(1,),
            dtype=current_type,
            data=arr,
            compression="gzip",
            chunks=True
        )
        pu_nevents[graph_label] += 1

    hdf5_file.close()

    dict_out = {}
    for pu_label, event_count in pu_nevents.items():
        dict_out[pu_label] = {
            out_file_name: event_count
        }

        # Write to file once
        with open(f'./{out_file_name}.json', "w") as outfile:
            json.dump(dict_out, outfile, indent=4)
    
            
if __name__ == "__main__":
    sys.exit(main())
