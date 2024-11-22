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


def GetEdgeInfo(eta, phi, max_distance):
    pi = ROOT.TMath.Pi()
    
    # Compute phi differences
    phi_diff = abs(phi.unsqueeze(1) - phi.unsqueeze(0))
    mask = phi_diff > pi
    delta_phi = phi_diff - mask * 2 * pi
    
    # Compute eta differences
    eta_diff = eta.unsqueeze(1) - eta.unsqueeze(0)
    
    # Calculate distances
    distances = torch.sqrt(eta_diff ** 2 + delta_phi ** 2)
    
    # Mask self-loops and lower triangle
    torch.diagonal(distances).fill_(1000.0)
    distances[torch.tril(torch.ones(distances.shape, dtype=torch.bool))] = 1000.0
    
    # Create mask for valid edges
    mask = distances < max_distance
    
    # Extract edge indices and attributes
    indices = mask.nonzero(as_tuple=False).t()
    edge_index_PP = indices.flip(0)  # Flip for (target, source) ordering
    edge_attr_PP = distances[mask]
    
    return edge_index_PP, edge_attr_PP

def GetDzAssociatedPVInfo(particle_vertex_keys, dzAssociatedPV):
    num_particles = len(particle_vertex_keys)
    
    # Create edge indices and attributes for vertex association
    edge_index_V = torch.stack((particle_vertex_keys, torch.arange(num_particles)), dim=0)
    edge_attr_V = dzAssociatedPV
    
    return edge_index_V, edge_attr_V



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '--input' , dest='input' , help='the input root file name' , type=str )
    parser.add_argument( '--maxDR' , dest='maxDR' , help='the cut on the dr to construct the graph' , type=float , default=0.5 )
    opt = parser.parse_args()
        
    file = ROOT.TFile(opt.input)
    tree = file.Get("analyzer1/Trees/Events")

    out_file_name , _ = os.path.splitext( os.path.basename(opt.input) )
    hdf5_file = h5py.File( './{0}.h5'.format(out_file_name) , "w")
    
    pu_groups = {}
    pu_nevents = {}


    num_events = tree.GetEntries()
    for e in tqdm(range(num_events)):
        tree.GetEntry(e)

        # Collect features for particles
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

        # Collect attributes for events
        nVertices = torch.tensor([tree.nVertices], dtype=torch.int32)
        nVGoodVertices = torch.tensor([tree.nVGoodVertices], dtype=torch.int32)
        graph_attr = torch.stack((nVertices, nVGoodVertices), dim=0)

        # Create particle features with only the specified attributes
        Particle_node_features = torch.stack(
            (phi, charge, energy, p, pt, particleDz_, particleDxy_, eta, Type), dim=1
        )

        Vertex_node_features = torch.stack((dzAssociatedPV, particle_vertex_keys), dim=1)

        # Define edges
        edge_index_PP, edge_attr_PP = GetEdgeInfo(eta, phi, opt.maxDR)
        edge_index_V, edge_attr_V = GetDzAssociatedPVInfo(particle_vertex_keys, dzAssociatedPV)

        # Build the heterogeneous graph
        data = HeteroData()
        data['Particles'].x = Particle_node_features
        data['Vertices'].x = Vertex_node_features

        # Add vertex-particle edges
        data['Vertices', 'connects_V', 'Particles'].edge_index = edge_index_V
        data['Vertices', 'connects_V', 'Particles'].edge_attr = edge_attr_V

        # Add particle-particle edges
        data['Particles', 'connects_PP', 'Particles'].edge_index = edge_index_PP
        data['Particles', 'connects_PP', 'Particles'].edge_attr = edge_attr_PP


        # Graph-level attribute (optional: can also keep it at the graph root level)
#        data['graph'].x = graph_attr

        # Graph label (applies to the entire graph)
        graph_label = tree.nInt
        data.y = torch.tensor(graph_label, dtype=torch.int64)

        if graph_label not in pu_groups:
            label = "PU{}".format(graph_label)
            group = hdf5_file.create_group(label)
            pu_groups[graph_label] = group
            pu_nevents[graph_label] = 0

            # Define dtype for heterogeneous graph storage
            current_type = np.dtype([
                # Node features
                ("node_features_Particles", np.dtype("({0},{1})f".format(*data['Particles'].x.shape))),
                ("node_features_Vertices", np.dtype("({0},{1})f".format(*data['Vertices'].x.shape))),
                # Edge attributes and indices for particle-particle edges
                ("edge_attributes_PP", np.dtype("({0},)f".format(len(data['Particles', 'connects_PP', 'Particles'].edge_attr)))),
                ("edge_indices_PP", np.dtype("(2,{0})i".format(len(data['Particles', 'connects_PP', 'Particles'].edge_attr)))),
                # Edge attributes and indices for vertex-particle edges
                ("edge_attributes_VP", np.dtype("({0},)f".format(len(data['Vertices', 'connects_V', 'Particles'].edge_attr)))),
                ("edge_indices_VP", np.dtype("(2,{0})i".format(len(data['Vertices', 'connects_V', 'Particles'].edge_attr)))),
                # Graph-level attributes
 #               ("graph_features", np.dtype("(2,1)i")),
                # Graph labels
                ("graph_labels", np.dtype("(1,)i"))
            ])

            # Create array to store the graph data
            arr = np.empty(shape=(1,), dtype=current_type)

            # Populate the array
            arr["node_features_Particles"] = data['Particles'].x.numpy()
            arr["node_features_Vertices"] = data['Vertices'].x.numpy()
            arr["edge_attributes_PP"] = data['Particles', 'connects_PP', 'Particles'].edge_attr.numpy()
            arr["edge_indices_PP"] = data['Particles', 'connects_PP', 'Particles'].edge_index.numpy()
            arr["edge_attributes_VP"] = data['Vertices', 'connects_V', 'Particles'].edge_attr.numpy()
            arr["edge_indices_VP"] = data['Vertices', 'connects_V', 'Particles'].edge_index.numpy()
  #          arr["graph_features"] = data.graph_attr.numpy()  # Assumes graph_attr is stored under Vertices
            arr["graph_labels"] = data.y.numpy()
            
            # Write graph data to HDF5
            pu_groups[graph_label].create_dataset(
                'E{0}'.format(pu_nevents[graph_label]),
                shape=(1,),
                dtype=current_type,
                data=arr,
                compression="gzip",
                chunks=True
            )
            pu_nevents[graph_label] += 1
            
            # Close HDF5 file after processing all events
    hdf5_file.close()

    import json 
    with open('./{0}.json'.format(out_file_name), "w") as outfile:
        dict_out = {}
        for i,j in pu_nevents.items():
            dict_out[i] = { out_file_name : j }
        outfile.write(json.dumps(dict_out, indent=4))
        
if __name__ == "__main__":
    sys.exit( main() )
