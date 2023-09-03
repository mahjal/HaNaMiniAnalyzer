import os
import sys
import json
import h5py
import torch
from torch_geometric import utils
import pandas as pd
from tqdm.notebook import trange, tqdm
import PUGNN as pg
import numpy   as np 
import pytz
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from torch_geometric.loader import DataLoader

import os.path as osp
import torch_geometric as pyg

from numpy           import random
from scipy.stats import poisson

import scipy
import scipy.optimize as optim

def is_ext(string:str, ext0='json'):
    
    res = string.split('.')
    if (len(res) != 2) or (string.find('.') < 0):
        return False, string
 
    name, ext = res
    return (ext == ext0), name

def check_and_summarize(directory, to='./', rewrite=True, log=True):
    nfiles = 0
    json_files = []
    summary = dict()
    
    for file in os.listdir(directory):
        is_hdf5_file, file_name = is_ext(file, 'h5')

        if is_hdf5_file:
            try:
                with h5py.File(directory + file) as f:
                    pass

                with open(osp.join(directory, file_name + '.json'), "r") as j:
                    d = json.load(j)
                    nfiles += 1
                    for pu in d:
                        try:
                            summary[pu][file_name] = d[pu][file_name]
                        except KeyError:
                            summary[pu] = d[pu]
                            
            except Exception as e:
                if log:
                    print(e, ':', file, file=sys.stderr)

    # summary
    if rewrite:
        with open(os.path.join(to, 'summary.json'), "w") as outfile:
                outfile.write(json.dumps(summary, indent=4))
    return nfiles

def reset_parameters(the_model):
    for layer in the_model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def check_tensors(*tensors):
    nnan = 0
    ninf = 0
    for tensor in tensors:
        nnan += tensor.isnan().sum()
        ninf += tensor.isinf().sum()   
    return nnan > 0, ninf > 0 

def check_data(data, b_ind=-1):
    inf_status, nan_status = check_tensors(data.edge_attr, data.edge_index, data.x, data.nvertices, data.nparticles, data.y)
    
    clean = not (inf_status or nan_status)
    err   = None
    
    if not clean:
        if inf_status and nan_status:
            err = f"`nan` and `inf` detected in batch no. {b_ind}. Skipping the batch..."

        elif inf_status:
            err = f"`inf` detected in batch no. {b_ind}. Skipping the batch..."

        elif nan_status:
            err = f"`nan` detected in batch no. {b_ind}. Skipping the batch..."
    
    return clean, err

def llikelihood_pois(lam, x):
    return (x * np.log(lam) - lam - x*np.log(x) + x).sum()

def max_log_likelihood(x, loglikelihood_func, init_guess):
    
#     optim_res = optim.minimize(lambda p: -loglikelihood_func(p, x), init_guess)
    
#     if not optim_res.success:
#         raise RuntimeError("Maximum Likelihood Estimation Failed")
        
    
    def helper(p, p0):
        L0 = loglikelihood_func(p0, x)
        return L0 - loglikelihood_func(p, x) - 1
    
#     mle = optim_res.x
    mle = x.mean()
    
    res_r = scipy.optimize.root(helper, mle + 0.1, args=(mle,))
    res_l = scipy.optimize.root(helper, mle - 0.1, args=(mle,))
    success = res_r.success and res_l.success
    
    if not success:
        raise RuntimeError("Error interval Computation Failed")
    
    return mle - res_l.x, mle, res_r.x - mle


def R_squared(y, yhat, y_prime=None):
    RSS = (y - yhat)**2
    
    if y_prime is not None:
        RSS_prime = (yhat - y_prime) ** 2
    else:
        RSS_prime = (yhat - y.mean()) ** 2
    
    return 1 - RSS/RSS_prime


def remove_nan_node(node_features, edge_index, edge_attributes):    
    node_ID, feature_ID = np.where(np.isnan(node_features))
    node_features       = np.delete(node_features, node_ID, axis=0)
    
    for node in node_ID:
        _, edge    = np.where(edge_index == node)
        edge_index = np.delete(edge_index, edge, axis=1)
        edge_attributes = np.delete(edge_attributes, edge, axis=0)
    
    return node_features, edge_index, edge_attributes


def remove_inf_node(node_features, edge_index, edge_attributes):    
    node_ID, feature_ID = np.where(np.isinf(node_features))
    node_features       = np.delete(node_features, node_ID, axis=0)
    
    for node in node_ID:
        _, edge    = np.where(edge_index == node)
        edge_index = np.delete(edge_index, edge, axis=1)
        edge_attributes = np.delete(edge_attributes, edge, axis=0)
    
    return node_features, edge_index, edge_attributes
    

def to_data(raw_data, log=False, *args):
    nf = np.array(raw_data["node_features"], dtype=np.float32)
    ea = np.array(raw_data["edge_attributes"], dtype=np.float32)
    ei = np.array(raw_data["edge_indecies"], dtype=np.int64)
    gf = np.array(raw_data["graph_features"], dtype=np.float32)
    gl = np.array(raw_data["graph_labels"], dtype=np.float32)
    
    edge_indecies, edge_weights = utils.to_undirected(torch.from_numpy(ei), torch.from_numpy(ea))
    
    if np.isnan(nf).sum() > 0:
        nf, ei, ea = remove_nan_node(node_features=nf, edge_index=ei, edge_attributes=ea)
        if log:
            print("'{}' : 'PU={}' : 'E{}' : nan".format(*args), file=sys.stderr)
            
    if np.isinf(nf).sum() > 0:
        nf, ei, ea = remove_inf_node(node_features=nf, edge_index=ei, edge_attributes=ea)
        if log:
            print("'{}' : 'PU={}' : 'E{}' : inf".format(*args), file=sys.stderr)
            
    return pyg.data.Data(x         =torch.from_numpy(nf),
                    edge_index     =edge_indecies,
                    edge_attr      =edge_weights,
                    nvertices      =torch.from_numpy(gf[0]),
                    nparticles     =torch.tensor([len(nf)]).unsqueeze(1),
                    y              =torch.from_numpy(gl)
                 )

class IndexingSystem(object):
    """
    An indexing system based on metadata defined on multiple files.
    
    Example of allowed metadata:
        {"1" :  {"file1":"62", "file2":"13", "file3":"24"},
         "2" :  {"file2":"2", "file5":"33", "file6":"245"},
         "14":  {"file1":"21", "file2":"23", "file3":"34"}}
    Here, we have three classes of data distributed in six files.
    
    For many puposes, we may want to collect a sample flom these 
    ambigious databases without distruction and recunstruction.
    
    This system let us to map a given index into a data inside 
    the database. Also, it allows us to do sampling over database.
    """
    
    def __init__(self, metadata, collection, seed, **sampling_strategy):
        self._MD         = metadata    # Metadata
        self._MDSummary  = dict()      # Metadata Summary
        self._SampleMD   = dict()      # Sample Metadata
        self._SMDInfo    = dict()      # Sample Metadata Information (= Summary)
        self._filenames  = None        # File Names Container
        self._No_SamData = None        # Number of sampled data we can access through this IndexingSystem
        self._No_TotData = None        # Number of total data we can access through this IndexingSystem
        self._seed       = seed
        self._summarize()
        self._smaple_from_collection(collection, **sampling_strategy)
        
    def _summarize(self):
        """ 
        Summarizing the metadata:
            1. List of files
            2. Dictionary for pu - #pu pair; i.e. DICT[PU] = #PU
        """
        files = set()
        for PU in self._MD:
            PU_MD = self._MD[PU]
            current_PU = 0
            for file_name in PU_MD:
                files.add(file_name)
                current_PU += int(PU_MD[file_name])
               
            self._MDSummary[PU] = current_PU
        self._filenames = list(files)
                        
    def _DCPrep(self, data_col:dict):
        """
        Data collection preprocessing:
            1. filtering mising pu's
            2. Avoiding overflow
        """
        rejected = dict()
        for PU in data_col:
            try:
                wanted_PU    = data_col[PU]
                available_PU = self._MDSummary[PU]
            except KeyError:
                print(f"PU = {PU} is not available. Excluding PU = {int(PU):03d} ...", file=sys.stderr)
                rejected[PU] = data_col[PU]
                # Excluding ...
                continue
            if available_PU < wanted_PU:
                w = f"PU = {int(PU):03d} overflows... Setting PU = {int(PU):03d} to {available_PU} instead of {wanted_PU}"
                print(w, file=sys.stderr)
                wanted_PU = available_PU
                
            self._SampleMD[PU] = wanted_PU
        self._No_SamData = np.sum(list(self._SampleMD.values()))
        return rejected
    
    def _smaple_from_collection(self, dc:dict, **sampling_kwargs):
        """
        Random sampling of the whole data from sample metadata
        """
        random.seed(self._seed)
        rejected_PU    = self._DCPrep(dc)        
        for PU in self._SampleMD:
            PU=str(int(PU))
            self._SMDInfo[PU] = random.choice(range(int(self._MDSummary[PU])), size=int(self._SampleMD[PU]), **sampling_kwargs)        
    
    def _GETITEM(SELF, CONTAINER_DICT, IND):
        """
        Assuming `CONTAINER_DICT` has a shape as `self._MDSummary` i.e. {cls_i: `freq of cls_i`}.
        In this case, indexing is over "All" classes not a specific one. 
        Knowing the frequency of each class, we iterate over `CONTAINER_DICT` to find "local" index.
        In summary, we want to find the `class` and local index corresponding to a given `IND`.
        """
        
        for key in CONTAINER_DICT:
            IND -= CONTAINER_DICT[key]
            if IND < 0 :
                IND += CONTAINER_DICT[key]
                return key, IND
        raise IndexError
    
    def _map_index(self, ind):
        """
        `ind` is a "global" index i.e. defind over all "sampled" data.
        Trivially, each "global" index is reffered to a "local" index.
        It is mapped into a local index of a given PU associated with `ind`.
        """
#         print(self._SMDInfo)
        PU, ind = self._GETITEM(self._SampleMD, ind)
#         print(PU, ind)
        real_index = self._SMDInfo[PU][int(ind)]
        return PU, real_index
    
    def _getitem(self, ind):
        """
        This method do the final job of index processing system.
        It returns all information in order to load the data.
        """
        PU, PU_index = self._map_index(ind) 
        PU_MD = self._MD[PU]
        file_name, infile_index = self._GETITEM(PU_MD, PU_index)
        return file_name, PU, infile_index
    
    @property
    def num_available_data(self):
        """
        Returns the number of sampled data
        """
        return self._No_SamData
   
    @property
    def num_all_data(self):
        """
        Returns the number of allowed data
        """
        return self._No_TotData

    @property
    def files(self):
        """
        Returns the file names
        """
        return self._filenames
    
    @property
    def keys(self):
        """
        Returns available PU classes
        """
        return self._SampleMD.keys()  
    
class GraphDataset(IndexingSystem, pyg.data.Dataset):
    
    def __init__(self, root, metadata_dir, sample_metadata, seed,
                 sampling_strategy={"replace":False, "p":None}, reader=to_data,
                 transform=None, pre_transform=None, pre_filter=None, log=False):
        
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))
        
        with open(metadata_dir, "r") as jsonfile:
            metadata_dict = json.load(jsonfile)
        
        super().__init__(metadata_dict, sample_metadata, seed, **sampling_strategy)
        
        self.root          = root
        self.mddir         = metadata_dir
        self.transform     = transform
        self.pre_transform = pre_transform
        self.pre_filter    = pre_filter
        self.log           = log
        self._indices      = None
        self._files        = dict()
        self._data_reader  = reader
        self.process()
        
        self._map_classes = dict(zip(self.keys, list(range(self.num_classes))))
        
    def map_class(self, data):
        data.y = self._map_classes[str(int(data.y))]
    
    def process(self):
        """Opening all files with read-only mode"""
        if self.log and 'pytest' not in sys.modules:
            print('Processing...', file=sys.stderr)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)
    
    def len(self):
        """Returns length of dataset i.e. numberr of graphs in dataset"""
        return int(self.num_available_data)
    
    def indices(self):
        """Returns indices array in increasing order"""
        return range(self.len()) if self._indices is None else self._indices

    def get(self, idx):
        """
        Returns the data associated with `idx` under our inherited `IndexingSystem`
        """
        filename, PU, infile_index = IndexingSystem._getitem(self, idx)
        
        path_to_file = osp.join(self.root, filename + ".h5")
        
        with h5py.File(path_to_file, "r") as file:
            raw_data = file[f"PU{PU}"][f"E{infile_index}"][0]
        
        if self.pre_filter is not None:
            raw_data = self.pre_filter(raw_data)
        
        if self.pre_transform is not None:
            raw_data = self.pre_transform(raw_data)
        
        data = self._data_reader(raw_data, self.log, filename, PU, infile_index)
        
        if self.transform is not None:
            data = self.transform(data)
            
        # Checking the data is cleaned:
        if not check_data(data)[0]:
            s = f"The record {idx} at {filename}: {PU}: {infile_index} contains nan, even after cleaning.\nMaybe there is a bug or a serious problem in data."
            print(s, file=sys.stderr)

        return data
    
    def file_dir(self) -> str:
        """
        Returns direcotry of files in which are located.
        It is assumed all files are in '~root/data/' directory.
        """
        return osp.join(self.root, 'data')
    
    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return len(self.keys)
    
    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0] 
        
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")
    
    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = self[0]
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_edge_features'")


class PUGNNModel(object):
    def __init__(self, name, desc, model, dataset, root='./', device='auto', seed=42, disable_progress_bar=False):
        self._prog    = disable_progress_bar
        self._root    = root
        self._name    = name
        self._desc    = desc
        self._seed    = seed
        self._model   = model
        self._dataset = dataset
        
        self._main_directory = os.path.join(root, f'PU-{self._name}')
        self._sub_directory  = str(dt.datetime.now(pytz.timezone('Asia/Tehran'))).split('.')[0].replace(' ', '@') + '--' +  self._desc
        
        self._train_gen      = None
        self._validation_gen = None
        self._test_gen       = None
        
        self._train_len      = None
        self._validation_len = None
        self._test_len       = None
        
        self._optimizer      = None
        self._loss_func      = None
        self._lr_scheduler   = None
        
        self._batch_size = None
        
        self._OUTPUST_SUM             = None
        self._OUTPUTS_SUMNORMSQUARED  = None
        
        if isinstance(device, str):
            if device.lower() != 'auto':
                err = "Unknown setting mode for device `{}`. The device will be set automatically".format(device)
                print(err, file=os.stderr)
            use_cuda = torch.cuda.is_available()
            self._device = torch.device("cuda:0" if use_cuda else "cpu")
        elif isinstance(device, torch.device):
            self._device = device
        else:
            raise ValueError("Unsupporetd device")
        
        if not os.path.isdir(self._main_directory):
            os.mkdir(self._main_directory)
        
        os.mkdir(os.path.join(self._main_directory, self._sub_directory))
        
        self._history = dict()
        ind = 0
        for entry in os.listdir(self._main_directory):
            print(entry)
            if os.path.isdir(os.path.join(self._main_directory, entry)):
                self._history[ind] = entry
                ind += 1
        
    @property
    def history(self):
        return self._history
    
    @property
    def training_loader(self):
        return self._train_gen
    
    @property
    def validation_loader(self):
        return self._validation_gen
    
    @property
    def test_loader(self):
        return self._test_gen
    
    @property
    def model(self):
        return self._model
    
    @property
    def device(self):
        return self._device
    
    def load_last_model(self, history_level=None, search=True):
        if history_level is None:
            history_level = len(self._history) - 1
            search = True
            
        if history_level < 0:
            print("The is no saved model", file=sys.stderr)
            return None
        
        direc = os.path.join(self._main_directory, self._history[history_level])
        models = list(filter(lambda name: is_ext(name, ext0='pt')[0], os.listdir(direc)))
        
        if len(models) == 0:
            if search:
                return self.load_last_model(history_level=history_level - 1, search=True)
            
            print(f"There's not any saved model at this level (={history_level}).")
        return torch.load(os.path.join(direc, models[-1]))
        
    def get_ready(self, test_set_per, validation_set_per, dataloader_params:dict):
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        self._validation_len = (len(self._dataset) * validation_set_per) // 100 
        self._test_len       = (len(self._dataset) * test_set_per) // 100
        self._train_len      = len(self._dataset) - self._validation_len - self._test_len
        
        train_subset, valid_subset, test_subset = torch.utils.data.random_split(self._dataset, lengths=[self._train_len, self._validation_len, self._test_len])
        self._batch_size = dataloader_params['batch_size']
        
        # Generators
        self._train_gen      = DataLoader(train_subset, **dataloader_params)
        self._validation_gen = DataLoader(valid_subset, **dataloader_params)
        self._test_gen       = DataLoader(test_subset,  **dataloader_params)
    
    def _train_one_epoch(self, epoch=-1, clip=None):
        self._model.train()        
        for data, b_ind in zip(self._train_gen, trange(len(self._train_gen) - 1, desc=f"Epoch {epoch+1:03d}", unit="Batch", disable=self._prog)):
            
            clean, err = check_data(data, b_ind)
            if not clean:
                print(err, file=sys.stderr)
                continue

            data   = data.to(self._device)
            out    = self._model(data)    # Perform a single forward pass.
            loss   = self._loss_func(out, data.y.unsqueeze(1))                           # Compute the loss.

            if np.isnan(loss.item()):
                w = "nan loss detected. Perhaps there is a divergance. Stopping the training..."
                return 1, (w, data)

            loss.backward()  # Derive gradients.

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), clip)

            self._optimizer.step()  # Update parameters based on gradients.
            self._optimizer.zero_grad()  # Clear gradients.
            del data 

        return 0, None

    def train(self, max_epochs, optimizer, optimizer_args,
                                loss_fn, loss_fn_args=dict(),
                                lr_scheduler=None, lr_scheduler_args=dict(),
                                grad_clipping_number=None, save_models=True, reset=True):

        print("Training device {}...\n".format(self._device))
        torch.backends.cudnn.benchmark = True
        
        if reset: 
            reset_parameters(self._model)
            
        self._model        = self._model.to(self._device)
        self._loss_func    = loss_fn(**loss_fn_args)
        self._optimizer    = optimizer(self._model.parameters(), **optimizer_args)
        self._lr_scheduler = lr_scheduler(self._optimizer, **lr_scheduler_args) if lr_scheduler is not None else None

        train_loss_arr = np.zeros(max_epochs)
        valid_loss_arr = np.zeros(max_epochs)
        
        is_failed = False

        for epoch in trange(max_epochs, desc=f'Training the {self._name}...', unit='Epoch', ncols=950, disable=self._prog):
            torch.cuda.empty_cache()
            
            is_failed, res = self._train_one_epoch(epoch, grad_clipping_number)
            if is_failed:
                print(res[0], file=sys.stderr)
                return is_failed, res
            
            is_failed, res = self.evaluate("train")
            if is_failed:
                print(res[0], file=sys.stderr)
                return is_failed, res
            train_loss_arr[epoch] = res
            
            is_failed, res = self.evaluate("validation")
            if is_failed:
                print(res[0], file=sys.stderr)
                return is_failed, res
            valid_loss_arr[epoch] = res
            
            if save_models:
                torch.save(self._model, f'{os.path.join(self._main_directory, self._sub_directory)}/model-epoch-{epoch:0{len(str(max_epochs))}d}.pt')
                        
            print(f'Training Set Loss: {train_loss_arr[epoch]:.4f}, Validation Set Loss: {valid_loss_arr[epoch]:.4f}\n')
            
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
        
        return is_failed, (train_loss_arr, valid_loss_arr)
        
    def evaluate(self, eval_set:str):
        assert eval_set.lower() in ['validation', 'train']
        self._model.eval()
        is_training_set = eval_set.lower() == 'train'        
        length   = len(self._train_gen) if is_training_set else len(self._validation_gen)
        loss_arr  = torch.Tensor(length).zero_()
        total = self._train_len if is_training_set else self._validation_len
        b_ind = 0
        
        failed = False
        
         
        with torch.no_grad():
            # Iterate in batches over the training/test/validation dataset.
            for data in (self._train_gen if is_training_set else self._validation_gen):
                clean, err = check_data(data, b_ind)
                if not clean:
                    print(err, file=sys.stderr)
                    b_ind += 1
                    continue

                data = data.to(self._device)
                out = self._model(data)  

                if out.cpu().detach().isnan().sum() > 0:
                    w = f"nan loss detected during evaluation of {'training set' if is_training_set else 'validation set'}. Perhaps there is a problem..."
                    failed = True
                    return failed, (w, data)

                loss_arr[b_ind] += self._loss_func(out, data.y.unsqueeze(1)).cpu().item() * len(data) / total
                b_ind += 1
                del data
        
        return failed, loss_arr.sum()
    
    def summary(self, last_saved=False):
        if last_saved:
            res = self.load_last_model()
            if res is None:
                print('PU Summary process for the last saved model was not successful. The current model is passed...')
            else:
                return PUGNNSummary(self._test_gen, res, seed=self._seed)
            
        return PUGNNSummary(self._test_gen, self._model, seed=self._seed)

        
class PUGNNSummary():
    def __init__(self, loader, model, device='auto', seed=42):
        self._seed   = seed
        self._loader = loader
        self._model  = model
        self._yhat   = None
        self._y      = None
        self._range  = None
        
        if isinstance(device, str):
            if device.lower() != 'auto':
                err = "Unknown setting mode for device `{}`. The device will be set automatically".format(device)
                print(err, file=os.stderr)
            use_cuda = torch.cuda.is_available()
            self._device = torch.device("cuda:0" if use_cuda else "cpu")
            
        elif isinstance(device, torch.device):
            self._device = device
        
        else:
            raise ValueError("Unsupporetd device")
        
        print("===================== PUGNNSummary =====================")
        print()
        print(f"The device is set to {self._device}")
        print()
        print(f"Model is set to {self._model}")
        print()
        print(f"Data loader with length {len(self._loader)} is set.")
        print()
        print("====================== Successful ======================")
    
    @property
    def yhat(self):
        return self._yhat
    
    @property
    def y(self):
        return self._y
        
    def process(self):
        self._model = self._model.to(self._device)
        self._model.eval()
        
        self._yhat = torch.tensor([], dtype=float)
        self._y    = torch.tensor([], dtype=float)
        
        b_ind = 0
        
        with torch.no_grad():
            for data in tqdm(self._loader, desc='Evaluating...', unit='Batch', ncols=1000):
                clean, err = check_data(data, b_ind)
                if not clean:
                    print(err, file=sys.stderr)
                    b_ind += 1
                    continue

                self._y = torch.concat([self._y, data.y])

                data = data.to(self._device)
                out  = self._model(data)

                self._yhat = torch.concat([self._yhat, out.cpu().detach()])

                b_ind += 1

                del data
        
        self._yhat = self._yhat.squeeze().detach().numpy()
        self._y    = self._y.detach().numpy()
        
        self._range = np.arange(self._y.min(), self._y.max() + 2)
        
    def histogram(self):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self._y, name='PU'))
        fig.add_trace(go.Histogram(x=self._yhat, name='Estimated PU'))
        fig.update_xaxes("PU")
        fig.update_yaxes("Count")
        return fig
    
    def residual_plot(self):
        residuals = self._yhat - self._y
        fig = px.scatter(x=self._yhat, y=residuals)
        fig.update_yaxes(title='Residuals')
        fig.update_xaxes(title='Estimated PU')
        fig.update_layout(title="Residual Plot")
        return fig
    
    def compare(self, **outputs):
        RSS = (self._y - self._yhat)**2
        TSS = (self._y - self._y.mean())**2
        
        r_squared_array = np.zeros(len(outputs) + 1)
        r_squared_array[0] = 1 - RSS.sum() / TSS.sum()
        
        fig = go.Figure()
        
        for i, output_name in enumerate(outputs):
            y, yhat = outputs[output_name]
            fig.add_trace(go.Scatter(x=y, y=yhat, name=output_name))
            RSS_p = (yhat - self._yhat)**2
            r_squared_array[i + 1] = 1 - RSS.sum() / RSS_p.sum()
        
        fig.add_trace(go.Scatter(x=self._y, y=self._yhat, name='The Model'))
        fig.add_trace(go.Scatter(x=self._range, y=self._range, name='Actual Trend'))
        fig.update_xaxes(title='No. PU')
        
        return fig, r_squared_array
    
    def _poisson_dist_extraction(self, av, f=0.6):
        prv = poisson(av)
        n  = len(self._y) / len(self._range)
        N  = int(f*n / prv.pmf(av))

        freq       = np.histogram(prv.rvs(N), self._range)[0]
        pu_freq    = dict(zip(self._range, freq))
        pu_counter = dict(zip(self._range, freq))
        pu_inds    = []

        for i, data in enumerate(tqdm(self._y,  desc=f'PU = {av} Indexing...', unit='Graph')):
            y = int(data)
            if pu_counter[y] > 0:
                pu_inds.extend([i])
                pu_counter[y] -= 1
        
        if sum(list(pu_counter.values())) > 1:
            w = f"Change factor variable `f` (current value is `{f}`) based on the following:"
            print(w, file=sys.stderr)
            for pu in pu_counter:
                if pu_counter[pu] > 0:
                    w = f'\tPoisson distribution({av}) for PU = {pu} is violated by {pu_counter[pu]} count less than expected'
                    print(w, file=sys.stderr)
        return pu_inds
    
    def LLEstimation(self, av_pu, f=0.6):
        inds = self._poisson_dist_extraction(av_pu, f)
        xhat = self._yhat[inds]
        x    = self._y[inds]
        
        return max_log_likelihood(xhat, llikelihood_pois, av_pu), x.mean()
    
    def rangeLLE(self, starting_pu, ending_pu, f):
        av_PUs = np.array([*range(starting_pu, ending_pu + 1)])

        mle_arr       = np.zeros(len(av_PUs))
        act_arr       = np.zeros_like(mle_arr)
        left_err_arr  = np.zeros_like(mle_arr)
        right_err_arr = np.zeros_like(mle_arr)

        for ind, pu in enumerate(tqdm(av_PUs, desc='Range-Based LLE',  unit='PU', ncols=950)):
            temp, act = self.LLEstimation(pu)
            le, mle, re = temp
            
            mle_arr[ind]       = mle
            act_arr[ind]       = act
            left_err_arr[ind]  = le
            right_err_arr[ind] = re
        
        fig = go.Figure(data=go.Scatter(
            x=av_PUs,
            y=mle_arr,
            error_y=dict(
                type='data',
                symmetric=False,
                array=right_err_arr,
                arrayminus=left_err_arr,
                visible=True), name='Estimated <PU>', mode='markers')
                )
        
        fig.add_trace(go.Scatter(x=av_PUs, y=av_PUs, name='Expected'))
        
        return fig, left_err_arr, mle_arr, right_err_arr, act_arr
        
        
class PUGNN(object):
    def __init__(self, name, in_dir, out_dir, from_pu, to_pu, freq, data_reader_function=to_data, transform=None, pre_transform=None, pre_filter=None, log=False, seed=42, disable_progress_bar=False):
        assert from_pu < to_pu
        assert isinstance(freq, int) and freq > (to_pu - from_pu)
        
        print('Checking the input directoy and making the summary metadata...', file=sys.stderr)
        nfiles = check_and_summarize(in_dir, out_dir)
        print(f'{nfiles} files found.', file=sys.stderr)
        if nfiles == 0:
            print("The input directory doesn't have any compatible files. Please change the directory and re-initialize '{self._name}' by calling it." , file=sys.stderr)
        
        self._in     = in_dir
        self._out    = out_dir
        self._name   = name
        self._dpb    = disable_progress_bar
        self._model  = None
        self._seed   = seed
        self._range  = range(from_pu, to_pu + 1)
        self._len    = freq * len(self._range)
        self._collec = dict(zip(map(str, list(self._range)), np.ones(len(self._range)) * freq ))
        self._reader = data_reader_function
        self._tr     = transform
        self._pre_tr = pre_transform
        self._pre_f  = pre_filter
        self._log    = log
        
        self._process()
    
    def _process(self):
        try:
            self._dataset = GraphDataset(
                                root=self._in, 
                                metadata_dir=os.path.join(self._out,"summary.json"),
                                sample_metadata=self._collec, 
                                seed=self._seed,
                                reader=self._reader,
                                transform=self._tr,
                                pre_transform=self._pre_tr,
                                pre_filter=self._pre_f,
                                log=self._log)
        except Exception as ex:
            print('During initiation, we got a problem.', file=sys.stderr)
            print(f'\tEXCEPTION OCCURED :: {ex}', file=sys.stderr)
            print(f"You can re-initiate the '{self._name}' by calling it.", file=sys.stderr)
    
    def __call__(self, *args, **kwargs):
        self.__init__(self, *args, **kwargs)
            
    def describe(self):
        print('*'*100)
        print('*'*100)
        print('Dataset: ')
        print()
        print(f'\t{self._dataset}:')
        print(f"\t{'='*25}")
        print(f'\tNumber of graphs: {len(self._dataset)}')
        print(f'\tNumber of features: {self._dataset.num_features}')
        print(f'\tNumber of classes: {self._dataset.num_classes}')

        data = self._dataset[100]  # Get the first graph object.

        print()
        print('\t', data)
        print(f"\t{'='*92}")

        # Gather some statistics about the first graph.
        print(f'\tNumber of nodes: {data.num_nodes}')
        print(f'\tNumber of edges: {data.num_edges}')
        print(f'\tAverage node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'\tHas isolated nodes: {data.has_isolated_nodes()}')
        print(f'\tHas self-loops: {data.has_self_loops()}')
        print(f'\tIs undirected: {data.is_undirected()}')
        print()
        print()
        print('*'*100)
        print()
        print("Model:")
        print()
        print(f"{self._model}")
        
        if self._model is not None:
            total_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            print("dof =", total_params)
        
        print('*'*100)
        print('*'*100)
    
    @property
    def model(self):
        return model
    
    def set_model(self, model):
        self._model = model
        
    def toPUGNNModel(self, sub_name, test_precentage, validation_percentage, data_loader_parameters):
        if self._model is None:
            return print("Please set the model first...", file=sys.stderr)
            
        pgm = PUGNNModel(name=self._name, desc=sub_name, model=self._model, dataset=self._dataset, root=self._out, seed=self._seed,
                         disable_progress_bar=self._dpb)
        pgm.get_ready(test_precentage, validation_percentage, data_loader_parameters)
        return pgm
        
        