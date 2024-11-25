import torch 
import torch.nn as nn 

class ConvLayer(nn.Module):
    def __init__ (self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_layer = nn.Linear(2*self.atom_fea_len + self.nbr_fea_len, 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        a, b = nbr_fea_idx.shape
        #convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat([atom_in_fea.unsqueeze(1).expand(a,b, self.atom_fea_len), atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_layer(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len*2)).view(a,b, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out 
    
class CGCNN(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,atom_fea_len=91, n_conv=3, h_fea_len=128, n_h=1):
        super(CGCNN, self).__init__()
        #Linear layer for embedding original atom features 
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        #Convolution layers
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)])
        #Linear layer after convolution 
        self.conv_to_fc_layer = nn.Linear(atom_fea_len, h_fea_len)
        #activation after convo to fc 
        self.conv_to_fc_softplus = nn.Softplus()
        #Output layer 
        self.fc_out = nn.Linear(h_fea_len, 1)
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        #pooling atomic features to form crystal features 
        crystal_fea = self.pooling(atom_fea, crystal_atom_idx)
        #passing throuh linear layers 
        crystal_fea = self.conv_to_fc_layer(self.conv_to_fc_softplus(crystal_fea))
        crystal_fea = self.conv_to_fc_softplus(crystal_fea)
        out = self.fc_out(crystal_fea)

        return out 
    
    def pooling(self, atom_fea, crystal_atom_idx):
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0,keepdim=True) for idx_map in crystal_atom_idx]

        return torch.cat(summed_fea, dim=0)
    

if __name__ == '__main__':
    # Dummy Inputs
    N_atoms = 5
    N_neighbors = 3 
    orig_atom_fea_len = 4
    nbr_fea_len = 2
    atom_fea_len = 6
    crystal_count = 2

    atom_fea = torch.rand(N_atoms, orig_atom_fea_len)
    nbr_fea = torch.rand(N_atoms, N_neighbors, nbr_fea_len)
    nbr_fea_idx = torch.randint(0, N_atoms, (N_atoms, N_neighbors))
    crystal_atom_idx = [torch.tensor([0, 1]), torch.tensor([2, 3, 4])]

    # Model Initialization
    model = CGCNN(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=atom_fea_len,
        n_conv=3,
        h_fea_len=8,
        n_h=1,
    )

    # Forward Pass
    try:
        out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")



