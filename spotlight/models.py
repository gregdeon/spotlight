import torch
import torch.nn as nn

class FactorizedAutoencoder(nn.Module):
    def __init__(self, index_train, index_eval=None, embedding_dim=32):
        super(FactorizedAutoencoder, self).__init__() 
        if index_eval is None:
            index_eval = index_train
        self.enc = SparseSequential(index_train, 
                            SparseExchangeable(5,150, index_train), 
                            nn.LeakyReLU(),
                            torch.nn.Dropout(p=0.5),
                           # SparseExchangeable(150,150, index_train),
                           # nn.LeakyReLU(),
                           # torch.nn.Dropout(p=0.5),
                            #SparseExchangeable(150,150, index_train),
                            #nn.LeakyReLU(),
                            #torch.nn.Dropout(p=0.5),
                            SparseExchangeable(150,150, index_train),
                            nn.LeakyReLU(),
                            torch.nn.Dropout(p=0.5),
                            SparseExchangeable(150,embedding_dim, index_train)
                        )

        self.dec = SparseSequential(index_eval, 
                            SparseExchangeable(2 * embedding_dim, 150, index_eval), 
                            nn.LeakyReLU(),
                            torch.nn.Dropout(p=0.5),
                          #  SparseExchangeable(150,150, index_eval),
                          #  nn.LeakyReLU(),
                          #  torch.nn.Dropout(p=0.5),
                            #SparseExchangeable(150,150, index_eval),
                            #nn.LeakyReLU(),
                            #torch.nn.Dropout(p=0.5),
                            SparseExchangeable(150,150, index_eval),
                            nn.LeakyReLU(),
                            torch.nn.Dropout(p=0.5),
                            SparseExchangeable(150,5, index_eval)
                        )
        
        self.pool_row = SparsePool(index_train[:, 0], embedding_dim, keep_dims=False)
        self.pool_col = SparsePool(index_train[:, 1], embedding_dim, keep_dims=False)
        self._index_train = index_train
        self._index_eval = index_eval

    def set_indices(self, index_train, index_eval=None):
        if index_eval is None:
            index_eval = index_train
        self.index_train = index_train
        self.index_eval = index_eval

    @property
    def index_train(self):
        return self._index_train
    
    @index_train.setter
    def index_train(self, index):
        self.enc.index = index
        self.pool_row.index = index[:, 0]
        self.pool_col.index = index[:, 1]
        self._index_train = index

    @property
    def index_eval(self):
        return self._index_eval
    
    @index_eval.setter
    def index_eval(self, index):
        self.dec.index = index
        self._index_eval = index

    def forward(self, input):
        encoded = self.enc(input)
        row_mean = self.pool_row(encoded)
        col_mean = self.pool_col(encoded)
        embeddings = torch.cat([torch.index_select(row_mean, 0, self.index_eval[:, 0]), 
                                torch.index_select(col_mean, 0, self.index_eval[:, 1])], dim=1)
        output = self.dec(embeddings)
        return output

xray_model = nn.Sequential(
    nn.Conv2d(1, 10, (3,3)),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.MaxPool2d((2,2)),
    nn.Flatten(),
    nn.Linear(54760, 100),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(100, 2)
)
