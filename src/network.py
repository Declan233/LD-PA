import torch
from torch import nn
import torch.nn.functional as F
import random

################# Random hyperparameter search ################# 
def get_hyperparameters_mlp(max_dense_layers=4):
    hp = {}
    hp["nb_FC_layer"] = random.randrange(1, max_dense_layers + 1, 1)
    hp["FC_neuron"] = random_neuron(hp["nb_FC_layer"])
    hp["activation"] = random.choice(["linear","relu", "selu", "elu", "leakyrelu", "tanh"])
    return hp

def random_neuron(layers):
    neurons = []
    neurons.append(random.choice([100, 200, 300, 400, 500])) 
    for i in range(1, layers):
        neurons.append(random.choice(range(neurons[i-1],0,-100)))
    return neurons

################# Construct network according to hyperparameters ################# 
def activation_function(type:str):
    if type=="relu":
        act = nn.ReLU()
    elif type=="selu":
        act = nn.SELU()
    elif type=="elu":
        act = nn.ELU()
    elif type=="leakyrelu":
        act = nn.LeakyReLU()
    elif type=="tanh":
        act = nn.Tanh()
    else:
        act = nn.Identity()
    return act

def dropout(rate:float):
    if rate>0 and rate<1:
        return nn.Dropout(rate)
    else:
        return nn.Identity()
    

class Generator(nn.Module):
    def __init__(self, target_dim, feature_dim, hp=None):
        '''Initialize network according to hp(hyperparameter) setting.'''
        super(Generator, self).__init__()
        self.target_dim = target_dim
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential()
        if hp is None:
            hp = get_hyperparameters_mlp()
        self.hp = hp
        for layer_index in range(hp["nb_FC_layer"]):
            if layer_index == 0:
                self.mlp.append(nn.Linear(self.target_dim, hp["FC_neuron"][layer_index]))
            else:
                self.mlp.append(nn.Linear(hp["FC_neuron"][layer_index-1], hp["FC_neuron"][layer_index]))
            self.mlp.append(activation_function(hp["activation"]))
        self.mlp.append(nn.Linear(hp["FC_neuron"][layer_index], self.feature_dim))
        self.mlp.append(activation_function(hp["activation"]))
        self.initialize_weights()

    def forward(self, x_tar):
        return self.mlp(x_tar) # (bs,target_dim)->(bs,feature_dim)

    def get_hp(self):
        return self.hp
    
    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)


class SCAClassifier(nn.Module):
    '''This class is the classifier for conducting DL-SCA.'''
    def __init__(self, feature_dim, class_dim, hp):
        '''Initialize network according to hp(hyperparameter) setting.'''
        super(SCAClassifier, self).__init__()
        assert hp is not None, print('Hyperparameters are not designated.')
        self.hp = hp
        self.criterion = nn.CrossEntropyLoss()

        self.cls = nn.Sequential()
        for layer_index in range(hp["nb_FC_layer"]):
            if layer_index == 0:
                self.cls.append(nn.Linear(feature_dim, hp["FC_neuron"][layer_index]))
            else:
                self.cls.append(nn.Linear(hp["FC_neuron"][layer_index-1], hp["FC_neuron"][layer_index]))
            self.cls.append(activation_function(hp["activation"]))
            # self.cls.append(dropout(hp["dropout"]))
        self.cls.append(nn.Linear(hp["FC_neuron"][-1], class_dim))
        self.initialize_weights()
    
    def forward(self, feature):
        '''Take the extracted leakage/feature as input'''
        return self.cls(feature)

    def loss(self, pred_cls, target_cls):
        '''Compute the CrossEntropy Loss between predicted class logits and the target class.'''
        return self.criterion(pred_cls, target_cls)
    
    def get_hp(self):
        return self.hp

    def initialize_weights(self):
        '''Initializing weights and bias.'''
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)


class KLDivLoss(nn.Module):
    def __init__(self, ref_noise=0):
        super(KLDivLoss, self).__init__()
        # additional noise added to the reference.
        self.ref_noise = ref_noise

    def forward(self, input, target):
        input_log_softmax = torch.log_softmax(input, dim=1)
        if self.ref_noise>0:
            target = target + torch.normal(0, self.ref_noise, size=target.shape, device=target.device)
        target_softmax = torch.softmax(target, dim=1)
        return F.kl_div(input_log_softmax, target_softmax, reduction='batchmean')


class MLP_SCA(nn.Module):
    def __init__(self, target_dim, feature_dim, class_dim, g_hp, c_hp, ref_noise=0):
        super(MLP_SCA, self).__init__()
        self.fe = Generator(target_dim, feature_dim, g_hp)
        self.cls = SCAClassifier(feature_dim, class_dim, c_hp)
        self.kl = KLDivLoss(ref_noise)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        f = self.fe(x)
        pred_y = self.cls(f)
        return f, pred_y
    
    def predict(self, x):
        f = self.fe(x)
        pred_y = self.cls(f)
        return pred_y

    def ldloss(self, pred_f, ref_f):
        return self.kl(pred_f, ref_f)
    
    def clsloss(self, pred_y, target_y):
        return self.ce(pred_y, target_y)
    
    def get_hp(self):
        return self.fe.get_hp(), self.cls.get_hp()

