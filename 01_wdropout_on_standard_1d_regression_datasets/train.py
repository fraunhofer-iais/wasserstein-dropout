import time
import re
import numpy as np

import torch
import torch.nn as nn
from sklearn.utils import shuffle
from functional import nll_floored, evidential_loss, new_exact_wasserstein_dropout_loss, concrete_dropout_loss
from models import Net, Net_PU, Net_Evidential, Net_SWAG, CDNet

# base parameters
n_output = 1
net_params = {'n_output': n_output,
            'layer_width':100,
            'num_layers':2,
            'nonlinearity':nn.ReLU(), #tanh,sigmoid
            'init_corrcoef':0.0,
            'de_components': 5} 

train_params = {'device': 'cpu', #torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
              'drop_bool': True,
              'drop_bool_ll': True,
              'drop_p': 0.1,
              'num_epochs': 1000,
              'batch_size': 100,
              'learning_rate': 0.001,
              'loss_func': torch.nn.MSELoss(reduction='mean'),
              'weight_decay': 0,
              'loss_params': [5,1,False]}

def get_net_from_method(method, n_feat, n_datapoints, net_params=net_params, train_params=train_params, n_output=1):
    
    net_params['n_input'] = n_feat
    net_params['n_datapoints'] = n_datapoints
    
    if method == 'vanilla':
        net_params['n_output'] = n_output
        train_params['drop_bool'] = False
        train_params['drop_bool_ll'] = False
        train_params['loss_func'] = torch.nn.MSELoss(reduction='mean')
        train_params['loss_params'] = None
        
        net = Net(net_params=net_params, train_params=train_params)
        net.to(train_params['device'])
        return net
    
    # new wdrop exact
    elif method.startswith('new_wdrop_exact'):
        
        pat = re.compile(r'new_wdrop_exact_l=([1-9][0-9]*)(_p=(0.[0-9]+))?((\Z)|_(\w+))')
        match = pat.match(method)
        
        if match is not None:
            grps = match.groups()
            
            if grps[0] is not None:
                l = int(grps[0])
                net_params['n_output']       = n_output    
                train_params['drop_bool']    = True
                train_params['drop_bool_ll'] = True
                train_params['loss_func'] = new_exact_wasserstein_dropout_loss

                if grps[2] is not None: # p specified
                    p = float(grps[2])
                    train_params['drop_p'] = p
                
                if grps[5] is None:
                    train_params['loss_params'] = [l, 1, False]
                    net = Net(net_params=net_params, train_params=train_params)
                    net.to(train_params['device'])
                    return net

    elif method == 'mc_pu':
        
        net_params['n_output']       = int(1.5*n_output + 0.5*n_output**2) # mean + lower triangle of covariance
        train_params['drop_bool']    = True
        train_params['drop_bool_ll'] = True
        train_params['loss_func']    = nll_floored if n_output == 1 else nll_floored_nd
        #train_params['num_epochs']   = 2000
        train_params['loss_params'] = None

        net = Net_PU(net_params=net_params,train_params=train_params)
        net.to(train_params['device'])
        return net
        
    elif method == 'mc_ll':  # dropout in last layer, standard mse
        net_params['n_output']       = n_output    
        train_params['drop_bool']    = False
        train_params['drop_bool_ll'] = True
        train_params['loss_func']    = torch.nn.MSELoss(reduction='mean')
        #train_params['num_epochs']   = 2000
        train_params['loss_params'] = None

        net = Net(net_params=net_params,train_params=train_params)
        net.to(train_params['device'])
        return net
        
    elif method.startswith('mc'):  # dropout in all layers, standard mse
        
        pat = re.compile(r'mc(_p=(0.[0-9]+))?(_wd=(0.[0-9]+))?')
        mat = pat.match(method)
        if mat is not None:
            grps = mat.groups()
            if grps[1] is not None:
                train_params['drop_p'] = float(grps[1])
            if grps[3] is not None:
                train_params['weight_decay'] = float(grps[3])
        
        net_params['n_output']       = n_output    
        train_params['drop_bool']    = True
        train_params['drop_bool_ll'] = True
        train_params['loss_func']    = torch.nn.MSELoss(reduction='mean')
        #train_params['num_epochs']   = 2000
        train_params['loss_params'] = None

        net = Net(net_params=net_params,train_params=train_params)
        net.to(train_params['device'])
        return net

    elif method == 'pu':    # trains mu, sigma uses nll loss
        net_params['n_output']       = int(1.5*n_output + 0.5*n_output**2) # mean + lower triangle of covariance
        train_params['drop_bool']    = False
        train_params['drop_bool_ll'] = False
        train_params['loss_func']    = nll_floored if n_output == 1 else nll_floored_nd
        #train_params['num_epochs']   = 2000
        train_params['loss_params'] = None

        net = Net_PU(net_params=net_params,train_params=train_params)
        net.to(train_params['device'])
        return net
    
    elif method == 'evidential':
        net_params['n_output'] = 4*n_output
        train_params['drop_bool'] = False
        train_params['drop_bool_ll'] = False
        train_params['lmbda'] = 0.001
        train_params['loss_func'] = lambda y_pred, y_gt: evidential_loss(y_pred, y_gt, train_params['lmbda'])
        train_params['loss_params'] = None
        net = Net_Evidential(net_params=net_params, train_params=train_params)
        net.to(train_params['device'])
        return net
        
    elif method == 'de':
        net_params['n_output']       = n_output
        train_params['drop_bool']    = False
        train_params['drop_bool_ll'] = False
        train_params['loss_func']    = torch.nn.MSELoss(reduction='mean')
        #train_params['num_epochs']   = 2000
        train_params['loss_params'] = None
        
        net = []
        for i in range(net_params['de_components']):
            net_ = Net(net_params=net_params,train_params=train_params)
            net_.to(train_params['device'])
            net.append(net_)
        return net
            
    elif method == 'pu_de':  
        net_params['n_output']       = int(1.5*n_output + 0.5*n_output**2) # mean + lower triangle of covariance
        train_params['drop_bool']    = False
        train_params['drop_bool_ll'] = False
        train_params['loss_func']    = nll_floored if n_output == 1 else nll_floored_nd
        #train_params['num_epochs']   = 2000
        train_params['loss_params'] = None

        net = []
        for i in range(net_params['de_components']):
            net_ = Net_PU(net_params=net_params,train_params=train_params)
            net_.to(train_params['device'])
            net.append(net_)
        return net
            
    elif method == 'swag':
        net_params['n_output'] = n_output
        net_params['num_col'] = 20
        train_params['drop_bool'] = False
        train_params['drop_bool_ll'] = False
        train_params['loss_func'] = torch.nn.MSELoss(reduction='mean')
        train_params['update_start'] = train_params['num_epochs']/2
        train_params['loss_params'] = None
        
        net = Net_SWAG(net_params=net_params,train_params=train_params)
        net.to(train_params['device'])  
        return net

    elif method.startswith('concrete_dropout'):

        # dropout in all layers, standard mse
        net_params['n_output'] = n_output
        train_params['loss_func'] = concrete_dropout_loss  # regularization is added in train func

        l = 1e-3
        if n_datapoints < 800:
            tau = 2
        elif n_datapoints < 5000:
            tau = 0.5
        elif n_datapoints < 10000:
            tau = 0.05
        elif n_datapoints < 100000:
            tau = 0.02
        else:
            tau = 0.001

        wr = l ** 2. / (tau * n_datapoints)
        dr = 2. / (tau * n_datapoints)
        train_params['sml_loss_params'] = [wr, dr]  # weight_regularizer, drop_regularizer

        net = CDNet(net_params=net_params, train_params=train_params)
        net.to(train_params['device'])

        return net

    else:
        raise Exception("Given method %s is not recognized" % method)

def get_mu_output(net, inp, method, n_samples=200):
    
    if method in ['de', 'pu_de']:
        net_out = []
        for i in range(len(net)):
            if method == 'de':
                out = net[i](inp).detach().numpy()
            elif method == 'pu_de':
                out = net[i](inp)[:, 0, None].detach().numpy()
            net_out.append(out)
        return np.mean(net_out, axis=0)
        
    elif method in ['pu', 'mc_pu']:
        return net(inp)[:, 0, None].detach().numpy()
    elif method in ['vanilla']:
        return net(inp, drop_bool=False).detach().numpy()
    elif 'mc' in method:
        return np.mean([net(inp).detach().numpy() for _ in range(n_samples)], axis=0)
    else:
        raise Exception("Given method %s is not recognized." % method)
            
def train_network(net, data, train_params, method, epoch_callback=None):

    if method in ['de','pu_de']:  # de = deep ensembles; net is a list, train all networks in that list
        for i in range(len(net)):
            train_network(net[i],data=data,train_params=train_params,method='mc') # TODO epoch callback for those methods
            
    else:
        
        start_time = time.time()
        
        X_train, y_train = data
        batch_size = train_params['batch_size']
        batch_no = len(X_train) // batch_size

        optimizer = torch.optim.Adam(net.parameters(), lr=train_params['learning_rate'],
                                     weight_decay=train_params['weight_decay'])
        #optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
        loss_func = train_params['loss_func']
        
        running_loss = 0.0
        for epoch in range(train_params['num_epochs']):

            X_train, y_train = shuffle(X_train, y_train)


            start_time_2 = time.time()
            for i in range(batch_no):
                
                #with torch.autograd.set_detect_anomaly(True):
                start  = i * batch_size
                end    = start + batch_size
                inputs = torch.FloatTensor(X_train[start:end]).to(train_params['device'])

                if net.n_output == 1:
                    labels = torch.FloatTensor(y_train[start:end].flatten()).to(train_params['device'])
                    labels = torch.unsqueeze(labels,dim=1)
                else:
                    labels = torch.FloatTensor(y_train[start:end]).to(train_params['device'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if loss_func == new_exact_wasserstein_dropout_loss:
                    loss = new_exact_wasserstein_dropout_loss(net=net, data=[inputs,labels], loss_params=train_params['loss_params'])
                elif loss_func in [concrete_dropout_loss]:
                    loss = loss_func(net(inputs), labels, net.regularisation())
                else:
                    outputs = net(inputs)
                    loss    = loss_func(outputs,labels)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            if 'swa' in method and epoch >= train_params['update_start']:
                net.update()
                
            if epoch % 25 == 0:
                if epoch_callback is not None and callable(epoch_callback):
                    if (method != 'swag') or (method == 'swag' and net.update_count >= net.num_col):
                        epoch_callback(net, epoch=epoch)
            
            end_time_2 = time.time()

            if epoch % 100 == 0:
                end_time = time.time()
                print('Epoch {}'.format(epoch), "loss: ",running_loss, "took: %.5fs (exp. total time: %.5fs)" % (end_time-start_time, (end_time-start_time)*train_params['num_epochs']/100) )
                start_time = time.time()
            running_loss = 0.0
                
            
            