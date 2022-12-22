#!/usr/bin/env python3


##Imports
import torch # arrays on GPU
from torchsummary import summary

import torch.autograd as autograd #build a computational graph
import torch.nn as nn ## neural net library
import torch.nn.functional as F ## most non-linearities are here
import torch.optim as optim # optimization package

import torchvision.models as models
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import pathlib
import cv2
import os
import gc


import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config as args
from img_utils import *
from colorama import Fore, Back, Style

WIDTH = 128
HEIGHT = 106

ACT_SIZE = 4
ACT_FET = 20
LSTM_IN_SIZE = 80
HIDDEN_SIZE = 80
COM_OUT = 2
SIM_SIZE = 3

chan_layer_1 = 16
chan_layer_2 = 16
chan_layer_3 = 32
chan_layer_4 = 32
pool_size = 3

''' Dimension of input image'''
W = 128.0 ##!!!! Important to make it float to prevent integer division becomes zeros
H = 106.0

#Training Params
epochs =50
num_action_batch = args.num_action / args.batch_size#None
hidden = None

sim_out_gt =np.random.rand(2,)
com_out_gt =np.random.rand(2,)

#saving path
cwd = str(pathlib.Path(__file__).parent.resolve())
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cwd = os.path.join(BASE_DIR, 'PushNet/trained_models/')
save_path = cwd +"push_net_model.pth"

def to_var(x, volatile=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


''' CNN Module '''
class COM_CNN(nn.Module):
    def __init__(self):
        super(COM_CNN, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=chan_layer_1,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(chan_layer_1, chan_layer_2, 3, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(chan_layer_2, chan_layer_3, 3, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(chan_layer_3, chan_layer_4, 3, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2.0))

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return x

''' Push Net '''
class COM_net_sim(nn.Module):
    def __init__(self, batch_size):
        super(COM_net_sim, self).__init__()
        self.cnn = COM_CNN()
        f_len = self.get_img_feature_len()
        self.batch_size = batch_size
        self.linear_act = nn.Linear(ACT_SIZE, ACT_FET)
        self.linear_act_img_curr = nn.Linear(f_len + ACT_FET,  LSTM_IN_SIZE)

        self.linear_img_img = nn.Linear(f_len, SIM_SIZE)
        self.lstm = nn.LSTM(LSTM_IN_SIZE, HIDDEN_SIZE, 1, batch_first=True)

        self.linear_fnext_out = nn.Linear(HIDDEN_SIZE, f_len)
        self.linear_com_out = nn.Linear(HIDDEN_SIZE, COM_OUT)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        ### two variables: (h0, c0)
        return (autograd.Variable(torch.zeros(1, self.batch_size,
            HIDDEN_SIZE).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size,
                    HIDDEN_SIZE).cuda()))

    def get_img_feature_len(self):
        test = torch.rand(1, 1, WIDTH, HEIGHT)
        return self.cnn(Variable(test)).size()[1]

    def forward(self, a0, I1, a1, Ig, lengths, bs):
        ''' get image and action feature representation'''

        ''' flatten mini-batch sequence of images'''
        f1 = self.cnn(I1.view(-1, 1, 106, 128))
        fg = self.cnn(Ig.view(-1, 1, 106, 128))
        ''' flatten mini-batch sequence of action'''
        fa1 = self.linear_act(a1.view(-1, 4))


        ''' combine img and previous action feature to form one-step history'''
        cat_f1_fa1 = torch.cat((f1, fa1), 1)


        lstm_inp = self.linear_act_img_curr(cat_f1_fa1)
        
        ''' pack sequence to feed LSTM '''
        lstm_inp = pack_padded_sequence(lstm_inp.view(bs, -1, LSTM_IN_SIZE),
                lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)

        ''' unpack sequence to feed in linear layer'''
        lstm_out_unpad, lengths_new = pad_packed_sequence(lstm_out, batch_first=True)

        com_out = self.linear_com_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        fnext = self.linear_fnext_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))

        ''' evaluate similarity between target internal state & true internal state'''
        output = self.linear_img_img(torch.abs(fnext - fg))

        ''' squash value between (0, 1)'''
        
        sim = F.sigmoid(output)
        com_out = F.sigmoid(com_out)

        ''' pack into sequence for output'''
        sim_pack = pack_padded_sequence(sim.view(bs, -1, SIM_SIZE), lengths, batch_first=True)
        com_pack = pack_padded_sequence(com_out.view(bs, -1, 2), lengths, batch_first=True)
        return sim_pack, com_pack


class PushController:
    def __init__(self):
        self.num_action = args.num_action
        self.bs = args.batch_size
        ''' goal specification '''
        self.w = 30 # orientation in degree (positive in counter clockwise direction)
        self.x = 10 # x direction translation in pixel (horizontal axis of image plane)
        self.y = -10 # y direction translation in pixel (vertical axis of image plane)

        ## instantiate Push-Net predictor
        self.Ic = cv2.imread('test.jpg')[:,:,0]


    def sample_action(self, img, num_actions):
        ''' sample [num_actions] numbers of push action candidates from current img'''
        s = 0.9
        safe_margin = 1.4
        out_margin = 2.0
        

        ## get indices of end push points inside object mask
        img_inner = cv2.resize(img.copy(), (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img_inner.shape
        img_end = np.zeros((int(H), int(W)))
        img_end[(int(H)-h)//2:(int(H)+h)//2, (int(W)-w)//2:(int(W)+w)//2] = img_inner.copy()
        (inside_y, inside_x) = np.where(img_end.copy()>0)

        ## get indices of start push points outside a safe margin of object
        img_outer1 = cv2.resize(img.copy(), (0,0), fx=safe_margin, fy=safe_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer1.shape
        img_start_safe = np.zeros((int(H), int(W)))
        img_start_safe = img_outer1.copy()[(h-int(H))//2:(h+int(H))//2, (w-int(W))//2:(w+int(W))//2]

        img_outer2 = cv2.resize(img.copy(), (0,0), fx=out_margin, fy=out_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer2.shape
        img_start_out = np.zeros((int(H), int(W)))
        img_start_out = img_outer2.copy()[(h-int(H))//2:(h+int(H))//2, (w-int(W))//2:(w+int(W))//2]

        img_start = img_start_out.copy() - img_start_safe.copy()
        (outside_y, outside_x) = np.where(img_start.copy()>100)

        num_inside = len(inside_x)
        num_outside = len(outside_x)

        actions = []
        for i in range(num_actions):
            start_x = 0
            start_y = 0
            end_x = 0
            end_y = 0
            while True:
                ## sample an inside point
                inside_idx = np.random.choice(num_inside)
                ## sample an outside point
                outside_idx = np.random.choice(num_outside)
                end_x = int(inside_x[inside_idx])
                end_y = int(inside_y[inside_idx])
                start_x = int(outside_x[outside_idx])
                start_y = int(outside_y[outside_idx])

                if start_x < 0 or start_x >= W or start_y < 0 or start_y >= H:
                    print('out of bound')
                    continue
                if img[start_y, start_x] == 0:
                    break
                else:
                    continue

            actions.append(start_x)
            actions.append(start_y)
            actions.append(end_x)
            actions.append(end_y)

        return actions



def evaluate_minibatch(model, img_curr, img_goal, actions, bs):
        ''' calculate the similarity score of actions '''
        bs = bs
        A1 = []
        I1 = []
        Ig = []

        for i in range(bs):
            a1 = [[actions[4*i]/W, actions[4*i+1]/H, actions[4*i+2]/W, actions[4*i+3]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        sim_out = None
        com_out = None

        sim_out, com_out = model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)

        sim_np = sim_out.data.cpu().data.numpy()
        com_np = com_out.data.cpu().data.numpy()

        sim_sum = np.sum(sim_np, 1) # measure (w ,x, y)
        com_sum = np.sum(com_np, 1)
        sim_sum = torch.tensor(sim_sum, requires_grad = True).float()
        com_sum = torch.tensor(com_sum, requires_grad = True).float()
       
        return sim_sum, com_sum

#NOTE: Akash added
####
def data_preparation(img_curr, img_goal, actions, bs):
        ''' calculate the similarity score of actions '''
        bs = bs
        A1 = []
        I1 = []
        Ig = []

        for i in range(bs):
            a1 = [[actions[4*i]/W, actions[4*i+1]/H, actions[4*i+2]/W, actions[4*i+3]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float().cuda()
        I1 = torch.from_numpy(np.array(I1)).float().div(255).cuda()
        Ig = torch.from_numpy(np.array(Ig)).float().div(255).cuda()

        #      "Input image>>>>",I1, "\n",
        #      "goal image>>>>>",Ig, "\n")
        return A1, I1, A1, Ig, [1 for j in range(bs)], bs

####





def train_model(train_dl, model):
    '''
    Train the model
    '''
    bs = args.batch_size
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        for i in range(int(num_action_batch)):
            # optimizer.zero_grad()
            ## keep hidden state the same for all action batches during selection
            if not hidden != None:
                model.hidden = model.init_hidden()
            action = train_dl[-1][4*i*bs: 4*(i+1)*bs]
            prepared_data = data_preparation( train_dl[0], train_dl[1], action, bs =bs)
            sim_out, com_out = model(prepared_data[0], #actionsrandom
                                    prepared_data[1], #input image
                                    prepared_data[2], ##actions
                                    prepared_data[3], #goal image
                                    prepared_data[4], #number of actions: for LSTM module
                                    prepared_data[5]) #batchsize
            sim_np = sim_out.data.cpu().data.numpy()
            com_np = com_out.data.cpu().data.numpy()
           
            
            sim_sum = np.sum(sim_np, 1) # measure (w ,x, y)
            com_sum = np.sum(com_np, 1)
            sim_score = torch.tensor(sim_sum, requires_grad = True).float().cuda()
            com_score = torch.tensor(com_sum, requires_grad = True).float().cuda()
            loss_sim = criterion(sim_out.data.sum(1).float(), torch.from_numpy(sim_out_gt).float().cuda())
            loss_com = criterion(com_out.data.sum(1).float(), torch.from_numpy(com_out_gt).float().cuda())
            #NOTE: Modify constant according to simulator values
            loss = 0.1*loss_com +((0.1*loss_sim)/(1+0.1*loss_com))
            print(
						Fore.GREEN +
						'\n-------------------------------------\n' +
                        '|Epoch: '+str(epoch+1) +' | '+'Minibatch:' +str(i+1)+ '|'
						'| loss ' + str(loss.item()) + ' |\n' +
						'---------------------------------------\n' +
						Style.RESET_ALL
					)
            loss.backward()
            # loss.item()
            optimizer.step()
            # gc.collect()
            # torch.cuda.empty_cache()

    torch.save(model.state_dict(), save_path)




if __name__ == "__main__":
    img = cv2.imread("test.jpg")[:,:,0]
    img_in_next = generate_goal_img(img.copy(), 30, 10, -10)
    actions = PushController()
    actions = actions.sample_action(img=img, num_actions=args.num_action)
    dl_train = [img, img_in_next, actions]
    dev=torch.device("cuda")
    train_model(train_dl=dl_train, model=COM_net_sim(args.batch_size).to(dev))