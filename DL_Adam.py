'''
before this file is ran,you should run data_generation.py first to get data set.
It should have data_set_rastrigin.npy.

In this programm,we use a list to represent the network,e.g. nodes=[2,3,1]
which means a net work has 3 layers,the first is input layer with 2 neurons,the hidden layer has 3 neurons
and output layer has 1 neuron.

The other hyperparameters like batch-size,epoch number,iteration number,and learning rate alpha 
can be changed in the front lines of code.

here are some important functions:
    
training : use mini-bath to train.after training the loss value and parameter will be saved
    
test() : use saved parameter to calculate mse in test set ,the test mse will be saved and shown.

get_symbol():get symbol expression via CasADi,output symbol variable and output expression of network

adam_minibatch():implement adam algorithm
   
batch_error_gradient():calculate m ean gradient of batch size 
    
initialization_parameter():get initial parameter.output [w0,w1,...,wN-1,b0,...,bN-1]
    
get_gradient():calculate the gradient value.output [dy/dw0,dy/dw1,...,dy/db0,...,dy/dbN-1]
'''
import numpy as np
from casadi import *
import json #  or YAML , to save list data,which elements are arrays with different dimension
#np.random.seed(12345)
num_batch = 4
# batch size,better smaller than 16，16,10,8,5 are aviable.
#When 1 is chosen,it is SGD
alpha = 0.001  # initial leaning rate,0.002,0.001,0.0008 
#nodes2 = [2,25,23,7,12,15,1]
#nodes = [2,15,20,19,17,1] 
nodes = [2,18,8,12,15,14,1]
epoch = 500   # epoch number
max_iter = 100 # max iterations
N=len(nodes)
m=sum(nodes)  # total numbers of network
start_type = 'warm'
#first time use 'cold' to get initial parameter, then you can use 'cold' or 'warm' 
#'warm' means that you can use last trained parameter to start training,also called 'warm start' in most time

# choose activation function
activation_fun = 'relu'  #  relu,tanh,mix 
        
# load  data 
data = np.load('data_set_rastrigin.npy')

# standaration of data 
mean = np.mean(data,axis=0)
var = np.var(data,axis=0)
std_var = var**0.5
norm_data = (data-mean)/std_var

# spilt data into training data and test data
spilt_factor = 0.9       # 90% of data for training
num_data = len(data)
norm_num = int(spilt_factor*num_data)
train_data = norm_data[0:norm_num]   
test_data = norm_data[norm_num:]

# set CasADi function
x = SX.sym("x")
tanh = Function('tanh',[x],[exp(x)/(1+exp(x))])
relu = Function('relu',[x],[fmax(x,0)])

def main():
    
    # start training and save data
    #training(nodes,epoch,num_batch,train_data,start_type)
    np.save('num_epoch',epoch) 
    # test model and show the result
    test(nodes,test_data)
    return 0
  
def training(nodes,epoch,num_batch,train_data,start_type): 
    
    if start_type == 'warm' :
        parameter = json.load(open('parameter_'+activation_fun+'_'+str(alpha)+'_'+str(m)+'.json'))   # parameter[0]=nodes
        parameter = parameter[1:]
        parameter = [np.array(parameter[i]) for i in range(len(parameter))]
    elif start_type == 'cold' :
        parameter = initialization_parameter(nodes)
        
    loss_hist = []
    for e in range(epoch):
        # shuffle data to get different data for each epoch
        np.random.shuffle(train_data)
        parameter,loss = adam_minibatch(train_data,nodes,num_batch,parameter,alpha)
        
        np.savetxt('loss_'+activation_fun+'_'+str(alpha)+'_'+str(m)+'.txt',loss,delimiter=',')
        loss_hist.append(loss[-1])
        ## after training,save the parameter include nodes info
        
    wb=[nodes]
    for i in range(len(parameter)):
        wb.append(parameter[i].tolist())  
    savepath = 'parameter_'+activation_fun+'_'+str(alpha)+'_'+str(m)+'.json'
    with open(savepath,'w') as f:
        f.write(json.dumps(wb))   
    np.savetxt('lossvalue.txt',loss_hist,delimiter=',')

         
def test(nodes,test_data):
      
    # load trained parameter and translate to array
    parameter = json.load(open('parameter_'+activation_fun+'_'+str(alpha)+'_'+str(m)+'.json'))   # parameter[0]=nodes
    print('nodes info ist:',parameter[0])
    parameter = parameter[1:]
    parameter = [np.array(parameter[i]) for i in range(len(parameter))]
    print('the last bias is:',parameter[-1])
    vari_name,vari,expr=get_symbol(nodes,activation_fun)         # vari=[input,w0,w1,...,wN-2,b0,b1,...bN-2],len=2*N-1
    forward = Function('forward',vari,[expr])  
    # calculate the  mean sq loss
    mean_loss=0
    
    for i in range(len(test_data)):   
        net_in = [test_data[i,0:2]]+parameter
        net_out = forward.call(net_in)[0]
        if i<5:
            print(test_data[i])
            print(net_out)
        mean_loss += (net_out-test_data[i,-1])**2/len(test_data)
        
    np.savetxt('test_mse_'+activation_fun+'_'+str(alpha)+'_'+str(m)+'.txt',mean_loss)
    print('mean squared loss in test set is :',mean_loss)
         

def adam_minibatch(train_data,nodes,num_batch,parameter,alpha,beta1=0.9,beta2=0.999,epsilon=10e-8):

    # set constant factor
    m=[]    # 1st moment vector
    v=[]    # 2nd moment vector
    m_hat=[]
    v_hat=[]
    N=len(nodes)
    for i in range(2*N-2):
        m.append(0) 
        v.append(0) 
        m_hat.append(0)
        v_hat.append(0)
    t=0     # time step
    
    loss,batch_g = batch_error_gradient(train_data,parameter,nodes,num_batch)
    loss_hist=[loss]
    #vari_name,vari,expr=get_symbol(nodes,activation_fun)         # vari=[input,w0,w1,...,wN-2,b0,b1,...bN-2],len=2*N-1
    #forward = Function('forward',vari,[expr])  
    #while(loss>10):
    for j in range(max_iter):         
        t = t+1       
        for i in range(2*N-2):
             
            m[i] = beta1*m[i]+(1-beta1)*batch_g[i][0]
            v[i] = beta2*v[i]+(1-beta2)*(batch_g[i][0]**2)
            m_hat[i] = m[i]/(1-beta1**t)
            v_hat[i] = v[i]/(1-beta2**t)
            parameter[i] = parameter[i]-alpha*m_hat[i]/(v_hat[i]**0.5+epsilon)
        
        loss,batch_g= batch_error_gradient(train_data,parameter,nodes,num_batch)
        
        loss_hist.append(loss) 
        if (loss_hist[-1]==loss_hist[-2]):
            print('parameter equals to zero,epoch training stop')
            break
        if (loss_hist[-1]>5000*min(loss_hist)):
            print('the loss value start rising strongly,epoch training stop early')
            break
            
    return parameter,loss_hist

def batch_error_gradient(train_data,parameter,nodes,num_batch):
    
    error = 0     # error between estimation and traing out
    loss = 0 
    batch_g=[0 for i in range(2*N-2)]
    vari_name,vari,expr=get_symbol(nodes,activation_fun)         # vari=[input,w0,w1,...,wN-2,b0,b1,...bN-2],len=2*N-1
    forward = Function('forward',vari,[expr])  
    for b in range(num_batch):
            train_in = train_data[b,0:2]   
            train_out = train_data[b,-1]
            net_in = [train_in]+parameter
            net_out = forward.call(net_in)   
            error += np.array(net_out-train_out)
            g = get_gradient(net_in)
            batch_g = [(g[m]*error*2+batch_g[m])/num_batch for m in range(2*N-2)]      
            loss += error**2/num_batch
    
    return loss,batch_g
    
def initialization_parameter(nodes=[]):
    # in this case ,train_in should be a list with a 1*2 array element 
    initial_weights=[]
    initial_bias=[]
    N=len(nodes)

    for i in range(N-1):    
        initial_weights.append(np.random.normal(0,pow(nodes[i],0),(nodes[i],nodes[i+1])))  
        initial_bias.append(np.zeros((1,nodes[i+1]))) 
        
    parameter=initial_weights+initial_bias
    return parameter  # [Weight,Bias] is a list

def get_symbol(nodes,activation_fun):
    
    weights_name = []
    bias_name = []
    weights = []
    bias = []
    
    for i in range(N-1):  
        weights_name.append('w'+str(i))     # string:w0,w1,...,w_N-2, for N layers we need N-1 weights   
        bias_name.append('b'+str(i))        # string:b0,b1,...b_N-2  
        weights.append(SX.sym(weights_name[i],nodes[i],nodes[i+1]))     # symbol weights=[w0,w1,w2,...]  
        bias.append(SX.sym(bias_name[i],1,nodes[i+1]))      # symbol bias=[b0,b1,b2,...]
        
    input = SX.sym('input',(1,2))     
    vari = [input]+weights+bias     # vari=[input,w0,w1,...,wN-2,b0,b1,...bN-2],len=2*N-1
    vari_name = ['input']+weights_name+bias_name
    
    # build symbol function expression for forward process
    expr = input    #input[0]
    # according to the activation fun ,get different expressions
    if activation_fun == 'relu':
        for i in range(N-2):
            expr = relu(expr@weights[i]+bias[i])
    elif activation_fun == 'tanh':
        for i in range(N-2):
            expr = tanh(expr@weights[i]+bias[i])
    elif activation_fun == 'mix':
        for i in range(N-2):
            if i%2==0:
                expr = relu(expr@weights[i]+bias[i])
            else :
                expr = tanh(expr@weights[i]+bias[i])
    
    expr = expr@weights[-1]+bias[-1]  # the last layer doesn't use activation function
    
    return vari_name,vari,expr
#now we define the symbol expression for gradient w.r.t weights and bias  
def get_gradient(net_in):
    
    
    vari_name,vari,expr=get_symbol(nodes,activation_fun) 
    gradient=[]     
    gradient_value=[]
    for i in range(2*N-2): # gradient=[dy/dw0,...,dy/dwN-2,dy/db0,...dy/dbN-2]
        gradient.append(Function(vari_name[i+1],vari,[jacobian(expr,vari[i+1])]))
        gradient_value.append(gradient[i].call(net_in))
    for i in range(N-1):
        gradient_value[i]=np.array(gradient_value[i][0])
        gradient_value[i]=gradient_value[i][0].reshape(nodes[i],nodes[i+1])    # reshape weights
        gradient_value[i+N-1]=np.array(gradient_value[N+i-1][0])
        gradient_value[i+N-1]=gradient_value[N+i-1][0].reshape(1,nodes[i+1])   # reshape bias
        
    return gradient_value  # gradient=[dy/dw0,...,dy/db0,...]

     
if __name__ == "__main__":
    
   main()

