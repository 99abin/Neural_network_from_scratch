import math

# softplus for the activation function
def softplus(x):
    return math.log(1 + math.exp(x))
    
def d_softplus(x):
    return 1 / (1 + math.exp(-x))
    
# dataset
dataset = [
    [0.0, 0.01],    
    [0.5, 0.08],   
    [1.0, 0.20],   
    [1.5, 0.35],   
    [2.0, 0.52],   
    [2.5, 0.70],  
    [3.0, 0.85],   
    [3.5, 0.92],    
    [4.0, 0.90],  
    [4.5, 0.83],   
    [5.0, 0.72],   
    [5.5, 0.60],   
    [6.0, 0.45],  
    [6.5, 0.30],   
    [7.0, 0.18],   
    [8.0, 0.10],
    [9.0, 0.05],  
    [10.0, 0.02]   
]

# weight and bias
w_hidden = [0.5, -0.5]
b_hidden = [0.1, -0.1]
w_output = [0.25, -0.25]
b_output = 0

# learning rate and epoch
learn_rate = 0.1
epoch = 10000

for train in range(epoch):
    for i in range(len(dataset)):
        # forward pass
        hidden_raw = [w * dataset[i][0] + bias for w, bias in zip(w_hidden, b_hidden)]
        hidden_output = [softplus(z) for z in hidden_raw]
        
        output_raw = sum([i * w for i, w in zip(hidden_output, w_output)]) + b_output
        output = softplus(output_raw)
        
        # backpropagation
        loss = (output - dataset[i][1]) ** 2
        
        d_loss_Oraw = 2 * (output - dataset[i][1]) * d_softplus(output_raw)
        d_Oraw_hiddenO = w_output
        d_hiddenO_hiddenR = [d_softplus(z) for z in hidden_raw]
        
        d_Oraw_wO = [h for h in hidden_output]
        d_Oraw_b3 = 1
        d_hiddenR_wH = dataset[i][0]
        
        # partial derivative for each weigh and bias
        d_loss_wH = [d_loss_Oraw * d_Oraw_hiddenO[p] * d_hiddenO_hiddenR[p] * d_hiddenR_wH for p in range(2)]
        d_loss_bH = [d_loss_Oraw * d_Oraw_hiddenO[p] * d_hiddenO_hiddenR[p] * 1 for p in range(2)]
        
        d_loss_wO = [d_loss_Oraw * d_Oraw_wO[p] for p in range(2)]
        d_loss_bO = d_loss_Oraw * d_Oraw_b3
        
        # update weight and bias
        w_hidden = [w_hidden[p] - learn_rate * d_loss_wH[p] for p in range(2)]
        b_hidden = [b_hidden[p] - learn_rate * d_loss_bH[p] for p in range(2)]
        
        w_output = [w_output[p] - learn_rate * d_loss_wO[p] for p in range(2)]
        b_output -= learn_rate * d_loss_bO

# check
dosage = float(input('input a dosage in gram: '))
hidden_raw = [w * dosage + bias for w, bias in zip(w_hidden, b_hidden)]
hidden_output = [softplus(z) for z in hidden_raw]
        
output_raw = sum([i * w for i, w in zip(hidden_output, w_output)]) + b_output
output = softplus(output_raw)
percentage = str(round(output * 100, 2)) + '%'

print(f"the efectiveness is {percentage}")
