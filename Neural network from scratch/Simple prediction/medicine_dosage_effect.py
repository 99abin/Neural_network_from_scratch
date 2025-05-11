import math

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

for i in range(len(dataset)):
    # forward pass
    hidden_raw = [w * dataset[i][0] + bias for w, bias in zip(w_hidden, b_hidden)]
    hidden_output = [softplus(z) for z in hidden_raw]
    
    output_raw = sum([i * w for i, w in zip(hidden_output, w_output)]) + b_output
    output = softplus(output_raw)
    
    # backpropagation
    loss = (output - dataset[i][1]) ** 2
    d_loss_output = 2 * (output - dataset[i][1])
    d_output_Oraw = d_softplus(output_raw)
    
    d_loss_Oraw = d_loss_output * d_output_Oraw
    
