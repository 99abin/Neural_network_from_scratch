import math

# ReLu for the hidden layer activation function
def relu(x):
    return max(0, x)

def d_relu(x):
    return 1 if x > 0 else 0
    
# softmax for the output layer activation function
def softmax(p, x, y, z):
    return (math.exp(p)) / ((math.exp(x)) + (math.exp(y)) + (math.exp(z)))

# cross entrophy for calculating loss
def c_entrophy(t, y):
    return -t * math.log(y)

# dataset
cerah = [1, 0, 0]
berawan = [0, 1, 0]
hujan = [0, 0, 1]

raw_dataset = [
    [30, 20, cerah],
    [28, 25, cerah],
    [25, 40, berawan],
    [22, 60, berawan],
    [18, 80, hujan],
    [15, 85, hujan],
    [32, 15, cerah],
    [20, 70, berawan],
    [12, 90, hujan],
    [26, 35, berawan],
    [24, 50, berawan],
    [17, 75, hujan],
    [29, 30, cerah],
    [21, 65, berawan],
    [14, 95, hujan]
]

inputs = [[data[0]/32, data[1]/95] for data in raw_dataset]
outputs = [n[2] for n in raw_dataset]

# weight and bias
w_hidden = [
    [0.5, -0.25],
    [-0.5, 0.25]
]
b_hidden = [-0.1, 0.1]
w_output = [
    [0.75, -0.75],
    [-0.5, 0.5],
    [0.25, -0.25]
]
b_output = [1, 0, -1]

epoch = 10000

# training
for i in range(len(raw_dataset)):
    # forward pass
    hidden_raw = [sum(x * y for x, y in zip(inputs[i], weight)) + bias for weight, bias in zip(w_hidden, b_hidden)]
    hidden_output = [relu(hidden_raw[0]), relu(hidden_raw[1])]
    
    output_raw = [sum(x * y for x, y in zip(hidden_output, weight)) + bias for weight, bias in zip(w_output, b_output)]
    output = [softmax(p, output_raw[0], output_raw[1], output_raw[2]) for p in output_raw]
    
    # backpropagation
    crossE = sum(c_entrophy(a, b) for a, b in zip(outputs[i], output))
    
    d_CE_output = -1 / sum(a * b for a, b in zip(outputs[i], output))
    d_output_rOutput = sum(a * b for a, b in zip(outputs[i], output))
    # d_rOutput_wOutput = 
    # d_wOutput_wHidden = 
    
    print(d_CE_output)
