import random

def sigmoid(z):
    return 1 / (1 + 2.7182818 ** (-z))

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

data = [
    [(0, 0), 0],
    [(1, 0), 1],
    [(0, 1), 1],
    [(1, 1), 0]
]

inputs = [a for a, b in data]

w_hidden =[[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
b_hidden = [random.uniform(-1, 1) for _ in range(2)]
w_output = [random.uniform(-1, 1) for _ in range(2)]
b_output = random.uniform(-1, 1)

learn_rate = 0.1
epoch = 10000

for train in range(epoch):
    for i in range(len(inputs)):
        #forward pass
        hidden_raw = [sum(a * b for a, b in zip(w_row, inputs[i])) + bias for w_row, bias in zip(w_hidden, b_hidden)]
        hidden_output = [sigmoid(z) for z in hidden_raw]
        
        output_raw = sum(a * b for a, b in zip(hidden_output, w_output)) + b_output
        output = sigmoid(output_raw)
        
        #back propagation 
        loss = (output - data[i][1]) ** 2
        d_loss_Oraw = 2 * (output - data[i][1]) * d_sigmoid(output_raw)
        
        d_loss_w_output = [d_loss_Oraw * a for a in hidden_output]
        
        d_loss_w_hidden = [[0 for _ in range(2)], [0 for _ in range(2)]]
        
        for x in range(2):
            for y in range(2):
                d_loss_w_hidden[x][y] = d_loss_Oraw * w_output[x] * d_sigmoid(hidden_raw[x]) * inputs[i][y]
                
        d_loss_b_hidden = [d_loss_Oraw * w_output[i] * d_sigmoid(hidden_raw[i]) for i in range(2)]
        
        d_loss_b_output = d_loss_Oraw * 1
        
        # update weight
        for x in range(2):
            for y in range(2):
                w_hidden[x][y] -= learn_rate * d_loss_w_hidden[x][y]
        
        for i in range(len(w_output)):
            w_output[i] -= learn_rate * d_loss_w_output[i]
            
        # update bias
        b_hidden = [b_hidden[a] - learn_rate * d_loss_b_hidden[a] for a in range(2)]
        
        b_output -= learn_rate * d_loss_b_output
        
#check
for i in range(len(inputs)):
        hidden_raw = [sum(a * b for a, b in zip(w_row, inputs[i])) + bias for w_row, bias in zip(w_hidden, b_hidden)]
        hidden_output = [sigmoid(z) for z in hidden_raw]
        
        output_raw = sum(a * b for a, b in zip([x for x in hidden_output], [y for y in w_output])) + b_output
        output = sigmoid(output_raw)
        
        print(f'{inputs[i]}\t = \t {output:<10.4f} = \t {round(output)}')
