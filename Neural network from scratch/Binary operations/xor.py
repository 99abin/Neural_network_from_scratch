def sigmoid(z):
    return 1/(1 + 2.71828 ** (-z))

def d_sigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))

data = [
    [(0, 0), 0],
    [(1, 0), 1],
    [(0, 1), 1],
    [(1, 1), 0]
]

inputs = [a for a, b in data]

w_hidden =[
    [0.5, 0.5],
    [0.5, 0.5]
]

b_hidden = [0.5, 0.5]

w_output = 0.5
b_output = 0.5
learn_rate = 0.1
epoch = 10000

for i in range(len(inputs)):
        hidden_raw = [sum(a * b for a, b in zip(w_row, inputs[i])) + bias for w_row, bias in zip(w_hidden, b_hidden)]
        output_hidden = sigmoid(sum(hidden_raw))
        
        
