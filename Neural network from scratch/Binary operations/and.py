def sigmoid(z):
    return 1/(1 + 2.71828 ** (-z))

def d_sigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))

data = [
    [(0, 0), 0],
    [(1, 0), 0],
    [(0, 1), 0],
    [(1, 1), 1]
]

inputs = [a for a, b in data]

w_output =[
    [0.5],
    [0.5]
]

b_output = 0.5
learn_rate = 0.1
epoch = 10000

for train in range(epoch):
    for i in range(len(inputs)):
        z = sum(a*b for a, b in zip(inputs[i], [w[0] for w in w_output])) + b_output
        
        output = sigmoid(z)
        loss = (output - data[i][1]) ** 2
        
        d_loss_output = 2 * (sigmoid(z) - data[i][1])
        d_output_z = d_sigmoid(z)
        d_z_w1 = inputs[i][0]
        d_z_w2 = inputs[i][1]
        d_z_b = 1
        
        d_loss_w1 = d_loss_output * d_output_z * d_z_w1
        d_loss_w2 = d_loss_output * d_output_z * d_z_w2
        d_loss_b = d_loss_output * d_output_z * d_z_b
        
        w_output[0][0] = w_output[0][0] - learn_rate * d_loss_w1
        w_output[1][0] = w_output[1][0] - learn_rate * d_loss_w2
        b_output = b_output - learn_rate * d_loss_b
 

for i in range(len(inputs)):
        z = sum(a*b for a, b in zip(inputs[i], [w[0] for w in w_output])) + b_output
        
        output = sigmoid(z)
        print(f'{inputs[i]} =  {output} = {round(output)}')
