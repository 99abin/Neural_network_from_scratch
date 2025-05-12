# dataset
raw_dataset = [
    [155, 36],  
    [158, 37],
    [160, 38],
    [163, 38],
    [165, 39],
    [168, 40],  
    [170, 41],
    [173, 42],
    [175, 42], 
    [178, 43],
    [180, 44], 
    [183, 44], 
    [185, 45],
    [188, 45],  
    [190, 46],
    [193, 46], 
    [195, 47],
    [198, 47], 
    [200, 48]   
]

dataset = [[raw_dataset[i][0]/200, raw_dataset[i][1]] for i in range(len(raw_dataset))] # normalizing dataset

# weight and bias
w_hidden = [0.5, -0.5]
b_hidden = [0.1, -0.1]
w_output = [0.25, -0.25]
b_output = 0

# learning rate and epoch
learn_rate = 0.01
epoch = 10000

for train in range(epoch):
    for i in range(len(dataset)):
        # forward pass
        hidden_output = [w * dataset[i][0] + bias for w, bias in zip(w_hidden, b_hidden)]
        output = sum([i * w for i, w in zip(hidden_output, w_output)]) + b_output
        
        # backpropagation
        loss = (output - dataset[i][1]) ** 2
        
        d_loss_output = 2 * (output - dataset[i][1])
        d_output_hiddenO = w_output
        
        d_output_wO = [h for h in hidden_output]
        d_output_b3 = 1
        d_hiddenO_wH = dataset[i][0]
        
        # partial derivative for each weigh and bias
        d_loss_wH = [d_loss_output * d_output_hiddenO[p] * d_hiddenO_wH for p in range(2)]
        d_loss_bH = [d_loss_output * d_output_hiddenO[p] * 1 for p in range(2)]
        
        d_loss_wO = [d_loss_output * d_output_wO[p] for p in range(2)]
        d_loss_bO = d_loss_output * d_output_b3
        
        # update weight and bias
        w_hidden = [w_hidden[p] - learn_rate * d_loss_wH[p] for p in range(2)]
        b_hidden = [b_hidden[p] - learn_rate * d_loss_bH[p] for p in range(2)]
        
        w_output = [w_output[p] - learn_rate * d_loss_wO[p] for p in range(2)]
        b_output -= learn_rate * d_loss_bO

# check
height = float(input("Height: "))
hidden_output = [w * (height/200) + bias for w, bias in zip(w_hidden, b_hidden)]
output = sum([i * w for i, w in zip(hidden_output, w_output)]) + b_output
print(f"your shoe size is {output:.1f} (EU)")
