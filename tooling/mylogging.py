"""
Parse logging 
"""

def yolov4tiny_parse(path, smooth_training=True):
    train_loss = []
    train_steps = []
    val_loss = []
    val_steps = []

    smooth_samples = 20
    val_count = 900

    # Using readlines()
    file0 = open(path, 'r')
    lines = file0.readlines()
    
    for cnt, line in enumerate(lines):

        if 'mean_average_precision' in line:
            val_count = val_count + 100
            val_steps.append(int(val_count))
            val_loss.append(float(line[38:]))

        elif 'avg loss' in line:
            train_steps.append(int(line[1:line.find(':')]))
            end_ind = line.find('avg loss')-1
            start_ind = line[:end_ind].rfind(',')+2
            train_loss.append(float(line[start_ind:end_ind]))

    if smooth_training == True:
        
        train_steps = [train_step for train_step in train_steps if train_step%smooth_samples==0]
        train_loss = [sum(train_loss[int(train_step-1):int(train_step-1)+smooth_samples])/smooth_samples for train_step in train_steps]

    train_steps = train_steps[:-1]
    train_loss = train_loss[:-1]

    return train_steps, train_loss, val_steps, val_loss
    

def nanodet_parse(path, smooth_training=True):
    train_loss = []
    train_steps = []
    val_loss = []
    val_steps = []

    smooth_samples = 20

    # Using readlines()
    file0 = open(path, 'r')
    lines = file0.readlines()
    
    for cnt, line in enumerate(lines):

        if 'loss_bbox' in line:

            if 'Train' in line:
                train_steps.append(val_extract(line, 'Epoch', '/'))
                train_loss.append(val_extract(line, 'loss_bbox:', '|'))
            
            elif 'Val' in line:
                val_steps.append(val_extract(line, 'Epoch', '/'))
                val_loss.append(val_extract(line, 'loss_bbox:', '|'))

    if smooth_training == True:
        
        train_steps = [train_step for train_step in train_steps if train_step%smooth_samples==0]
        train_loss = [sum(train_loss[int(train_step-1):int(train_step-1)+smooth_samples])/smooth_samples for train_step in train_steps]

    return train_steps, train_loss, val_steps, val_loss
    
def val_extract(line, key, end):
    idx = line.index(key)
    start_num = idx + len(key)
    end_num = start_num + line[start_num:].find(end)
    return float(line[start_num:end_num])
