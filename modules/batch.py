import numpy as np

def expand_one_hot_vector(numbers,num_batch,out_put_size):  #(20,1),<class 'numpy.ndarray'>
    expanded_vector = np.zeros((num_batch, out_put_size))       #(20,784)
    batch_ones      = np.ones(num_batch)*(out_put_size // 10)   #リストみたいな挙動になっている。よくわからない。
    start_index     = numbers * batch_ones                                    #(20,)
    end_index       = start_index + batch_ones                                #(20,)
    for idx in range(num_batch):
        expanded_vector[idx][int(start_index[idx]):int(end_index[idx])] = 1  #(20,784)
    return expanded_vector.reshape(num_batch, out_put_size)

def reduct_pred(vector, num_batch,out_put_size):
    pred_list=[]
    for i in range(num_batch):
        reduct_pred  = []
        for _idx in range(10):
            start_index = int(_idx * int(out_put_size / 10))
            end_index   = start_index + int(out_put_size / 10)
            reduct_pred.append(np.sum(vector[i][start_index:end_index]))
        pred_list.append(np.argmax(reduct_pred))
    return np.array(pred_list)    #(20,)