import tensorflow as tf
import numpy as np
import pickle
import os
import cv2

BATCH_SIZE = 200

model = tf.keras.models.load_model('new_model.h5')

def find_l2_norm(a,b):
    '''
    a : 2D numpy array
    b : 2D numpy array
    returns the L2_norm between each vector of a and each vector of b
    if a : 4 x 256 and b : 7 x 256
       output is 4 x 7 norm matrix

     '''
    try:
        a = a.reshape([1,256])
    except:
        pass
    try:
        b = b.reshape([1,256])
    except:
        pass
    dot_product = np.matmul(a, np.transpose(b))
    a = np.square(a)
    b = np.square(b)
    norm = np.sum((np.expand_dims(a,axis=1) + b), axis=2) - 2*dot_product
    norm = np.sqrt(norm)
    return norm

def stats(correct,incorrect):
    correct = np.around(correct,2)
    incorrect = np.around(incorrect,2)
    mean_correct,mean_incorrect = np.mean(correct),np.mean(incorrect)
    md_correct,md_incorrect = np.median(correct),np.median(incorrect)
    sd_correct,sd_incorrect = np.std(correct),np.std(incorrect)
    min_correct,max_correct = np.min(correct),np.max(correct)
    min_incorrect,max_incorrect = np.min(incorrect),np.max(incorrect)

    print('\n\n')
    print('Correct distance')
    print('mean_correct : ',mean_correct)
    print('sd_correct : ',sd_correct)
    print('md_correct : ',md_correct)
    print('min : ',min_correct, '  ','max : ',max_correct)
    print('val : ', (np.sum(np.cast['int'](correct<0.33)) / np.sum(np.cast['int'](correct>0.0)))*100 )

    print('\n\n')
    print('Incorrect distance')
    print('mean_incorrect : ',mean_incorrect)
    print('sd_incorrect : ',sd_incorrect)
    print('md_incorrect : ',md_incorrect)
    print('min : ',min_incorrect, '  ','max : ',max_incorrect)
    print('\n\n')



def market_test():

    '''
    query_dir : path to the directory containing query images
    test_dir : path to the directory containing test images

    output : ouputs the rank-1 accuracy
    
    '''
    file=open('test_dataset.pkl','rb')
    test_mat,test_imgs,query_mat,query_imgs = pickle.load(file)
    test_embed=get_embeddings(test_mat)
    query_embed= get_embeddings(query_mat)
    test_embed = np.array(test_embed)
    query_embed = np.array(query_embed)

    #counts the number of correct query images identified
    correct  = 0
    #counts the total number of query images
    tot = 0 
    #keeps track of query embeddings to be tested
    counter = 0
    #number of embeddings which are tested at once
    batch_size = 500

    correct_dist=[]
    incorrect_dist=[]

    tot_num = query_embed.shape[0]

    while(counter+batch_size < tot_num):
        distances = find_l2_norm(query_embed[counter:counter+batch_size],test_embed)
        imgs = query_imgs[counter:counter+batch_size]
        for i in range(distances.shape[0]):
            dis = distances[i].reshape([-1])
            count = zip(dis,test_imgs)
            count2 = sorted(count,key=lambda x:x[0])
            if int(imgs[i].split('_')[0]) == int(count2[0][1].split('_')[0]):
                correct += 1
                correct_dist.append(count2[0][0])
            else:
                incorrect_dist.append(count2[0][0])
            tot += 1
        counter += batch_size
    
    distances = find_l2_norm(query_embed[counter:tot_num],test_embed)
    imgs = query_imgs[counter:tot_num]
    for i in range(distances.shape[0]):
        dis = distances[i].reshape([-1])
        count = zip(dis,test_imgs)
        count2 = sorted(count,key=lambda x:x[0])
        if int(imgs[i].split('_')[0]) == int(count2[0][1].split('_')[0]):
            correct += 1
            correct_dist.append(count2[0][0])
        else:
            incorrect_dist.append(count2[0][0])
        tot += 1
    counter += batch_size

    stats(correct_dist,incorrect_dist)

    print('rank 1 accuracy : ',(correct/tot)*100,'%')




def get_embeddings(imgs):
    '''
    imgs : numpy array of list of images of shape [n X 128 x 64 x 3]
    outputs : outputs the embeddings after passing through the 
              base model
    '''
    #initialize the session object


    embeddings = []
    count = 0
    tot_size = imgs.shape[0]

    while (count + BATCH_SIZE ) < tot_size :
        images = imgs[count:count+BATCH_SIZE]
        #run the session the object and not the output
        output = model.predict(images)[0]
        embeddings+=list(output)
        count += BATCH_SIZE

    images = imgs[count:tot_size]


    #run the session the object and not the output
    output = model.predict(images)[0]
    embeddings+=list(output)

    return embeddings


if __name__=='__main__':
    market_test()