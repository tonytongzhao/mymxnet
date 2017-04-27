import os

if __name__=='__main__':
    eta=[0.01,0.005,0.008]
    upass=[2]
    ipass=[2]
    num_embed=[200]
    num_hidden=[200]
    batch_size=[1000,1500,500]
    npass=[1]
    for e in eta:
        for u in upass:
            for i in ipass:
                for embed in num_embed:
                    for hidden in num_hidden:
                        for n in npass:
                            for b in batch_size:
                                os.system('python cdmemnn.py -model colattnn -train ~/Data/ml-1m/ratings.dat.random.tr -val ~/Data/ml-1m/ratings.dat.random.te -log 1 -eta %f -upass %d -ipass %d -nembed %d -nhidden %d -batch_size %d  -nepoch 100 -npass %d&' %(e,u,i,embed,hidden,b,n))
