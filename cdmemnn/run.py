import os

if __name__=='__main__':
    eta=[0.1,0.05,0.08]
    upass=[2,5]
    ipass=[2,5]
    num_embed=[500]
    num_hidden=[500]
    for e in eta:
        for u in upass:
            for i in ipass:
                for embed in num_embed:
                    for hidden in num_hidden:
                        os.system('python cdmemnn.py -train ~/Data/ml-1m/ratings.dat.random.tr -val ~/Data/ml-1m/ratings.dat.random.te -log 1 -eta %f -upass %d -ipass %d -nembed %d -nhidden %d -dropout 0.5 -nepoch 100 -npass 3&' %(e,u,i,embed,hidden))
