import os

if __name__=='__main__':
    eta=[0.05,0.02,0.03,0.04,0.06,0.07,0.08]
    upass=[5]
    ipass=[5]
    num_embed=[150]
    num_hidden=[150]
    for e in eta:
        for u in upass:
            for i in ipass:
                for embed in num_embed:
                    for hidden in num_hidden:
                        os.system('python cdmemnn.py -train ~/Data/ml-1m/ratings.dat.random.tr -val ~/Data/ml-1m/ratings.dat.random.te -eta %f -upass %d -ipass %d -nembed %d -nhidden %d -nepoch 100&' %(e,u,i,embed,hidden))
