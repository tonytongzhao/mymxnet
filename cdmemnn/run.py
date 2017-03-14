import os

if __name__=='__main__':
    eta=[0.2, 0.01, 0.001]
    upass=[5]
    ipass=[5]
    num_embed=[100,500]
    num_hidden=[100,500]
    for e in eta:
        for u in upass:
            for i in ipass:
                for embed in num_embed:
                    for hidden in num_hidden:
                        os.system('python cdmemnn.py -train ~/Data/ml-100k/u.data -eta %f -upass %d -ipass %d -nembed %d -nhidden %d >> ~/Data/ml-100k/cdmemnn_result/cdmemnn_eta_%f_upass_%d_ipass_%d_nembed_%d_nhidden_%d &' %(e,u,i,embed,hidden,e,u,i,embed, hidden))
