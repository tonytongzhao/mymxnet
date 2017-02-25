import mxnet as mx
ctx1=mx.gpu(0)

a=mx.sym.Variable('a')
b=mx.sym.Variable('b')


s=[a*b+a for _ in xrange(5)]
c=mx.sym.ElementWiseSum(*s)
c=mx.sym.sum_axis(data=c, axis=0)


t=[mx.sym.square(i) for i in s]

for _ in xrange(2):
    for i in xrange(len(t)):
        t[i]=mx.sym.square(t[i])

d=mx.sym.ElementWiseSum(*t)


x=mx.nd.array(range(3), ctx=ctx1)
y=mx.nd.array(range(10,13),ctx=ctx1)

exe_c=c.bind(ctx=ctx1,args={'a':x, 'b':y})
exe_d=d.bind(ctx=ctx1, args={'a':x, 'b':y})


exe_c.forward()
exe_d.forward()

print exe_c.outputs[0].asnumpy(), exe_c.outputs[0].shape
print exe_d.outputs[0].asnumpy()
