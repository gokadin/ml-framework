package tensor

import (
    "log"
    "math"
)

type operation struct {
    tensor *Tensor
    creators []*Tensor
    metadata []float64
    function func ()
}

func newOperation(tensor *Tensor, creators []*Tensor, metadata ...float64) *operation {
    return &operation{
        tensor: tensor,
        creators: creators,
        metadata: metadata,
    }
}

func newOperationSub(tensor *Tensor, creators []*Tensor) *operation {
    o := newOperation(tensor, creators)
    o.function = o.backpropagateSub
    return o
}

func newOperationAdd(tensor *Tensor, creators []*Tensor) *operation {
    o := newOperation(tensor, creators)
    o.function = o.backpropagateAdd
    return o
}

func newOperationMul(tensor *Tensor, creators []*Tensor) *operation {
    o := newOperation(tensor, creators)
    o.function = o.backpropagateMul
    return o
}

func newOperationMulScalar(tensor *Tensor, creators []*Tensor, scalar float64) *operation {
    o := newOperation(tensor, creators, scalar)
    o.function = o.backpropagateMulScalar
    return o
}

func newOperationDiv(tensor *Tensor, creators []*Tensor) *operation {
    o := newOperation(tensor, creators)
    o.function = o.backpropagateDiv
    return o
}

func newOperationDivScalar(tensor *Tensor, creators []*Tensor, scalar float64) *operation {
    o := newOperation(tensor, creators, scalar)
    o.function = o.backpropagateDivScalar
    return o
}

func newOperationPow(tensor *Tensor, creators []*Tensor, power float64) *operation {
    o := newOperation(tensor, creators, power)
    o.function = o.backpropagatePow
    return o
}

func newOperationPowSelf(tensor *Tensor, power float64) *operation {
    o := newOperation(tensor, nil, power)
    o.function = o.backpropagatePowSelf
    return o
}

func newOperationSum(tensor *Tensor, creators []*Tensor, axis float64) *operation {
    o := newOperation(tensor, creators, axis)
    o.function = o.backpropagateSum
    return o
}

func newOperationExpand(tensor *Tensor, creators []*Tensor, axis float64) *operation {
    o := newOperation(tensor, creators, axis)
    o.function = o.backpropagateExpand
    return o
}

func newOperationDot(tensor *Tensor, creators []*Tensor) *operation {
    o := newOperation(tensor, creators)
    o.function = o.backpropagateDot
    return o
}

func newOperationSigmoid(tensor *Tensor, creators []*Tensor) *operation {
    o := newOperation(tensor, creators)
    o.function = o.backpropagateSigmoid
    return o
}

func (o *operation) backpropagateSub() {
    o.creators[0].grad = o.tensor.grad

    o.creators[1].grad = make([][]float64, len(o.tensor.grad))
    for i := range o.creators[1].grad {
    	o.creators[1].grad[i] = make([]float64, len(o.tensor.grad[0]))
        for j := range o.creators[1].grad[i] {
            o.creators[1].grad[i][j] = o.tensor.grad[i][j] * -1
        }
    }
}

func (o *operation) backpropagatePow() {
    o.creators[0].grad = o.tensor.grad
	if o.metadata[0] - 1 <= 1 {
        for i := range o.creators[0].grad {
           for j := range o.creators[0].grad[i] {
               o.creators[0].grad[i][j] *= o.creators[0].mat[i][j] * o.metadata[0]
           }
        }
    } else {
    	exponent := o.metadata[0] - 1
        for i := range o.creators[0].grad {
            for j := range o.creators[0].grad[i] {
                o.creators[0].grad[i][j] *= math.Pow(o.creators[0].mat[i][j], exponent) * o.metadata[0]
            }
        }
    }
}

func (o *operation) backpropagatePowSelf() {
    if o.metadata[0] == 2 {
        for i := range o.tensor.grad {
            for j := range o.tensor.grad[i] {
                o.tensor.grad[i][j] *= math.Pow(o.tensor.mat[i][j], 0.5) * 2
            }
        }
    } else {
    	log.Fatal("get out of here it's not ready")
        //exponent := o.metadata[0] - 1
        //for i := range o.creators[0].grad {
        //    for j := range o.creators[0].grad[i] {
        //        o.creators[0].grad[i][j] *= math.Pow(o.creators[0].mat[i][j], exponent) * o.metadata[0]
        //    }
        //}
    }
}

func (o *operation) backpropagateSum() {
    o.creators[0].grad = expand(o.tensor.grad, int(o.metadata[0]), len(o.creators[0].mat))
}

func (o *operation) backpropagateExpand() {
    o.creators[0].grad = sum(o.tensor.grad, int(o.metadata[0]))
}

func (o *operation) backpropagateAdd() {
    o.creators[0].grad = o.tensor.grad
    o.creators[1].grad = o.tensor.grad
}

func (o *operation) backpropagateMul() {
    o.creators[0].grad = mul(o.tensor.grad, o.creators[1].mat)
    o.creators[1].grad = mul(o.tensor.grad, o.creators[0].mat)
}

func (o *operation) backpropagateMulScalar() {
    o.creators[0].grad = mulScalar(o.tensor.grad, o.metadata[0])
}

func (o *operation) backpropagateDiv() {
    o.creators[0].grad = mul(o.tensor.grad, divideScalarBy(o.creators[1].mat, 1))
    o.creators[1].grad = mul(o.tensor.grad, divideScalarBy(pow(o.creators[1].mat, 2), 1))
}

func (o *operation) backpropagateDivScalar() {
    o.creators[0].grad = mulScalar(o.tensor.grad, 1 / o.metadata[0])
}

func (o *operation) backpropagateDot() {
    o.creators[1].grad = transpose(dot(transpose(o.tensor.grad), o.creators[0].mat))

    o.creators[0].grad = make([][]float64, len(o.tensor.grad))
    for i := range o.creators[0].grad {
        o.creators[0].grad[i] = make([]float64, len(o.creators[1].mat))
    }
    for i := range o.tensor.grad {
        for j := range o.creators[1].mat {
            o.creators[0].grad[i][j] = o.tensor.grad[i][0] * o.creators[1].mat[j][0]
        }
    }

    //o.creators[1].grad = make([][]float64, len(o.creators[0].mat))
    //for i := range o.creators[1].grad {
    //   o.creators[1].grad[i] = make([]float64, len(o.tensor.grad[0]))
    //}
    //for i := range o.tensor.grad {
    //    for j := range o.creators[0].mat {
    //        o.creators[1].grad[j][0] =
    //    }
    //}
}

func (o *operation) backpropagateSigmoid() {
    o.creators[0].grad = mul(o.tensor.grad, mul(o.tensor.mat, subFromScalar(o.tensor.mat, 1)))
}

func (o *operation) backward() {
    o.function()
}

