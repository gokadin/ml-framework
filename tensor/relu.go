package tensor

//#include <relu.h>
import "C"

const operationRelu = "opRelu"

type opRelu struct {
	a *Tensor
}

func (o *opRelu) name() string {
	return operationRelu
}

func (o *opRelu) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opRelu) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opRelu) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape}
}

func (o *opRelu) forward(tensor *Tensor) {
	C.relu_forward(tensor._tensor, o.a._tensor)
}

func (o *opRelu) backward(tensor *Tensor) {
	C.relu_backward(tensor._tensor, o.a._tensor)
}

func Relu(a *Tensor) *Tensor {
	o := &opRelu{a}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
