package tensor

//#include <relu.h>
import "C"
import "ml-framework/mat"

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

func (o *opRelu) forwardShape() mat.Shape {
	return o.a.Shape()
}

func (o *opRelu) backwardShapes(tensorShape mat.Shape) []mat.Shape {
	return []mat.Shape{tensorShape}
}

func (o *opRelu) forward(tensor *Tensor) {
	C.relu_forward(tensor._tensor, o.a._tensor)
}

func (o *opRelu) backward(tensor *Tensor) {
	C.relu_backward(tensor._tensor, o.a._tensor)
}

func Relu(a *Tensor) *Tensor {
	o := &opRelu{a}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
