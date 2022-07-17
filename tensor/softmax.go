package tensor

//#include <softmax.h>
import "C"
import "ml-framework/mat"

const operationSoftmax = "opSoftmax"

type opSoftmax struct {
	a *Tensor
}

func (o *opSoftmax) name() string {
	return operationSoftmax
}

func (o *opSoftmax) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opSoftmax) forwardShape() mat.Shape {
	return o.a.Shape()
}

func (o *opSoftmax) backwardShapes(tensorShape mat.Shape) []mat.Shape {
	return []mat.Shape{tensorShape}
}

func (o *opSoftmax) forward(tensor *Tensor) {
	C.softmax_forward(tensor._tensor, o.a._tensor)
}

func (o *opSoftmax) backward(tensor *Tensor) {
	o.a.SetGradient(tensor.GradientToFloat32())
}

func Softmax(a *Tensor) *Tensor {
	o := &opSoftmax{a}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
