package tensor

//#include <softmax-cross-entropy.h>
import "C"

const operationSoftmaxCrossEntropy = "opSoftmaxCrossEntropy"

type opCrossEntropy struct {
	a *Tensor
	target *Tensor
}

func (o *opCrossEntropy) name() string {
	return operationSoftmaxCrossEntropy
}

func (o *opCrossEntropy) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opCrossEntropy) forwardShape() Shape {
	return Shape{1, 1}
}

func (o *opCrossEntropy) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{o.a.Shape()}
}

func (o *opCrossEntropy) forward(tensor *Tensor) {
	if !o.a.Shape().Equals(o.target.Shape()) {
		handleIncompatibleShapes("softmax-cross-entropy", o.a.Shape(), o.target.Shape())
	}
	C.sce_forward(tensor._tensor, o.a._tensor, o.target._tensor)
}

func (o *opCrossEntropy) backward(tensor *Tensor) {
	C.sce_backward(tensor._tensor, o.a._tensor, o.target._tensor)
}

func SoftmaxCrossEntropy(pred, target *Tensor) *Tensor {
	o := &opCrossEntropy{pred, target}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
