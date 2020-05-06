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
	return []Shape{tensorShape, tensorShape}
}

func (o *opRelu) forward(tensor *Tensor) {
	//C.relu(opw.a._tensor, tensor._tensor)
	//tensor.SetData(mat.Apply(tensor.ToMat32f(), func(value float32) float32 {
	//	if value > 0 {
	//		return value
	//	}
	//	return 0
	//}).Data())
}

func (o *opRelu) backward(tensor *Tensor) {
	//d := mat.Apply(tensor.ToMat32f(), func(value float32) float32 {
	//	if value > 0 {
	//		return 1
	//	}
	//	return 0
	//})
	//opw.a.SetGradient(mat.Mul(tensor.GradientToMat32(), d).Data())
}

func Relu(a *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opRelu{a}
	result._tensor.op = C.alloc_relu(a._tensor)
	return result
}
