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
	return []Shape{o.a.Shape(), o.target.Shape()}
}

func (o *opCrossEntropy) forward(tensor *Tensor) {
	//opw.a.SetData(mat.Softmax(opw.a.ToMat32f()).Data())
	//C.cross_entropy(opw.a._tensor, opw.target._tensor, tensor._tensor)
}

func (o *opCrossEntropy) backward(tensor *Tensor) {
	//expandedGrad := mat.Expand(mat.Expand(mat.NewMat32f(mat.WithShape(tensor.Shape().X, tensor.Shape().Y), tensor.GradientToFloat32()), 1, opw.a.Shape().Y), 0, opw.a.Shape().X)
	//opw.a.SetGradient(mat.Mul(expandedGrad, mat.Sub(mat.NewMat32f(mat.WithShape(opw.a.Shape().X, opw.a.Shape().Y), opw.a.ToFloat32()), mat.NewMat32f(mat.WithShape(opw.target.Shape().X, opw.target.Shape().Y), opw.target.ToFloat32()))).Data())
}

func SoftmaxCrossEntropy(pred, target *Tensor) *Tensor {
	result := OfShape(1, 1)
	result.op = &opCrossEntropy{pred, target}
	result._tensor.op = C.alloc_sce(pred._tensor, target._tensor)
	return result
}
