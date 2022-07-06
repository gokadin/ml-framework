package tensor

import (
	"ml-framework/mat"
)

const operationConv2d = "opConv2d"

type opConv2d struct {
	a           *Tensor
	filters     *Tensor
	filterCount int
	kernelSize  mat.ShapeN
	stride      mat.ShapeN
}

func (o *opConv2d) name() string {
	return operationConv2d
}

func (o *opConv2d) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opConv2d) forwardShape() Shape {
	// filterCount assumed to be 1 for now
	return Shape{o.a.Shape().X - o.kernelSize.D[0] + 1, o.a.Shape().Y - o.kernelSize.D[1] + 1}
}

func (o *opConv2d) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape, tensorShape}
}

func (o *opConv2d) forward(tensor *Tensor) {
	// for each filter
	aData := o.a.ToFloat32()
	fData := o.filters.ToFloat32()
	data := mat.NewMat32fZeros(mat.WithShape(tensor.Shape().X, tensor.Shape().Y))
	for i := 0; i < tensor.Shape().X; i++ {
		for j := 0; j < tensor.Shape().Y; j++ {
			resultIndex := i*tensor.Shape().Y + j
			var sum float32
			for ki := 0; ki < o.kernelSize.D[0]; ki++ {
				for kj := 0; kj < o.kernelSize.D[1]; kj++ {
					kIndex := ki*o.kernelSize.D[1] + kj
					inputIndex := (i+ki)*o.a.Shape().Y + (j + kj)
					sum += aData[inputIndex] * fData[kIndex]
				}
			}
			data.Set(resultIndex, sum) // plus bias
		}
	}
	tensor.SetData(data.Data())
}

func (o *opConv2d) backward(tensor *Tensor) {

}

func Conv2d(a, filters *Tensor, filterCount int, kernelSize, stride mat.ShapeN) *Tensor {
	o := &opConv2d{a, filters, filterCount, kernelSize, stride}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
