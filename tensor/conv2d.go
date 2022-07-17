package tensor

import (
	"ml-framework/mat"
)

const operationConv2d = "opConv2d"

type opConv2d struct {
	a           *Tensor
	filters     *Tensor
	filterCount int
	kernelSize  mat.Shape
	stride      mat.Shape
}

func (o *opConv2d) name() string {
	return operationConv2d
}

func (o *opConv2d) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opConv2d) forwardShape() mat.Shape {
	// filterCount assumed to be 1 for now
	return mat.Dim(o.filterCount, o.a.Shape().D[0]-o.kernelSize.D[0]+1, o.a.Shape().D[1]-o.kernelSize.D[1]+1)
}

func (o *opConv2d) backwardShapes(tensorShape mat.Shape) []mat.Shape {
	return []mat.Shape{tensorShape, tensorShape}
}

func (o *opConv2d) forward(tensor *Tensor) {
	aData := o.a.ToFloat32()
	fData := o.filters.ToFloat32()
	result := mat.NewMat32fZeros(tensor.Shape())
	for f := 0; f < o.filters.Shape().D[0]; f++ {
		o.forwardSingleFilter(tensor, aData, fData, result, f)
	}
	tensor.SetData(result.Data())
}

func (o *opConv2d) forwardSingleFilter(tensor *Tensor, aData, fData []float32, result *mat.M32f, filterIndex int) {
	for i := 0; i < tensor.Shape().D[1]; i++ {
		for j := 0; j < tensor.Shape().D[2]; j++ {
			resultIndex := i*tensor.Shape().D[2] + j
			var sum float32
			for ki := 0; ki < o.kernelSize.D[0]; ki++ {
				for kj := 0; kj < o.kernelSize.D[1]; kj++ {
					kIndex := ki*o.kernelSize.D[1] + kj
					inputIndex := (i+ki)*o.a.Shape().D[1] + (j + kj)
					sum += aData[inputIndex] * fData[kIndex+filterIndex*o.kernelSize.Size()]
				}
			}
			result.Set(resultIndex+filterIndex*tensor.Shape().D[1]*tensor.Shape().D[2], sum)
		}
	}
}

func (o *opConv2d) backward(tensor *Tensor) {

}

func Conv2d(a, filters *Tensor, filterCount int, kernelSize, stride mat.Shape) *Tensor {
	o := &opConv2d{a, filters, filterCount, kernelSize, stride}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
