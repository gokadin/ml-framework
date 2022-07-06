package modules

import (
	"ml-framework/mat"
	"ml-framework/tensor"
)

type Conv2dModule struct {
	Type          string     `json:"type"`
	FilterCount   int        `json:"filterCount"`
	KernelSize    mat.ShapeN `json:"kernelSize"`
	Stride        mat.ShapeN `json:"stride"`
	IsInitialized bool       `json:"is_initialized"`
	filters       *tensor.Tensor
}

func Conv2d(filterCount int, kernelSize, stride mat.ShapeN) *Conv2dModule {
	if kernelSize.Count() == 1 {
		kernelSize = mat.Dim(kernelSize.D[0], kernelSize.D[0])
	}
	if stride.Count() == 1 {
		stride = mat.Dim(stride.D[0], stride.D[0])
	}
	return &Conv2dModule{
		Type:        "Conv2dModule",
		FilterCount: filterCount,
		KernelSize:  kernelSize,
		Stride:      stride,
	}
}

func (d *Conv2dModule) Build(input *tensor.Tensor) *tensor.Tensor {
	if !d.IsInitialized {
		for i := 0; i < d.FilterCount; i++ {
			d.filters = tensor.FromMat32(mat.FromSlice32f(mat.WithShape(d.KernelSize.D[0], d.KernelSize.D[1]), []float32{
				1, 1, 1,
				1, 1, 1,
				1, 1, 1,
			}))
		}
		//d.filters = tensor.FromMat32(mat.Initialize(mat.InitNormalized, d.KernelSize))

		//d.Weights = tensor.From(tensor.InitXavier, input.Shape().Y, d.UnitCount).SetName(fmt.Sprintf("Conv2dModule layer (%d) weights", d.UnitCount))
		//d.Bias = tensor.Zeros(1, d.UnitCount).SetName(fmt.Sprintf("Conv2dModule layer (%d) biases", d.UnitCount))
		d.IsInitialized = true
	}

	return tensor.Conv2d(input, d.filters, d.FilterCount, d.KernelSize, d.Stride)
}

func (d *Conv2dModule) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

func (d *Conv2dModule) Copy() Module {
	module := Conv2d(d.FilterCount, d.KernelSize, d.Stride)
	module.IsInitialized = d.IsInitialized
	return module
}

func (d *Conv2dModule) GetType() string {
	return d.Type
}
