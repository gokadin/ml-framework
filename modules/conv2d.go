package modules

import (
	"fmt"
	"ml-framework/mat"
	"ml-framework/tensor"
)

type Conv2dModule struct {
	Type          string    `json:"type"`
	FilterCount   int       `json:"filterCount"`
	KernelSize    mat.Shape `json:"kernelSize"`
	Stride        mat.Shape `json:"stride"`
	IsInitialized bool      `json:"is_initialized"`
	filters       *tensor.Tensor
}

func Conv2d(filterCount int, kernelSize, stride mat.Shape) *Conv2dModule {
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
		d.filters = tensor.From(mat.InitNormalized, d.conv2DFilterShape().D...).
			SetName(fmt.Sprintf("Conv2dModule layer (%d) filters", d.FilterCount))
		d.IsInitialized = true
	}

	return tensor.Conv2d(input, d.filters, d.FilterCount, d.KernelSize, d.Stride)
}

func (d *Conv2dModule) OverrideFilters(tensor *tensor.Tensor) {
	if !tensor.Shape().Equals(d.conv2DFilterShape()) {
		panic(fmt.Sprintf("cannot override filters with invalid shape: %s, requires: %s\n", tensor.Shape().Print(),
			d.conv2DFilterShape().Print()))
	}

	d.filters = tensor
	d.IsInitialized = true
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

func (d *Conv2dModule) conv2DFilterShape() mat.Shape {
	return d.KernelSize.Copy().Expand(d.FilterCount)
}
