package examples

import (
	"fmt"
	"ml-framework/mat"
	"ml-framework/modules"
	"ml-framework/tensor"
	"testing"
)

func Test_temp(t *testing.T) {
	some()
}

func some() {
	a := tensor.FromMat32(mat.FromSlice32f(mat.Dim(4, 6), []float32{
		2, 3, 1, 2, 3, 0,
		1, 1, 2, 2, 0, 5,
		2, 0, 5, 0, 1, 0,
		0, 0, 1, 0, 2, 4,
	}))

	// filters 1 kernel 3x3 stride 1x1
	// [17 16 16 13
	//  12 11 13 14]

	// filters 2 kernel 3x3 stride 1x1
	// [17 16 16 13
	//  12 11 13 14
	//  17 16 16 13
	//  12 11 13 14]

	m := modules.Conv2d(2, mat.Dim(3), mat.Dim(1))
	filterData := []float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,

		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}
	m.OverrideFilters(tensor.FromMat32(mat.FromSlice32f(mat.Dim(2, 3, 3), filterData)))
	y := m.Build(a)

	graph := tensor.NewGraph()

	graph.Forward(y)

	fmt.Println(y.ToFloat32())
}
