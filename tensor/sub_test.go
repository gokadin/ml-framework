package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type subTestCases struct {
	name     string
	a        *Tensor
	b        *Tensor
	runOnGpu bool
}

func buildSubTestCases() []subTestCases {
	return []subTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), false},
	}
}

func Test_sub_forward(t *testing.T) {
	for _, test := range buildSubTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Sub(test.a.ToMat32f(), test.b.ToMat32f()).Data()
			c := Sub(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_sub_backward(t *testing.T) {
	for _, test := range buildSubTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Sub(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))

			c.backward()

			assert.Equal(t, c.GradientToFloat32(), test.a.GradientToFloat32())
			assert.Equal(t, mat.Neg(c.GradientToMat32()).Data(), test.b.GradientToFloat32())
		})
	}
}

func Test_sub_forward_invalid_shapes(t *testing.T) {
	a := OfShape(1, 2).SetData([]float32{1, 2})
	b := OfShape(1, 3).SetData([]float32{1, 2, 3})
	c := Sub(a, b)

	assert.Panics(t, c.forward)
}

func Test_sub_forward_reshape(t *testing.T) {
	a := OfShape(1, 2)
	b := OfShape(1, 2)
	c := Sub(a, b)
	a.Reshape(2, 2)
	b.Reshape(2, 2)
	a.SetData([]float32{1, 2, 3, 4})
	b.SetData([]float32{1, 2, 3, 4})
	expected := mat.Sub(a.ToMat32f(), b.ToMat32f()).Data()

	c.forward()

	assert.Equal(t, a.Shape().ToArray(), c.Shape().ToArray())
	assert.Equal(t, expected, c.ToFloat32())
}
