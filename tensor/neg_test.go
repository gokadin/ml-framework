package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type negTestCases struct {
	name     string
	a        *Tensor
	runOnGpu bool
}

func buildNegTestCases() []negTestCases {
	return []negTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), false},
	}
}

func Test_neg_forward(t *testing.T) {
	for _, test := range buildNegTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Neg(test.a.ToMat32f()).Data()
			c := Neg(test.a)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_neg_backward(t *testing.T) {
	for _, test := range buildNegTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Neg(test.a)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))

			c.backward()

			assert.Equal(t, c.GradientToFloat32(), test.a.GradientToFloat32())
		})
	}
}

func Test_neg_forward_reshape(t *testing.T) {
	a := OfShape(1, 2)
	b := OfShape(1, 2)
	c := Neg(a)
	a.Reshape(2, 2)
	b.Reshape(2, 2)
	a.SetData([]float32{1, 2, 3, 4})
	b.SetData([]float32{1, 2, 3, 4})
	expected := mat.Neg(a.ToMat32f()).Data()

	c.forward()

	assert.True(t, a.Shape().Equals(c.Shape()))
	assert.Equal(t, expected, c.ToFloat32())
}
