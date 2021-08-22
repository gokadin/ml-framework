package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type divScalarTestCases struct {
	name     string
	a        *Tensor
	scalar   float32
	runOnGpu bool
}

func buildDivScalarTestCases() []divScalarTestCases {
	return []divScalarTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), 2, true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), 2, false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), 2, true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), 3, false},
	}
}

func Test_divScalar_forward(t *testing.T) {
	for _, test := range buildDivScalarTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.DivScalar(test.a.ToMat32f(), test.scalar).Data()
			c := DivScalar(test.a, test.scalar)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_divScalar_backward(t *testing.T) {
	for _, test := range buildDivScalarTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := DivScalar(test.a, test.scalar)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))
			multiplier := 1.0 / test.scalar
			expectedGrad := mat.MulScalar(c.GradientToMat32(), multiplier).Data()

			c.backward()

			assert.Equal(t, expectedGrad, test.a.GradientToFloat32())
		})
	}
}
