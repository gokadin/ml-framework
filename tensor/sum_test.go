package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type sumTestCases struct {
	name     string
	a        *Tensor
	runOnGpu bool
}

func buildSumTestCases() []sumTestCases {
	return []sumTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), false},
		{"3x4 GPU", Ones(3, 4), true},
		{"3x4 CPU", Ones(3, 4), false},
		{"1200x2 GPU", Ones(1200, 2), true},
		{"1200x2 CPU", Ones(1200, 2), false},
		{"2x1200 GPU", Ones(2, 1200), true},
		{"2x1200 CPU", Ones(2, 1200), false},
	}
}

func Test_sum0_forward(t *testing.T) {
	for _, test := range buildSumTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Sum(test.a.ToMat32f(), 0).Data()
			c := Sum(test.a, 0)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_sum1_forward(t *testing.T) {
	for _, test := range buildSumTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Sum(test.a.ToMat32f(), 1).Data()
			c := Sum(test.a, 1)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_sum0_backward(t *testing.T) {
	for _, test := range buildSumTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Sum(test.a, 0)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))
			expectedGrad := mat.Expand(c.GradientToMat32(), 0, test.a.Shape().D[0]).Data()

			c.backward()

			assert.Equal(t, expectedGrad, test.a.GradientToFloat32())
		})
	}
}

func Test_sum1_backward(t *testing.T) {
	for _, test := range buildSumTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Sum(test.a, 1)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))
			expectedGrad := mat.Expand(c.GradientToMat32(), 1, test.a.Shape().D[1]).Data()

			c.backward()

			assert.Equal(t, expectedGrad, test.a.GradientToFloat32())
		})
	}
}
