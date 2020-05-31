package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type logTestCases struct {
	name string
	a *Tensor
	runOnGpu bool
}

func buildLogTestCases() []logTestCases {
	return []logTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), false},
	}
}

func Test_log_forward(t *testing.T) {
	for _, test := range buildLogTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Log(test.a.ToMat32f()).Data()
			c := Log(test.a)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_log_backward(t *testing.T) {
	for _, test := range buildLogTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Log(test.a)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))
			expectedGrad := mat.DivScalarBy(c.GradientToMat32(), 1).Data()

			c.backward()

			assert.Equal(t, expectedGrad, test.a.GradientToFloat32())
		})
	}
}
