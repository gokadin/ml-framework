package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type reluTestCases struct {
	name string
	a *Tensor
	runOnGpu bool
}

func buildReluTestCases() []reluTestCases {
	return []reluTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), false},
		{"1200x2 ones GPU", Ones(1200, 2), true},
		{"1200x2 ones CPU", Ones(1200, 2), false},
		{"1200x2 zeros GPU", Zeros(1200, 2), true},
		{"1200x2 zeros CPU", Zeros(1200, 2), false},
	}
}

func Test_relu_forward(t *testing.T) {
	t.Parallel()
	for _, test := range buildReluTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			expected := make([]float32, test.a.Shape().Size())
			afl32 := test.a.ToFloat32()
			for i := 0; i < len(expected); i++ {
				if afl32[i] > 0 {
					expected[i] = afl32[i]
				} else {
					expected[i] = 0
				}
			}
			c := Relu(test.a)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_relu_backward(t *testing.T) {
	t.Parallel()
	for _, test := range buildReluTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			expectedGrad := make([]float32, test.a.Shape().Size())
			agfl32 := test.a.GradientToFloat32()
			for i := 0; i < len(expectedGrad); i++ {
				if agfl32[i] > 0 {
					expectedGrad[i] = 1
				} else {
					expectedGrad[i] = 0
				}
			}
			c := Relu(test.a)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))

			c.backward()

			assert.Equal(t, c.GradientToFloat32(), test.a.GradientToFloat32())
		})
	}
}

