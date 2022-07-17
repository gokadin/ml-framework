package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type leakyReluTestCase struct {
	name     string
	a        *Tensor
	runOnGpu bool
}

func buildLeakyReluTestCases() []leakyReluTestCase {
	return []leakyReluTestCase{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, -3, 4}), false},
		{"1200x2 ones GPU", From(mat.InitRandom, 1200, 2), true},
		{"1200x2 ones CPU", From(mat.InitRandom, 1200, 2), false},
		{"1200x2 zeros GPU", Zeros(1200, 2), true},
		{"1200x2 zeros CPU", Zeros(1200, 2), false},
		{"2x1200 ones GPU", From(mat.InitRandom, 2, 1200), true},
		{"2x1200 ones CPU", From(mat.InitRandom, 2, 1200), false},
	}
}

func Test_leakyRelu_forward(t *testing.T) {
	for _, test := range buildLeakyReluTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := make([]float32, test.a.Shape().Size())
			afl32 := test.a.ToFloat32()
			for i := 0; i < len(expected); i++ {
				if afl32[i] > 0 {
					expected[i] = afl32[i]
				} else {
					expected[i] = 0.01 * afl32[i]
				}
			}
			c := LeakyRelu(test.a)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
		})
	}
}

func Test_leakyRelu_backward(t *testing.T) {
	for _, test := range buildLeakyReluTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := LeakyRelu(test.a)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Random32f(c.Shape().Size()))
			expectedGrad := make([]float32, test.a.Shape().Size())
			cfl32 := c.ToFloat32()
			cgfl32 := c.GradientToFloat32()
			for i := 0; i < len(expectedGrad); i++ {
				if cfl32[i] > 0 {
					expectedGrad[i] = cgfl32[i]
				} else {
					expectedGrad[i] = 0.01 * cgfl32[i]
				}
			}

			c.backward()

			assert.InDeltaSlice(t, expectedGrad, test.a.GradientToFloat32(), 0.0001)
		})
	}
}
