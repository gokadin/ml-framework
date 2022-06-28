package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type softmaxCrossEntropyTestCases struct {
	name     string
	a        *Tensor
	b        *Tensor
	runOnGpu bool
}

func buildSoftmaxCrossEntropyTestCases() []softmaxCrossEntropyTestCases {
	return []softmaxCrossEntropyTestCases{
		{"5x3 GPU", From(InitRandom, 5, 3), From(InitRandom, 5, 3), true},
		{"5x3 CPU", From(InitRandom, 5, 3), From(InitRandom, 5, 3), false},
		{"2x5 GPU", From(InitRandom, 2, 5), From(InitRandom, 2, 5), true},
		{"2x5 CPU", From(InitRandom, 2, 5), From(InitRandom, 2, 5), false},
		{"1000x10 GPU", From(InitRandom, 1000, 10), From(InitRandom, 1000, 10), true},
		{"1000x10 CPU", From(InitRandom, 1000, 10), From(InitRandom, 1000, 10), false},
	}
}

func Test_softmaxCrossEntropy_forward(t *testing.T) {
	for _, test := range buildSoftmaxCrossEntropyTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			aSoftmaxMat := mat.Softmax(test.a.ToMat32f())
			expected := mat.DivScalar(mat.Sum(mat.Neg(mat.Log(mat.Sum(mat.Mul(test.b.ToMat32f(), aSoftmaxMat), 1))), 0), float32(test.a.Shape().X)).Data()
			c := SoftmaxCrossEntropy(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
			assert.InDeltaSlice(t, aSoftmaxMat.Data(), test.a.ToFloat32(), 0.0001)
		})
	}
}

func Test_softmaxCrossEntropy_backward(t *testing.T) {
	for _, test := range buildSoftmaxCrossEntropyTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := SoftmaxCrossEntropy(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Random32f(c.Shape().Size()))
			expandedGrad := mat.Expand(mat.Expand(c.GradientToMat32(), 1, test.a.Shape().Y), 0, test.a.Shape().X)
			expectedGrad := mat.Mul(expandedGrad, mat.Sub(test.a.ToMat32f(), test.b.ToMat32f())).Data()

			c.backward()

			assert.InDeltaSlice(t, expectedGrad, test.a.GradientToFloat32(), 0.0001)
		})
	}
}
