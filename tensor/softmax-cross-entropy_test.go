package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type softmaxCrossEntropyTestCases struct {
	name string
	a *Tensor
	b *Tensor
	runOnGpu bool
}

func buildSoftmaxCrossEntropyTestCases() []softmaxCrossEntropyTestCases {
	return []softmaxCrossEntropyTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), true},
		//{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), true},
		//{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), false},
	}
}

func Test_softmaxCrossEntropy_forward(t *testing.T) {
	t.Parallel()
	for _, test := range buildSoftmaxCrossEntropyTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			expected := mat.Neg(mat.Sum(mat.Sum(mat.Mul(test.a.ToMat32f(), mat.Log(mat.Softmax(test.b.ToMat32f()))), 1), 0)).Data()
			c := SoftmaxCrossEntropy(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}
