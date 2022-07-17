package modules

import (
	"log"
	"ml-framework/tensor"
)

const (
	LossMeanSquared         = "LossMeanSquared"
	LossSoftmaxCrossEntropy = "LossSoftmaxCrossEntropy"
	LossBinaryCrossEntropy  = "LossBinaryCrossEntropy"
	LossPassThrough         = "LossPassThrough"
)

type Criterion interface {
	Forward(pred, target *tensor.Tensor) *tensor.Tensor
}

func NewCriterion(loss string) Criterion {
	switch loss {
	case LossMeanSquared:
		return newMeanSquaredCriterion()
	case LossSoftmaxCrossEntropy:
		return newCrossEntropyCriterion()
	case LossBinaryCrossEntropy:
		return newBinaryCrossEntropyCriterion()
	case LossPassThrough:
		return newCrossEntropyCriterion()
	}

	log.Fatalf("unknown loss function selected: %s", loss)
	return nil
}

type meanSquaredCriterion struct{}

func newMeanSquaredCriterion() *meanSquaredCriterion {
	return &meanSquaredCriterion{}
}

func (msc *meanSquaredCriterion) Forward(pred, target *tensor.Tensor) *tensor.Tensor {
	return tensor.DivScalar(tensor.Sum(tensor.Pow(tensor.Sub(pred, target), 2), 0), float32(pred.Shape().D[0]))
}

type crossEntropyCriterion struct{}

func newCrossEntropyCriterion() *crossEntropyCriterion {
	return &crossEntropyCriterion{}
}

func (cec *crossEntropyCriterion) Forward(pred, target *tensor.Tensor) *tensor.Tensor {
	return tensor.SoftmaxCrossEntropy(pred, target)
}

type binaryCrossEntropyCriterion struct{}

func newBinaryCrossEntropyCriterion() *binaryCrossEntropyCriterion {
	return &binaryCrossEntropyCriterion{}
}

func (bce *binaryCrossEntropyCriterion) Forward(pred, target *tensor.Tensor) *tensor.Tensor {
	return tensor.BinaryCrossEntropy(pred, target)
	return tensor.MulScalar(tensor.Mean(
		tensor.Add(
			tensor.Mul(pred, tensor.Log(target)),
			tensor.Mul(
				tensor.SubFromScalar(pred, 1),
				tensor.Log(tensor.SubFromScalar(target, 1)),
			),
		)), -1)
}

type wasserteinCriticCriterion struct{}

func newWasserteinCriticCriterion() *wasserteinCriticCriterion {
	return &wasserteinCriticCriterion{}
}

func (wc *wasserteinCriticCriterion) Forward(criticOutputReal, criticOutputFake *tensor.Tensor) *tensor.Tensor {
	return tensor.Sub(criticOutputReal, criticOutputFake)
}

type wasserteinGeneratorCriterion struct{}

func newWasserteinGeneratorCriterion() *wasserteinGeneratorCriterion {
	return &wasserteinGeneratorCriterion{}
}

func (wg *wasserteinGeneratorCriterion) Forward(criticOutputFake *tensor.Tensor) *tensor.Tensor {
	return criticOutputFake
}
