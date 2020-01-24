package modules

import (
	"github.com/gokadin/ml-framework/tensor"
	"log"
)

const ActivationIdentity = "identity"
const ActivationSigmoid = "sigmoid"
const ActivationRelu = "relu"
const ActivationSoftmax = "softmax"

func activate(tensor *tensor.Tensor, activation string) *tensor.Tensor {
	switch activation {
	case ActivationIdentity:
		return tensor
	case ActivationSigmoid:
		return tensor.Sigmoid()
	case ActivationRelu:
		return tensor.Relu()
	case ActivationSoftmax:
		return tensor.Softmax()
	}

	log.Fatal("activation function is unknown:", activation)
	return nil
}
