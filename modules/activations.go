package modules

import (
	"github.com/gokadin/ml-framework/tensor"
	"log"
)

const ActivationIdentity = "identity"
const ActivationSigmoid = "sigmoid"
const ActivationRelu = "relu"
const ActivationSoftmax = "softmax"

func activate(t *tensor.Tensor, activation string) *tensor.Tensor {
	switch activation {
	case ActivationIdentity:
		return t
	case ActivationSigmoid:
		return tensor.Sigmoid(t)
	case ActivationRelu:
		return tensor.Relu(t)
	case ActivationSoftmax:
		//return tensor.Softmax()
		return nil
	}

	log.Fatal("activation function is unknown:", activation)
	return nil
}
