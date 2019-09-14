package runners

import (
	"github.com/gokadin/ml-framework/core"
	"github.com/gokadin/ml-framework/tensor"
	"testing"
)

func TestNewNetworkRunner(t *testing.T) {
	net := core.NewNetwork().
		AddInputLayer(2).
		AddHiddenLayer(2, tensor.ActivationFunctionSigmoid).
		AddOutputLayer(1, tensor.ActivationFunctionIdentity)

	data := tensor.NewTensor([][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
	target := tensor.NewTensor([][]float64{{0}, {1}, {1}, {0}})

	runner := NewNetworkRunner()
	runner.Train(net, data, target)
}
