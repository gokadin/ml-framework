package runners

import (
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
)

const (
	defaultEpochs = 0
	defaultMaxError = 0.001
	defaultValidOutputRange = 0.1
	defaultLearningRate = 0.001
	defaultOptimizerType = models.OptimizerAdam
	defaultLoss = modules.LossMeanSquared
	defaultLogFolder = "logs"
)

type ModelConfig struct {
	Epochs int
	MaxError float32
	ValidOutputRange float32
	Optimizer string
	OptimizerOverrides []float32
	LearningRate float32
	Loss string
	LogFolder string
	LogToFile bool
}

func (mc *ModelConfig) populateDefaults() {
	if mc.Epochs == 0 {
		mc.Epochs = defaultEpochs
	}

	if mc.MaxError == 0 {
		mc.MaxError = defaultMaxError
	}

	if mc.ValidOutputRange == 0 {
		mc.ValidOutputRange = defaultValidOutputRange
	}

	if mc.Optimizer == "" {
		mc.Optimizer = defaultOptimizerType
	}

	if mc.LearningRate == 0 {
		mc.LearningRate = defaultLearningRate
	}

	if mc.Loss == "" {
		mc.Loss = defaultLoss
	}

	if mc.LogFolder == "" {
		mc.LogFolder = defaultLogFolder
	}
}
