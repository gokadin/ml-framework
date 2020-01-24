package models

const (
	defaultBatchSize = 1
	defaultEpochs = 0
	defaultMaxError = 0.001
	defaultValidOutputRange = 0.1
	defaultOptimizerType = OptimizerAdam
	defaultLoss = lossFunctionMeanSquared
)

type ModelConfig struct {
	BatchSize int
	Epochs int
	MaxError float64
	ValidOutputRange float64
	Optimizer string
	OptimizerOverrides []float64
	LearningRate float64
	Loss string
}

func (mc *ModelConfig) populateDefaults() {
	if mc.BatchSize == 0 {
		mc.BatchSize = defaultBatchSize
	}

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
}
