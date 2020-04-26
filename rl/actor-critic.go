package rl

import (
	"github.com/encryptio/alias"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	gym "github.com/openai/gym-http-api/binding-go"
	"math/rand"
	"time"
)

const (
	epochs = 1000
	gamma = 0.95
)

type actorCriticModel struct {
	baseModel *models.Model
	actorModel *models.Model
	criticModel *models.Model
	state *tensor.Tensor
	policy *tensor.Tensor
	value *tensor.Tensor
	graph *tensor.Graph
}

func (acm *actorCriticModel) predict() {
	acm.graph = tensor.NewGraph()
	state := tensor.Variable(1, 4)
	base := acm.baseModel.Predict(state)
	acm.policy = acm.actorModel.Predict(base)
	acm.value = acm.criticModel.Predict(base)
}

func (acm *actorCriticModel) forward(stateMat []float32) {
	acm.state.SetData(stateMat)
	acm.graph.Forward(acm.policy)
	acm.graph.Forward(acm.value)
}

func RunActorCritic() {
	model := &actorCriticModel{
		baseModel: buildBaseModel(),
		actorModel: buildActorModel(),
		criticModel: buildCriticModel(),
	}
	model.predict()

	worker(model)
}

func worker(model *actorCriticModel) {
	env, envId := gymClient()

	for i := 0; i < epochs; i++ {
		values, logProbs, rewards := runEpisode(env, envId, model)
		updateParams(values, logProbs, rewards)
	}

	_ = env.Close(envId)
}

func runEpisode(env *gym.Client, envId gym.InstanceID, model *actorCriticModel) ([][]float32, []float32, []float32) {
	envState, _ := env.Reset(envId)
	stateMat := obsToState(envState.([]float64))
	values := make([][]float32, 0)
	logProbs := make([]float32, 0)
	rewards := make([]float32, 0)
	done := false
	j := 0

	for !done {
		j += 1
		model.forward(stateMat)
		values = append(values, model.value.ToFloat32())
		action := sampleFrom(model.policy.ToFloat64())
		logProbs = append(logProbs, model.policy.ToFloat32()[action])
		envState, _, envDone, _, _ := env.Step(envId, action, false)
		stateMat = obsToState(envState.([]float64))
		done = envDone

		var reward float32
		if done {
			reward = -10
			_, _ = env.Reset(envId)
		} else {
			reward = 1
		}
		rewards = append(rewards, reward)
	}

	return values, logProbs, rewards
}

func updateParams(values [][]float32, logProbs, rewards []float32) {
	returns := make([]float32, len(rewards))
	var partialReturn float32
	for i := len(rewards) - 1; i >= 0; i-- {
		partialReturn = rewards[i] + partialReturn * gamma
		returns[len(rewards) - i - 1] = partialReturn
	}
}

func sampleFrom(probabilities []float64) int {
	distribution, _ := alias.New(probabilities)
	index := distribution.Gen(rand.New(rand.NewSource(time.Now().UTC().UnixNano())))
	return int(index)
}

func buildBaseModel() *models.Model {
	return models.Build(
		modules.Dense(25, modules.ActivationRelu),
		modules.Dense(50, modules.ActivationRelu))
}

func buildActorModel() *models.Model {
	return models.Build(
		modules.Dense(2, modules.ActivationLogSoftmax))
}

func buildCriticModel() *models.Model {
	return models.Build(
		modules.Dense(25, modules.ActivationRelu),
		modules.Dense(1, modules.ActivationTanh))
}
