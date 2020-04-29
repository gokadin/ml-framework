package rl

import (
	"github.com/encryptio/alias"
	"github.com/gokadin/ml-framework/mat"
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
	clc = 0.1
)

type actorCriticModel struct {
	baseModel *models.Model
	actorModel *models.Model
	criticModel *models.Model
	state *tensor.Tensor
	policy *tensor.Tensor
	value *tensor.Tensor
	logProbs *tensor.Tensor
	values *tensor.Tensor
	returns *tensor.Tensor
	actorLoss *tensor.Tensor
	criticLoss *tensor.Tensor
	graph *tensor.Graph
}

func (acm *actorCriticModel) predict() {
	acm.graph = tensor.NewGraph()
	acm.state = tensor.Variable(1, 4)
	base := acm.baseModel.Predict(acm.state)
	acm.policy = acm.actorModel.Predict(base)
	acm.value = acm.criticModel.Predict(base)

	acm.logProbs = tensor.Variable(1, 1)
	acm.values = tensor.Variable(1, 1)
	acm.returns = tensor.Variable(1, 1)
	acm.actorLoss = calculateActorLoss(acm.logProbs, acm.values, acm.returns)
	acm.criticLoss = calculateCriticLoss(acm.values, acm.returns)
}

func calculateActorLoss(logProbs, values, returns *tensor.Tensor) *tensor.Tensor {
	return tensor.Sum(tensor.Sum(tensor.Mul(tensor.Neg(logProbs), tensor.Sub(returns, values)), 1), 0)
}

func calculateCriticLoss(values, returns *tensor.Tensor) *tensor.Tensor {
	return tensor.MulScalar(tensor.Sum(tensor.Sum(tensor.Pow(tensor.Sub(values, returns), 2), 1), 0), clc)
}

func (acm *actorCriticModel) forward(stateMat []float32) {
	acm.state.SetData(stateMat)
	acm.graph.Forward(acm.policy)
	acm.graph.Forward(acm.value)
}

func (acm *actorCriticModel) backward(values []float32, logProbs, returns []float32) {
	acm.values.Reshape(1, len(values)).SetData(values)
	acm.logProbs.Reshape(1, len(values)).SetData(logProbs)
	acm.returns.Reshape(1, len(values)).SetData(returns)

	acm.graph.Backward(acm.actorLoss, append(acm.baseModel.TrainableVariables(), acm.actorModel.TrainableVariables()...)...)
	acm.graph.Backward(acm.criticLoss, acm.criticModel.TrainableVariables()...)
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
		updateParams(model, values, logProbs, rewards)
	}

	_ = env.Close(envId)
}

func runEpisode(env *gym.Client, envId gym.InstanceID, model *actorCriticModel) ([]float32, []float32, []float32) {
	envState, _ := env.Reset(envId)
	stateMat := obsToState(envState.([]float64))
	values := make([]float32, 0)
	logProbs := make([]float32, 0)
	rewards := make([]float32, 0)

	for {
		model.forward(stateMat)
		values = append(values, model.value.ToFloat32()...)
		action := sampleFrom(model.policy.ToFloat64())
		logProbs = append(logProbs, model.policy.ToFloat32()[action])
		envState, _, done, _, _ := env.Step(envId, action, false)
		stateMat = obsToState(envState.([]float64))

		if done {
			_, _ = env.Reset(envId)
			rewards = append(rewards, -10)
			break
		}

		rewards = append(rewards, 1)
	}

	return values, logProbs, rewards
}

func updateParams(model *actorCriticModel, values []float32, logProbs, rewards []float32) {
	returns := make([]float32, len(rewards))
	var partialReturn float32
	for i := len(rewards) - 1; i >= 0; i-- {
		partialReturn = rewards[i] + partialReturn * gamma
		returns[len(rewards) - i - 1] = partialReturn
	}
	mat.Normalize32f(returns)

	model.backward(values, logProbs, returns)
}

func sampleFrom(probabilities []float64) int {
	a := make([]float32, len(probabilities))
	for i := 0; i < len(probabilities); i++ {
		a[i] = float32(probabilities[i])
	}
	x := mat.Softmax(mat.NewMat32f(mat.WithShape(1, len(a)), a))
	b := make([]float64, len(a))
	for i := 0; i < len(b); i++ {
		b[i] = float64(x.Data()[i])
	}
	distribution, err := alias.New(b)
	must(err)
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
