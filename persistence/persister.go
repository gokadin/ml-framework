package persistence

import (
	"bufio"
	"fmt"
	"github.com/gokadin/ml-framework/mat"
	"log"
	"os"
	"strconv"
	"strings"
)

const (
	modelStoreRoot                    = ".cache/models"
	maxFloatsPerLine				  = 300
	persisterTypeKey                  = "TYPE"
	persisterModelType                = "MODEL"
	persisterModelConfigCriterionKey  = "CRITERION"
	persisterModelConfigOptimizerKey  = "OPTIMIZER"
	persisterModelConfigEpochsKey     = "EPOCHS"
	persisterModelModuleKey           = "MODULE"
	persisterModelModuleEndKey        = "MODULE_END"
	persisterModelModuleSHAPEKEY      = "SHAPE"
	persisterModelModuleActivationKey = "ACTIVATION"
	persisterModelModuleWeightsBeginKey    = "WEIGHTS_BEGIN"
	persisterModelModuleWeightsKey    = "WEIGHTS"
	persisterModelModuleBiasesBeginKey     = "BIASES_BEGIN"
	persisterModelModuleBiasesKey     = "BIASES"
)

func BuildDefinition(name string) *modelDefinition {
	_, err := os.Stat(fmt.Sprintf("%s/%s", modelStoreRoot, name))
	if os.IsNotExist(err) {
		log.Fatalf("model %s does not exist", name)
	}

	file, err := os.Open(fmt.Sprintf("%s/%s", modelStoreRoot, name))
	if err != nil {
		log.Fatal("Could not retrieve file from model store")
	}
	defer file.Close()

	model := &modelDefinition{Name: name, Modules: make([]*moduleDefinition, 0), Configuration: &configurationDefinition{}}
	scanner := bufio.NewScanner(file)
	var moduleShape mat.Shape
	var moduleActivationFunction string
	var moduleWeights []float32
	var moduleWeightsIndex int
	var moduleBiases []float32
	var moduleBiasesIndex int
	for scanner.Scan() {
		line := scanner.Text()
		split := strings.Split(line, ":")
		if line == "" || len(split) == 0 {
			continue
		}
		switch split[0] {
		case persisterModelConfigCriterionKey:
			model.Configuration.Loss = strings.TrimSpace(split[1])
			break
		case persisterModelConfigOptimizerKey:
			model.Configuration.Optimizer = strings.TrimSpace(split[1])
			break
		case persisterModelConfigEpochsKey:
			model.Configuration.Epochs, _ = strconv.Atoi(strings.TrimSpace(split[1]))
			break
		case persisterModelModuleKey:
			break
		case persisterModelModuleSHAPEKEY:
			parameters := strings.Split(strings.TrimSpace(split[1]), " ")
			moduleShape.X, _ = strconv.Atoi(parameters[0])
			moduleShape.Y, _ = strconv.Atoi(parameters[1])
			break
		case persisterModelModuleActivationKey:
			moduleActivationFunction = strings.TrimSpace(split[1])
			break
		case persisterModelModuleWeightsBeginKey:
			moduleWeights = make([]float32, moduleShape.X * moduleShape.Y)
			moduleWeightsIndex = 0
			break
		case persisterModelModuleWeightsKey:
			parameters := strings.Split(strings.TrimSpace(split[1]), " ")
			for _, value := range parameters {
				parsed, _ := strconv.ParseFloat(value, 32)
				moduleWeights[moduleWeightsIndex] = float32(parsed)
				moduleWeightsIndex++
			}
			break
		case persisterModelModuleBiasesBeginKey:
			moduleBiases = make([]float32, moduleShape.Y)
			moduleBiasesIndex = 0
			break
		case persisterModelModuleBiasesKey:
			parameters := strings.Split(strings.TrimSpace(split[1]), " ")
			for _, value := range parameters {
				parsed, _ := strconv.ParseFloat(value, 32)
				moduleBiases[moduleBiasesIndex] = float32(parsed)
				moduleBiasesIndex++
			}
			break
		case persisterModelModuleEndKey:
			module := &moduleDefinition{
				Type: "dense",
				Shape: &ShapeDefinition{X: moduleShape.X, Y: moduleShape.Y},
				Activation: moduleActivationFunction,
				Weights: moduleWeights,
				Biases: moduleBiases,
			}
			model.Modules = append(model.Modules, module)
			break
		}
	}

	return model
}

type modelDefinition struct {
	Name string `json:"name"`
	Modules []*moduleDefinition `json:"modules"`
	Configuration *configurationDefinition `json:"configuration"`
}

type moduleDefinition struct {
	Type string `json:"type"`
	Shape *ShapeDefinition `json:"shape"`
	Activation string `json:"activation"`
	Weights []float32 `json:"weights"`
	Biases []float32 `json:"biases"`
}

type ShapeDefinition struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type configurationDefinition struct {
	Loss string `json:"loss"`
	Optimizer string `json:"optimizer"`
	Epochs int `json:"epochs"`
}