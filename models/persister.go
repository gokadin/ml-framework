package models

import (
	"bufio"
	"fmt"
	"log"
	"ml-framework/mat"
	"ml-framework/modules"
	"os"
	"strconv"
	"strings"
)

const (
	modelStoreRoot                      = ".cache/models"
	maxFloatsPerLine                    = 300
	persisterTypeKey                    = "TYPE"
	persisterModelType                  = "MODEL"
	persisterModelModuleKey             = "MODULE"
	persisterModelModuleEndKey          = "MODULE_END"
	persisterModelModuleSHAPEKEY        = "SHAPE"
	persisterModelModuleWeightsBeginKey = "WEIGHTS_BEGIN"
	persisterModelModuleWeightsKey      = "WEIGHTS"
	persisterModelModuleBiasesBeginKey  = "BIASES_BEGIN"
	persisterModelModuleBiasesKey       = "BIASES"
)

/*
	THIS IS NOT FUNCTIONAL
*/

func saveModel(model *Model, name string) {
	_ = os.Mkdir(modelStoreRoot, os.ModePerm)

	f, err := os.Create(fmt.Sprintf("%s/%s", modelStoreRoot, name))
	if err != nil {
		log.Fatal("Could not write to model store")
	}
	defer f.Close()

	f.WriteString(modelToString(model))
}

func Restore(name string) *Model {
	_, err := os.Stat(fmt.Sprintf("%s/%s", modelStoreRoot, name))
	if os.IsNotExist(err) {
		log.Fatalf("model %s does not exist", name)
	}

	file, err := os.Open(fmt.Sprintf("%s/%s", modelStoreRoot, name))
	if err != nil {
		log.Fatal("Could not retrieve file from model store")
	}
	defer file.Close()

	model := NewModel()
	scanner := bufio.NewScanner(file)
	var moduleShape mat.Shape
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
		//case persisterModelConfigCriterionKey:
		//	model.configuration.Loss = strings.TrimSpace(split[1])
		//	model.criterion = NewCriterion(model.configuration.Loss)
		//	break
		//case persisterModelConfigOptimizerKey:
		//	model.configuration.Optimizer = strings.TrimSpace(split[1])
		//	model.optimizer = NewOptimizer(model.configuration.Optimizer)
		//	break
		//case persisterModelConfigEpochsKey:
		//	model.configuration.Epochs, _ = strconv.Atoi(strings.TrimSpace(split[1]))
		//	break
		//case persisterModelConfigLearningRateKey:
		//	learningRate64, _ := strconv.ParseFloat(strings.TrimSpace(split[1]), 32)
		//	model.configuration.LearningRate = float32(learningRate64)
		//	break
		case persisterModelModuleKey:
			break
		case persisterModelModuleSHAPEKEY:
			parameters := strings.Split(strings.TrimSpace(split[1]), " ")
			moduleShape.X, _ = strconv.Atoi(parameters[0])
			moduleShape.Y, _ = strconv.Atoi(parameters[1])
			break
		case persisterModelModuleWeightsBeginKey:
			moduleWeights = make([]float32, moduleShape.X*moduleShape.Y)
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
			module := modules.Linear(moduleShape.Y)
			model.Add(module)
			break
		}
	}

	return model
}

func modelToString(model *Model) string {
	content := fmt.Sprintf("%s: %s\n\n", persisterTypeKey, persisterModelType)

	//content += fmt.Sprintf("%s: %s\n", persisterModelConfigCriterionKey, model.configuration.Loss)
	//content += fmt.Sprintf("%s: %s\n", persisterModelConfigOptimizerKey, model.configuration.Optimizer)
	//content += fmt.Sprintf("%s: %d\n", persisterModelConfigEpochsKey, model.configuration.Epochs)
	//content += fmt.Sprintf("%s: %g\n", persisterModelConfigLearningRateKey, model.configuration.LearningRate)
	content += "\n"

	for _, module := range model.modules {
		parameters := module.GetParameters()

		content += fmt.Sprintf("%s: DENSE\n", persisterModelModuleKey)
		content += fmt.Sprintf("%s: %d %d\n", persisterModelModuleSHAPEKEY, parameters[0].Shape().X, parameters[0].Shape().Y)
		content += fmt.Sprintf("%s", persisterModelModuleWeightsBeginKey)
		for i, value := range parameters[0].ToFloat32() {
			if i%maxFloatsPerLine == 0 {
				content += fmt.Sprintf("\n%s: ", persisterModelModuleWeightsKey)
			}
			content += fmt.Sprintf("%g ", value)
		}
		content += fmt.Sprintf("\n%s", persisterModelModuleBiasesBeginKey)
		for i, value := range parameters[1].ToFloat32() {
			if i%maxFloatsPerLine == 0 {
				content += fmt.Sprintf("\n%s: ", persisterModelModuleBiasesKey)
			}
			content += fmt.Sprintf("%g ", value)
		}
		content += fmt.Sprintf("\n%s\n\n", persisterModelModuleEndKey)
	}

	return content
}
