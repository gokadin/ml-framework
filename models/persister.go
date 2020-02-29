package models

import (
	"bufio"
	"fmt"
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/modules"
	"log"
	"os"
	"strconv"
	"strings"
)

const (
	modelStoreRoot                    = ".cache/models"
	persisterTypeKey                  = "TYPE"
	persisterModelType                = "MODEL"
	persisterModelConfigCriterionKey  = "CRITERION"
	persisterModelConfigEpochsKey     = "EPOCHS"
	persisterModelModuleKey           = "MODULE"
	persisterModelModuleEndKey        = "MODULE_END"
	persisterModelModuleSHAPEKEY      = "SHAPE"
	persisterModelModuleActivationKey = "ACTIVATION"
	persisterModelModuleWeightsKey    = "WEIGHTS"
	persisterModelModuleBiasesKey     = "BIASES"
)

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
	reader := bufio.NewReader(file)
	var moduleShape mat.Shape
	var moduleActivationFunction string
	var moduleWeights []float32
	var moduleBiases []float32
	for {
		bytes, _, err := reader.ReadLine()
		if err != nil {
			break
		}

		line := string(bytes)
		split := strings.Split(line, ":")
		if line == "" || len(split) == 0 {
			continue
		}
		switch split[0] {
		case persisterModelConfigCriterionKey:
			model.configuration.Loss = strings.TrimSpace(split[1])
			break
		case persisterModelConfigEpochsKey:
			model.configuration.Epochs, _ = strconv.Atoi(strings.TrimSpace(split[1]))
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
		case persisterModelModuleWeightsKey:
			moduleWeights = make([]float32, moduleShape.X * moduleShape.Y)
			parameters := strings.Split(strings.TrimSpace(split[1]), " ")
			for i, value := range parameters {
				parsed, _ := strconv.ParseFloat(value, 32)
				moduleWeights[i] = float32(parsed)
			}
			break
		case persisterModelModuleBiasesKey:
			moduleBiases = make([]float32, moduleShape.Y)
			parameters := strings.Split(strings.TrimSpace(split[1]), " ")
			for i, value := range parameters {
				parsed, _ := strconv.ParseFloat(value, 32)
				moduleBiases[i] = float32(parsed)
			}
			break
		case persisterModelModuleEndKey:
			module := modules.Dense(moduleShape.Y, moduleActivationFunction)
			module.InitializeWith(mat.NewMat32f(moduleShape, moduleWeights), mat.NewMat32f(mat.WithShape(1, moduleShape.Y), moduleBiases))
			model.Add(module)
			break
		}
	}

	return model
}

func modelToString(model *Model) string {
	content := fmt.Sprintf("%s: %s\n\n", persisterTypeKey, persisterModelType)

	content += fmt.Sprintf("%s: %s\n", persisterModelConfigCriterionKey, model.configuration.Loss)
	content += fmt.Sprintf("%s: %d\n", persisterModelConfigEpochsKey, model.configuration.Epochs)
	content += "\n"

	for _, module := range model.modules {
		parameters := module.GetParameters()

		content += fmt.Sprintf("%s: DENSE\n", persisterModelModuleKey)
		content += fmt.Sprintf("%s: %d %d\n", persisterModelModuleSHAPEKEY, parameters[0].Shape().X, parameters[0].Shape().Y)
		content += fmt.Sprintf("%s: %s\n", persisterModelModuleActivationKey, module.GetActivation())
		content += fmt.Sprintf("%s: ", persisterModelModuleWeightsKey, )
		for _, value := range parameters[0].Data().Data() {
			content += fmt.Sprintf("%g ", value)
		}
		content += fmt.Sprintf("\n%s: ", persisterModelModuleBiasesKey, )
		for _, value := range parameters[1].Data().Data() {
			content += fmt.Sprintf("%g ", value)
		}
		content += fmt.Sprintf("\n%s\n\n", persisterModelModuleEndKey)
	}

	return content
}
