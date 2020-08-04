package main

import (
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/simple"
	"io/ioutil"
	"log"
)

func main() {
	//examples.RunMnist()

	backend := simple.NewSimpleGraph()

	model := onnx.NewModel(backend)

	b, _ := ioutil.ReadFile("")
	err := model.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}

}
