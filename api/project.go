package main

import (
	"encoding/json"
	"net/http"
)

func getProjects(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	b, _ := json.Marshal([]project{
		{id: "1", name: "projecta"},
		{id: "2", name: "projectb"},
	})
	w.WriteHeader(http.StatusOK)
	w.Write(b)
}