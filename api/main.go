package main

import (
	"github.com/gorilla/mux"
	"log"
	"net/http"
)

func main() {
	r := mux.NewRouter()
	projects := r.PathPrefix("/api/projects").Subrouter()
	projects.HandleFunc("", getProjects).Methods(http.MethodGet)
	log.Fatal(http.ListenAndServe(":8080", r))
}
