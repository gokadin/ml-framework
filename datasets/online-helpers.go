package datasets

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
)

func downloadFile(baseUrl, filename string) []byte {
	if isCached(filename) {
		return getCachedFile(filename)
	}

	response, err := http.Get(fmt.Sprintf("%s%s", baseUrl, filename))
	if err != nil {
		log.Fatalf("Could not download file %s", filename)
	}
	defer response.Body.Close()

	content, err := ioutil.ReadAll(response.Body)
	if err != nil {
		log.Fatal("Could not decode response message")
	}
	addToCache(filename, content)
	return content
}

func unzip(compressed []byte) []byte {
	compressedReader := bytes.NewReader(compressed)
	reader, err := gzip.NewReader(compressedReader)
	if err != nil {
		log.Fatal("Could not unzip file")
	}
	defer reader.Close()

	uncompressed, _ := ioutil.ReadAll(reader)
	return uncompressed
}