PROJECT_NAME=semla

.PHONY: init add-build-dependencies sync clean

venv:
	uv venv --python 3.10

# First instal torch because it is used as a build dependency in Detectron
add-build-dependencies:
	uv add torch torchvision

# Install without build isolation so that Detectron can use torch installed in previous step
sync: venv add-build-dependencies

clean:
	rm -rf .venv uv.lock