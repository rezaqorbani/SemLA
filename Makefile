PROJECT_NAME=semla

.PHONY: init add-build-dependencies sync clean

venv:
	uv venv --seed --python 3.10

# First instal torch because it is used as a build dependency in Detectron
add-build-dependencies:
	uv pip install torch torchvision

# Install without build isolation so that Detectron can use torch installed in previous step
sync: venv add-build-dependencies 
	uv sync

clean:
	rm -rf .venv uv.lock