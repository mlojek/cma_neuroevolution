all_code = **/*.py *.py

install:
	pip install -r requirements.txt

clean:
	git clean -fdx

format:
	isort ${all_code} --profile black
	black ${all_code}

check:
	black ${all_code} --check
	isort ${all_code} --check --profile black
	pylint ${all_code} --disable duplicate-code