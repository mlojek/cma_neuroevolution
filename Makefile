all_code = data models

install:
	pip -r requirements.txt

clean:
	git clean -fdx

format:
	isort ${all_code} --profile black
	black ${all_code}

check:
	black ${all_code} --check
	isort ${all_code} --check --profile black
	pylint ${all_code}