# Test one specific module
python __main__.py --project-path . --module-name services

# Test everything
python __main__.py --project-path . --all-modules

python __main__.py --project-path "D:\study material\project\PyQu\local\b1" --all-modules

python __main__.py --project-path local/b1/before --module-name Orange.evaluation.scoring

python __main__.py --project-path local/b1/before --module-name Orange.evaluation.scoring --force-venv

python __main__.py --project-path local/b1/before --all-modules

python __main__.py --project-path ./local/b1 --all-modules --max-search-time 180 -- --maximum-test-executions 10000

python __main__.py --project-path ./local/b1 --all-modules --algorithm WHOLE_SUITE --max-search-time 180

python __main__.py --project-path ./local/b1/before --all-modules --max-search-time 120

python __main__.py --project-path ./local/b1 --all-modules --max-search-time 120

python __main__.py --project-path ./local/b1 --all-modules

python __main__.py --project-path "D:\study material\project\PyQu\orange3-master\Orange" --all-modules

python __main__.py --project-path "D:\study material\project\PyQu\local\block_0000" --all-modules --force-deps