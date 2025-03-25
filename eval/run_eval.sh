# bash eval/run_eval.sh --test_path {dir} --task consistency

which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

python eval/eval.py \
${@:1}