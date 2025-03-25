which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

model_type=$1
dset_name=$2
video_root="/data/video_datasets"

case ${dset_name} in
    tvr)
        if [[ ${model_type} == "Gemini" ]]; then
            extra_args+=(--skip)
        elif [[ ${model_type} == "GPT-4o" ]]; then
            n_frames=10
            extra_args+=(--skip)

        elif [[ ${model_type} == "Video-LLaMA" ]]; then
            n_frames=8

        elif [[ ${model_type} == "Video-ChatGPT" ]]; then
            n_frames=100
        elif [[ ${model_type} == "TimeChat" ]]; then
            n_frames=96
        elif [[ ${model_type} == "VTimeLLM" ]]; then
            n_frames=100
        fi
esac

python run.py \
--model_type=${model_type} \
--dset_name=${dset_name} \
--video_root=${video_root} \
--n_frames=${n_frames} \
${extra_args[@]} \
${@:3}