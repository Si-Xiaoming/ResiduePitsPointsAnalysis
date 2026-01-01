for i in {0..4}; do
    echo "Submitting task $i..."
    sbatch --export=ALL,MY_TASK_ID=$i backbones.sbatch
done