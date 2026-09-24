export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"

use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

dataset_name=$(basename "$0" .sh)
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name"

model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
p_target=${SPAQR_P_TARGET:-8}
min_patch_mass=${SPAQR_MIN_PATCH_MASS:-4}
token_budget=${SPAQR_TOKEN_BUDGET:-48}
decoder_layers=${SPAQR_DECODER_LAYERS:-4}
missing_rate=${MISSING_RATE:-0}
train_epochs=${TRAIN_EPOCHS:-300}
itr=${ITR:-5}
pred_lengths=${PRED_LENS:-300}
model_id="${model_name}_fixed_budget_P${p_target}_B${token_budget}_QCSU_D${decoder_layers}"

if [ "$token_budget" -le 0 ]; then
    echo "SPAQR_TOKEN_BUDGET must be a positive fixed budget." >&2
    exit 2
fi

# ────────────────────────────────────────────────────────────────────────
# Fixed-budget hard tokenizer and query-conditioned support unfolding decoder
#   N = 12, seq_len = 3000 ms, pred_len = 300 ms
#   Pmax=8, fixed budget=48, coarse-global read then target-conditioned unfolding.
# ────────────────────────────────────────────────────────────────────────
seq_len=3000
for pred_len in $pred_lengths; do
    $launch_command main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "MSE" \
    --use_multi_gpu $use_multi_gpu \
    --dataset_root_path $dataset_root_path \
    --model_id $model_id \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --dataset_id $dataset_id \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $n_variables \
    --dec_in $n_variables \
    --c_out $n_variables \
    --train_epochs $train_epochs \
    --patience 10 \
    --val_interval 1 \
    --itr $itr \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --missing_rate $missing_rate \
    --spaqr_p_target $p_target \
    --spaqr_min_patch_mass $min_patch_mass \
    --spaqr_token_budget $token_budget \
    --spaqr_count_penalty ${SPAQR_COUNT_PENALTY:-1.0} \
    --spaqr_span_penalty ${SPAQR_SPAN_PENALTY:-1.0} \
    --spaqr_gap_penalty ${SPAQR_GAP_PENALTY:-1.0} \
    --spaqr_d_model 128 \
    --spaqr_num_geometric_freq 6 \
    --spaqr_n_heads 4 \
    --spaqr_decoder_layers $decoder_layers \
    --spaqr_debug ${SPAQR_DEBUG:-0} \
    --spaqr_dropout 0.0
done
