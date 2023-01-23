#!/usr/bin/env bash
python3 scripts/main.py \
    --task=$TASK \
    --method=$METHOD \
    --gpu_nums=$GPU_NUMS \
    --edit_folder_num=$EDIT_FOLDER_NUM \
    --process_folders=$PROCESS_FOLDERS \
    --task_id=$TASK_ID \
    --tasks_per_gpu=$TASKS_PER_GPU \
    --example_repeat=$EXAMPLE_REPEAT \
    --temp_mode=$TEMP_MODE \
    --get_heat_map=$GET_HEAT_MAP \
    --log_path=$LOG_PATH \
    --log_name=$LOG_NAME \
    --batch_size=$BATCH_SIZE \
    --model_path=$MODEL_PATH \
    --max_edit_step=$MAX_EDIT_STEP \
    --num_workers=$NUM_WORKERS \
    --start_val_epoch=$START_VAL_EPOCH \
    --optim=$OPTIM \
    --lr=$LR \
    --use_init_weight=$USE_INIT_WEIGHT \
    --amplify_v=$AMPLIFY_V \
    --amplify_con=$AMPLIFY_CON \
    --check_val_every_n_epoch=$CHECK_VAL_EVERY_N_EPOCH \
    --memory_loss=$MEMORY_LOSS \
    --mlc=$MLC \
    --update_memory=$UPDATE_MEMORY \
    --margin_val1=$MARGIN_VAL1 \
    --margin_val2=$MARGIN_VAL2 \
    --activate_loss=$ACTIVATE_LOSS \
    --act_loss_thd=$ACT_LOSS_THD \
    --alc=$ALC \
    --act_margin_val=$ACT_MARGIN_VAL \
    --drop_num=$DROP_NUM \
    --drop_rate=$DROP_RATE \
    --freeze_model=$FREEZE_MODEL \
    --max_add_neuron_num=$MAX_ADD_NEURON_NUM \
    --use_val=$USE_VAL \
    --ft_optim=$FT_OPTIM \
    --ft_lr=$FT_LR \
    --use_kl=$USE_KL \
    --alpha=$ALPHA \
    --loc_num=$LOC_NUM \
    --ft_update_memory=$FT_UPDATE_MEMORY \
    --layer=$LAYER \


:<<BLOCK
$TASK: fever or zsre
$METHOD: ft or T-patch
$PROCESS_FOLDERS: all_folders if you want to run all edit folder simultaneously, or you can pass a list containing the folder_num, such as [1,3,5,7,9],
                  or you can pass seg_i_j to run on folder [i,j), such as seg_0_3 means folder 1 and 2


BLOCK


