LOCAL_ARGS=(
  "--n_teams" 18
  "--n_rounds" 6
  "--threshold_ranks" 4 12 
  "--threshold_rewards" 30 20 
  "--env_name" "lin_4"  
  "--max_draw_probability" 0.5
  "--max_strength" 4
  "--strength_decay_factor" 1
  "--strength_decay_method" "linear"
  "--agent_id" 12
  "--train_episodes" 32000
  "--test_episodes" 2000
  #"--log_dir" "users/p25003/arnaud/reinforcement_learning/logs"
  #"--history_log_prefix" "cp"
)

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 64 64 \
    --dropout 0.2 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.0003 \
    --epsilon_start 1 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.999 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 64 64 \
    --dropout 0.3 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.0003 \
    --epsilon_start 1 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.999 \


python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 256 128 64 \
    --dropout 0.2 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.0003 \
    --epsilon_start 1 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.999 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 256 64 \
    --dropout 0.2 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.0003 \
    --epsilon_start 1 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.999 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 256 128 64 \
    --dropout 0.3 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.0003 \
    --epsilon_start 1 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.999 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 256 64 \
    --dropout 0.3 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.0003 \
    --epsilon_start 1 \
    --epsilon_end 0.05 \
    --epsilon_decay 0.999 \