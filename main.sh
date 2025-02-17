LOCAL_ARGS=(
  "--n_teams" 18
  "--n_rounds" 6
  "--threshold_ranks" 4 12 
  "--threshold_rewards" 30 20   
  "--max_draw_probability" 0.3
  "--max_strength" 9
  "--strength_decay_factor" 1
  "--strength_decay_method" "linear"
  "--agent_id" 12
  "--train_episodes" 20000
  "--test_episodes" 2000
  "--log_dir" "users/p25003/arnaud/reinforcement_learning/logs"
)

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 256 128 64 \
    --dropout 0.1 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.001 \
    --epsilon_start 1 \
    --epsilon_end 0.02 \
    --epsilon_decay 0.9995 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 512 128 32 \
    --dropout 0.1 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.001 \
    --epsilon_start 1 \
    --epsilon_end 0.02 \
    --epsilon_decay 0.9995 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 512 256 128 64 32 \
    --dropout 0.1 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.001 \
    --epsilon_start 1 \
    --epsilon_end 0.02 \
    --epsilon_decay 0.9995 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 256 128 64 \
    --dropout 0.2 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.001 \
    --epsilon_start 1 \
    --epsilon_end 0.02 \
    --epsilon_decay 0.9995 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 512 128 32 \
    --dropout 0.2 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.001 \
    --epsilon_start 1 \
    --epsilon_end 0.02 \
    --epsilon_decay 0.9995 \

python -u swiss_round_run.py "${LOCAL_ARGS[@]}" \
    --hidden_sizes 512 256 128 64 32 \
    --dropout 0.2 \
    --buffer_size 10000 \
    --batch_size 64 \
    --gamma 1 \
    --lr 0.001 \
    --epsilon_start 1 \
    --epsilon_end 0.02 \
    --epsilon_decay 0.9995 \