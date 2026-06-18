Генерация искусственных моделей винтов на основе исходного датасета из 100 моделей, рассчет МКЭ и предсказание первых 12-ти собственных частот исходного датасета.

1) Рассчет FEM для искусственных моделей
QT_QPA_PLATFORM=offscreen freecadcmd run_fem_calibrated.py
2) Профиль для исходных винтов
.venv/bin/python analyze_traceparts_profile.py models --out data/traceparts_specs.csv
3) Обучение + кэш
.venv/bin/python train_frequency_model.py train --build-cache \
  --checkpoint checkpoints/freq_predictor_12_v5.pt
4) Предсказание на исходных винтах
.venv/bin/python train_frequency_model.py batch \
  --models-dir models \
  --output results-predicted-12.csv \
  --checkpoint checkpoints/freq_predictor_12_v5.pt \
  --cache-dir data/geometry_cache_v4 \
  --tta-views 8
5) Рассчет MAPE
.venv/bin/python evaluate_predictions.py results-predicted-12.csv results_init2.csv
