FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install dependencies first (separate layer, cached across code-only changes).
# requirements-api.txt is a minimal runtime subset of requirements.txt - the
# API doesn't need jupyter/shap/matplotlib/seaborn, only what's imported by
# src/api/ and src/features/extract.py at request time.
COPY requirements-api.txt .
# xgboost's Linux wheel pulls in nvidia-nccl-cu13 (~216MB) for multi-GPU
# training support, which this CPU-only inference service never uses;
# uninstalling it after install keeps xgboost's CPU predict path working
# while avoiding shipping an unused ~216MB CUDA library in the image.
RUN uv pip install --system --no-cache -r requirements-api.txt \
    && uv pip uninstall --system nvidia-nccl-cu13

# Copy only what the API needs at runtime: the src package, the selected
# final model artifact + its metadata (both small and committed to git -
# see .gitignore), NOT the raw/full data, notebooks, or the other 3
# candidate model artifacts from Phase 4, which are unused here.
COPY src/ ./src/
COPY models_saved/final_model.joblib models_saved/final_model_metadata.json ./models_saved/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
