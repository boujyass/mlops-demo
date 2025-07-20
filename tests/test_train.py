import os
import subprocess

def test_train_script_runs_and_creates_model():
    # Run the training script
    result = subprocess.run(['python', 'model/train.py'], capture_output=True, text=True)
    assert result.returncode == 0, f"Train script failed with error: {result.stderr}"

    # Check model file exists
    assert os.path.exists('model/model.pkl'), "Model file not found after training"
