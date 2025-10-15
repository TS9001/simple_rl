# Remote Training Guide

## Quick Start

### 1. Start Training on Remote Server

```bash
# Connect to server
ssh user@your-server.com

# Navigate to project
cd /path/to/simple_rl

# Start training with nohup (keeps running after disconnect)
nohup python -u scripts/full_pipeline_sft_grpo_training.py > training.log 2>&1 &

# Save the process ID
echo $! > training.pid
echo "Training started with PID: $(cat training.pid)"

# Disconnect safely
exit
```

### 2. Monitor Training Progress

```bash
# Reconnect to server
ssh user@your-server.com
cd /path/to/simple_rl

# Method 1: View complete log output
tail -f training.log

# Method 2: Watch JSON progress (auto-refresh every 5 seconds)
watch -n 5 cat logs/progress.json

# Method 3: Use formatted monitoring script
python scripts/monitor_training.py

# Method 4: Continuous monitoring (built-in watch mode)
python scripts/monitor_training.py --watch
```

### 3. Check Training Status

```bash
# Check if training is still running
ps aux | grep full_pipeline_sft_grpo_training.py

# Or use saved PID
ps -p $(cat training.pid)

# Check latest metrics
tail -20 training.log

# View progress summary
python scripts/monitor_training.py
```

### 4. Stop Training (if needed)

```bash
# Graceful stop using saved PID
kill $(cat training.pid)

# Force kill if needed (last resort)
kill -9 $(cat training.pid)

# Verify it stopped
ps -p $(cat training.pid)
```

## Output Files

- **training.log**: Complete console output (all print statements)
- **logs/training_*.log**: Structured logging with timestamps
- **logs/progress.json**: Real-time progress tracking (JSON format)
- **checkpoints/grpo_qwen_math/**: Model checkpoints (saved every N episodes)

## Understanding Progress JSON

The `logs/progress.json` file contains:

```json
{
  "status": "training",
  "phase": "grpo_training",
  "start_time": "2025-10-15T10:30:00",
  "last_update": "2025-10-15T11:45:23",
  "elapsed_time_seconds": 4523,
  "current_episode": 45,
  "total_episodes": 500,
  "progress_percent": 9.0,
  "latest_metrics": {
    "pg_loss": 0.0012,
    "kl_divergence": 0.015,
    "reward_mean": 0.67,
    "format_reward_mean": 0.85,
    "correctness_reward_mean": 0.49,
    "grad_norm": 0.082,
    "lr": 1e-05
  },
  "checkpoints": [
    {
      "path": "checkpoints/grpo_qwen_math/checkpoint_episode_15.pt",
      "episode": 15,
      "timestamp": "2025-10-15T11:00:00"
    }
  ],
  "error": null
}
```

## Monitoring Commands

### Tail Log Output
```bash
# Last 50 lines
tail -50 training.log

# Follow in real-time
tail -f training.log

# Search for errors
grep -i error training.log
grep -i warning training.log
```

### Watch Progress
```bash
# Auto-refresh JSON every 5 seconds
watch -n 5 cat logs/progress.json

# Auto-refresh with formatting
watch -n 5 'python scripts/monitor_training.py'
```

### Check Checkpoints
```bash
# List saved checkpoints
ls -lh checkpoints/grpo_qwen_math/

# See checkpoint sizes
du -sh checkpoints/grpo_qwen_math/*
```

## Resuming Training

If training stops or you want to continue from a checkpoint:

```bash
# Edit the script to set CONTINUE_FROM
nano scripts/full_pipeline_sft_grpo_training.py

# Change this line:
# CONTINUE_FROM = 45  # Episode number to resume from

# Restart training
nohup python -u scripts/full_pipeline_sft_grpo_training.py > training_resume.log 2>&1 &
```

## Troubleshooting

### Training Not Running
```bash
# Check if process exists
ps aux | grep full_pipeline_sft_grpo_training.py

# Check for errors in log
tail -100 training.log | grep -i error

# Check GPU/CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Out of Memory
```bash
# Check memory usage
free -h

# Check GPU memory
nvidia-smi

# Consider reducing batch_size in config
```

### Progress Not Updating
```bash
# Check last update time
cat logs/progress.json | grep last_update

# Verify process is running
ps aux | grep full_pipeline_sft_grpo_training.py

# Check if process is hung (CPU usage should be >0%)
top -p $(cat training.pid)
```

## Tips

1. **Use screen or tmux** for better session management:
   ```bash
   screen -S training
   python -u scripts/full_pipeline_sft_grpo_training.py
   # Detach with Ctrl+A, D
   # Reattach with: screen -r training
   ```

2. **Monitor system resources**:
   ```bash
   htop          # CPU and RAM
   nvidia-smi -l 5   # GPU (updates every 5 seconds)
   ```

3. **Backup checkpoints regularly**:
   ```bash
   rsync -av checkpoints/ backup/checkpoints-$(date +%Y%m%d)/
   ```

4. **Download logs/checkpoints locally**:
   ```bash
   # From your local machine
   scp -r user@server:/path/to/simple_rl/checkpoints ./
   scp user@server:/path/to/simple_rl/training.log ./
   scp user@server:/path/to/simple_rl/logs/progress.json ./
   ```

## Example Workflow

```bash
# 1. Start training
ssh user@server
cd /path/to/simple_rl
nohup python -u scripts/full_pipeline_sft_grpo_training.py > training.log 2>&1 &
echo $! > training.pid
exit

# 2. Check progress (from local machine or reconnect)
ssh user@server "cd /path/to/simple_rl && python scripts/monitor_training.py"

# 3. View live log
ssh user@server "tail -f /path/to/simple_rl/training.log"

# 4. Download checkpoints when done
scp -r user@server:/path/to/simple_rl/checkpoints ./local_checkpoints/
```

## Advanced: Using systemd (Optional)

Create a systemd service for automatic restart:

```bash
# Create service file
sudo nano /etc/systemd/system/grpo-training.service

# Add:
[Unit]
Description=GRPO Training Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/simple_rl
ExecStart=/usr/bin/python3 -u scripts/full_pipeline_sft_grpo_training.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/path/to/simple_rl/training.log
StandardError=append:/path/to/simple_rl/training.log

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable grpo-training
sudo systemctl start grpo-training

# Monitor
sudo systemctl status grpo-training
sudo journalctl -u grpo-training -f
```
