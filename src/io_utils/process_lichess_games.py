import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from src.engine.evaluators.simple_nn_eval import ChessNN
from src.io_utils.to_tensor import create_tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Constants
DATA_FILE = "data/raw/lichess_eval/lichess_db_eval.jsonl"
MODEL_PATH = "data/models/simple_nn.pth"
PROGRESS_FILE = "data/interim/lichess_eval/progress.txt"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS_PER_CHUNK = 1
SAVE_EVERY = 1000  # Save every 1000 batches
MAX_EVAL = 20.0


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, cp = self.data[idx]
        try:
            board = chess.Board(fen)
            tensor = create_tensor(board)
        except ValueError:
            # Invalid board, return dummy or skip, but for simplicity, return zeros
            tensor = torch.zeros((28, 8, 8), dtype=torch.float32)
        label = torch.tensor(
            [min(max(cp / 100.0 / MAX_EVAL, -1.0), 1.0)], dtype=torch.float32
        )
        return tensor, label


def get_resume_line():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def update_progress(line_num):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(line_num))


def process_chunk(lines, model, optimizer, device, start_line):
    data = []
    for i, line in enumerate(lines):
        try:
            item = json.loads(line)
            fen = item["fen"]
            # Use the first pv's cp from the first eval
            pv = item["evals"][0]["pvs"][0]
            if "cp" in pv:
                cp = pv["cp"]
            elif "mate" in pv:
                cp = 10000 if pv["mate"] > 0 else -10000
            else:
                continue
            # If it's mate, perhaps handle differently, but for now use cp
            data.append((fen, cp))
        except Exception as e:
            print(f"Error parsing line {start_line + i}: {e}")

    if not data:
        return 0

    dataset = ChessDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS_PER_CHUNK):
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return len(lines)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNN().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    resume_line = get_resume_line()
    print(f"Resuming from line {resume_line}")

    with open(DATA_FILE, "r") as f:
        # Skip to resume line
        for _ in range(resume_line):
            f.readline()

        batch_count = 0
        while True:
            lines = []
            for _ in range(BATCH_SIZE * 10):  # Read a chunk of lines
                line = f.readline()
                if not line:
                    break
                lines.append(line)

            if not lines:
                break

            processed = process_chunk(lines, model, optimizer, device, resume_line)
            resume_line += processed
            batch_count += 1

            if batch_count % SAVE_EVERY == 0:
                torch.save(model.state_dict(), MODEL_PATH)
                update_progress(resume_line)
                print(f"Saved model and progress at line {resume_line}")

    # Final save
    torch.save(model.state_dict(), MODEL_PATH)
    update_progress(resume_line)
    print("Processing complete.")


if __name__ == "__main__":
    main()
