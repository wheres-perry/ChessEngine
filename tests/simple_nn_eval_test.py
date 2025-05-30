import random
from pathlib import Path

import chess
import chess.engine
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.engine.evaluators.simple_nn_eval import ChessNN, NN_Eval
from src.io_utils.load_games import random_board
from src.io_utils.to_tensor import create_tensor
from tqdm import tqdm


# Set random seed for reproducibility

random.seed(42)
torch.manual_seed(42)


# Configuration

STOCKFISH_PATH = "/usr/games/stockfish"
NUM_POSITIONS = 10  # Number of positions to use for training
TRAINING_ITERATIONS = 10  # Number of individual training steps
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "chess_nn_model_minimal_test.pth"


def get_stockfish_eval(board, engine, depth=10):
    """Get evaluation from Stockfish for a given position."""
    result = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = result["score"].white().score(mate_score=10000)

    # Normalize score to be between -1 and 1

    if score is not None:
        # Cap extreme values

        score = max(-10000, min(10000, score))
        # Normalize to [-1, 1]

        score = score / 10000
    else:
        # If score is None (e.g., checkmate), use appropriate extreme value

        if board.is_checkmate():
            score = -1.0 if board.turn else 1.0
        else:
            score = 0.0  # Draw
    return score


def generate_training_data(num_positions, engine):
    """Generate training data from random positions with Stockfish evaluations."""
    X = []
    y = []

    print(f"Generating {num_positions} training positions...")
    for _ in tqdm(range(num_positions)):
        # Get a random board position

        board = random_board()
        if board is None:
            continue
        # Get tensor representation

        try:
            tensor = create_tensor(board)
            stockfish_eval = get_stockfish_eval(board, engine)

            X.append(tensor)
            y.append(stockfish_eval)
        except Exception as e:
            print(f"Error processing board: {e}")
            continue
    return torch.stack(X), torch.tensor(y, dtype=torch.float32).view(-1, 1)


def train_model(model, X_train, y_train, iterations, learning_rate):
    """Train the neural network model for a fixed number of iterations on single samples."""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    print(f"Starting minimal training for {iterations} iterations...")

    num_train_samples = X_train.size(0)
    if num_train_samples == 0:
        print("No training samples available. Skipping training.")
        return model, losses
    for i in range(iterations):
        model.train()

        # Select a random sample from the training set

        idx = random.randint(0, num_train_samples - 1)
        sample_X = X_train[idx].unsqueeze(0).to(DEVICE)  # Add batch dimension
        sample_y = y_train[idx].unsqueeze(0).to(DEVICE)  # Add batch dimension

        # Forward pass

        output = model(sample_X)
        loss = criterion(output, sample_y)

        # Backward pass and optimize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Iteration {i+1}/{iterations}, Loss: {loss.item():.6f}")
    print("Minimal training complete.")
    return model, losses


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(DEVICE)
        y_test = y_test.to(DEVICE)

        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test).item()

        # Calculate correlation

        predictions_np = predictions.cpu().numpy().flatten()
        y_test_np = y_test.cpu().numpy().flatten()
        correlation = np.corrcoef(predictions_np, y_test_np)[0, 1]
    return mse, correlation


def plot_results(losses, y_test, predictions):
    """Plot training loss and prediction vs actual values."""
    # plt.figure(figsize=(12, 5))

    # # Plot training loss
    # plt.subplot(1, 2, 1)
    # plt.plot(losses)
    # plt.title('Training Loss per Iteration')
    # plt.xlabel('Iteration')
    # plt.ylabel('MSE Loss')

    # # Plot predictions vs actual
    # plt.subplot(1, 2, 2)
    # plt.scatter(y_test, predictions, alpha=0.5)
    # plt.plot([-1, 1], [-1, 1], 'r--')  # Diagonal line for perfect predictions
    # plt.title('Predictions vs Actual')
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')

    # plt.tight_layout()
    # plt.savefig('training_results.png')
    # plt.show()

    pass


def main():
    # Initialize Stockfish engine

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"Error initializing Stockfish engine: {e}")
        print("Please make sure Stockfish is installed and the path is correct.")
        return
    try:
        # Generate training data

        X, y = generate_training_data(NUM_POSITIONS, engine)

        # Split data into training and testing sets (80/20)

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

        # Initialize and train model

        model = ChessNN()
        trained_model, losses = train_model(
            model,
            X_train,
            y_train,
            iterations=TRAINING_ITERATIONS,
            learning_rate=LEARNING_RATE,
        )

        # Save the trained model
        # torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
        # print(f"Model saved to {MODEL_SAVE_PATH}")

        # Evaluate model

        mse, correlation = evaluate_model(trained_model, X_test, y_test)
        print(f"Test MSE: {mse:.6f}")
        print(f"Correlation: {correlation:.6f}")

        # Make predictions for visualization

        trained_model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(DEVICE)
            predictions = trained_model(X_test_device).cpu().numpy().flatten()
            y_test_np = y_test.cpu().numpy().flatten()
        # Plot results
        # plot_results(losses, y_test_np, predictions)

        # Test the NN_Eval class with the trained model

        test_board = random_board()
        if test_board:
            nn_evaluator = NN_Eval(
                test_board, model_path=None, model_instance=trained_model
            )  # Pass the model instance directly
            eval_score = nn_evaluator.evaluate()
            print(f"NN evaluation for test position: {eval_score}")
            print(f"Board: {test_board}")

            # Compare with Stockfish

            stockfish_score = (
                get_stockfish_eval(test_board, engine) * 20.0
            )  # Scale to match NN_Eval.MAX_EVAL
            print(f"Stockfish evaluation: {stockfish_score}")
    finally:
        # Clean up

        if "engine" in locals():
            engine.quit()


if __name__ == "__main__":
    main()
